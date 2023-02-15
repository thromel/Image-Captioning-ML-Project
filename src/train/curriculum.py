"""
Curriculum learning strategies for image captioning.

This module implements various curriculum learning approaches that
progressively increase training difficulty by sorting samples from
easy to hard based on different criteria.
"""

import torch
import numpy as np
from typing import List, Dict, Any, Optional, Tuple
from torch.utils.data import Sampler, Dataset
import logging


class CurriculumSampler(Sampler):
    """
    Curriculum learning sampler that orders training samples from easy to hard.

    The difficulty of samples is determined based on a specified strategy:
    - caption_length: Shorter captions are easier
    - num_objects: Fewer objects are easier
    - clip_score: Higher CLIP scores (better image-text alignment) are easier
    """

    def __init__(
        self,
        dataset: Dataset,
        strategy: str = "caption_length",
        num_epochs: int = 15,
        warmup_epochs: int = 3,
        difficulty_scores: Optional[List[float]] = None,
        shuffle_within_bins: bool = True,
        num_bins: int = 10
    ):
        """
        Initialize the curriculum sampler.

        Args:
            dataset: The training dataset
            strategy: Curriculum strategy ('caption_length', 'num_objects', 'clip_score')
            num_epochs: Total number of training epochs
            warmup_epochs: Number of epochs to use for curriculum (rest is random)
            difficulty_scores: Pre-computed difficulty scores (optional)
            shuffle_within_bins: Whether to shuffle samples within difficulty bins
            num_bins: Number of difficulty bins to divide samples into
        """
        super().__init__(dataset)
        self.dataset = dataset
        self.strategy = strategy
        self.num_epochs = num_epochs
        self.warmup_epochs = warmup_epochs
        self.shuffle_within_bins = shuffle_within_bins
        self.num_bins = num_bins
        self.current_epoch = 0

        self.logger = logging.getLogger(__name__)
        self.logger.info(f"Initializing curriculum sampler with strategy: {strategy}")

        # Compute or use pre-computed difficulty scores
        if difficulty_scores is not None:
            self.difficulty_scores = difficulty_scores
        else:
            self.difficulty_scores = self._compute_difficulty_scores()

        # Sort indices by difficulty (ascending - easy to hard)
        self.sorted_indices = np.argsort(self.difficulty_scores)

    def _compute_difficulty_scores(self) -> List[float]:
        """
        Compute difficulty scores for all samples in the dataset.

        Returns:
            List of difficulty scores (lower = easier)
        """
        self.logger.info(f"Computing difficulty scores using strategy: {self.strategy}")
        difficulty_scores = []

        for idx in range(len(self.dataset)):
            sample = self.dataset[idx]

            if self.strategy == "caption_length":
                # Shorter captions are easier
                if isinstance(sample, dict) and 'caption_tokens' in sample:
                    caption_tokens = sample['caption_tokens']
                    if isinstance(caption_tokens, torch.Tensor):
                        # Count non-padding tokens
                        pad_token_id = getattr(self.dataset, 'pad_token_id', 0)
                        length = (caption_tokens != pad_token_id).sum().item()
                    else:
                        length = len(caption_tokens)
                elif isinstance(sample, dict) and 'caption' in sample:
                    length = len(sample['caption'].split())
                else:
                    # Default difficulty if we can't determine length
                    length = 10

                difficulty_scores.append(float(length))

            elif self.strategy == "num_objects":
                # Fewer objects are easier
                # This requires pre-computed object counts
                if isinstance(sample, dict) and 'num_objects' in sample:
                    num_objects = sample['num_objects']
                else:
                    # Default: use a random score or neutral difficulty
                    num_objects = 5  # Neutral value

                difficulty_scores.append(float(num_objects))

            elif self.strategy == "clip_score":
                # Higher CLIP scores are easier (better alignment)
                # This requires pre-computed CLIP scores
                if isinstance(sample, dict) and 'clip_score' in sample:
                    clip_score = sample['clip_score']
                    # Invert so higher score = lower difficulty
                    difficulty = 1.0 / (clip_score + 1e-8)
                else:
                    # Default: neutral difficulty
                    difficulty = 1.0

                difficulty_scores.append(float(difficulty))

            else:
                # Unknown strategy - use neutral difficulty (random order)
                self.logger.warning(f"Unknown strategy '{self.strategy}', using neutral difficulty")
                difficulty_scores.append(float(idx))

        return difficulty_scores

    def set_epoch(self, epoch: int):
        """
        Set the current epoch for the sampler.

        Args:
            epoch: Current epoch number
        """
        self.current_epoch = epoch

    def __iter__(self):
        """
        Generate sample indices based on curriculum strategy.

        Returns:
            Iterator over sample indices
        """
        # After warmup epochs, use random sampling
        if self.current_epoch >= self.warmup_epochs:
            # Random sampling after curriculum phase
            indices = np.random.permutation(len(self.dataset)).tolist()
            return iter(indices)

        # During curriculum learning, gradually introduce harder samples
        # Calculate how many samples to include based on current epoch
        progress = (self.current_epoch + 1) / self.warmup_epochs
        num_samples_to_include = int(progress * len(self.dataset))

        # Ensure we include at least some samples
        num_samples_to_include = max(num_samples_to_include, len(self.dataset) // 10)

        # Get the easiest samples
        curriculum_indices = self.sorted_indices[:num_samples_to_include].copy()

        if self.shuffle_within_bins:
            # Divide into bins and shuffle within each bin
            bin_size = len(curriculum_indices) // self.num_bins
            if bin_size > 0:
                shuffled_indices = []
                for i in range(self.num_bins):
                    start_idx = i * bin_size
                    end_idx = start_idx + bin_size if i < self.num_bins - 1 else len(curriculum_indices)
                    bin_indices = curriculum_indices[start_idx:end_idx].copy()
                    np.random.shuffle(bin_indices)
                    shuffled_indices.extend(bin_indices)
                curriculum_indices = np.array(shuffled_indices)
            else:
                np.random.shuffle(curriculum_indices)
        else:
            np.random.shuffle(curriculum_indices)

        return iter(curriculum_indices.tolist())

    def __len__(self):
        """
        Return the number of samples.

        Returns:
            Number of samples in the dataset
        """
        # During curriculum, return only the samples we're using
        if self.current_epoch < self.warmup_epochs:
            progress = (self.current_epoch + 1) / self.warmup_epochs
            num_samples = int(progress * len(self.dataset))
            return max(num_samples, len(self.dataset) // 10)
        else:
            return len(self.dataset)


class PacingFunction:
    """
    Pacing function for curriculum learning that determines what portion
    of the data to use at each training step.
    """

    @staticmethod
    def linear(epoch: int, total_epochs: int) -> float:
        """
        Linear pacing: gradually increase from 0% to 100%.

        Args:
            epoch: Current epoch
            total_epochs: Total number of epochs

        Returns:
            Proportion of data to use (0.0 to 1.0)
        """
        return min(1.0, (epoch + 1) / total_epochs)

    @staticmethod
    def root(epoch: int, total_epochs: int, power: float = 2.0) -> float:
        """
        Root pacing: slower increase at the beginning.

        Args:
            epoch: Current epoch
            total_epochs: Total number of epochs
            power: Power for the root function

        Returns:
            Proportion of data to use (0.0 to 1.0)
        """
        progress = (epoch + 1) / total_epochs
        return min(1.0, progress ** (1.0 / power))

    @staticmethod
    def exponential(epoch: int, total_epochs: int, rate: float = 2.0) -> float:
        """
        Exponential pacing: faster increase at the beginning.

        Args:
            epoch: Current epoch
            total_epochs: Total number of epochs
            rate: Rate of exponential growth

        Returns:
            Proportion of data to use (0.0 to 1.0)
        """
        progress = (epoch + 1) / total_epochs
        return min(1.0, progress ** rate)

    @staticmethod
    def step(epoch: int, total_epochs: int, num_steps: int = 3) -> float:
        """
        Step pacing: discrete steps in data inclusion.

        Args:
            epoch: Current epoch
            total_epochs: Total number of epochs
            num_steps: Number of discrete steps

        Returns:
            Proportion of data to use (0.0 to 1.0)
        """
        progress = (epoch + 1) / total_epochs
        step_size = 1.0 / num_steps
        step = int(progress / step_size)
        return min(1.0, (step + 1) * step_size)


def create_curriculum_sampler(
    dataset: Dataset,
    config: Any,
    difficulty_scores: Optional[List[float]] = None
) -> Optional[CurriculumSampler]:
    """
    Factory function to create a curriculum sampler based on config.

    Args:
        dataset: Training dataset
        config: Training configuration
        difficulty_scores: Pre-computed difficulty scores (optional)

    Returns:
        CurriculumSampler if curriculum learning is enabled, None otherwise
    """
    if not config.training.use_curriculum:
        return None

    return CurriculumSampler(
        dataset=dataset,
        strategy=config.training.curriculum_strategy,
        num_epochs=config.training.num_epochs,
        warmup_epochs=min(5, config.training.num_epochs // 3),  # Use first 1/3 for curriculum
        difficulty_scores=difficulty_scores,
        shuffle_within_bins=True,
        num_bins=10
    )
