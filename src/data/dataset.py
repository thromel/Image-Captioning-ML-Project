import os
import json
import torch
import numpy as np
from PIL import Image
from typing import Dict, List, Tuple, Union, Optional, Any, Callable
from torch.utils.data import Dataset, DataLoader

from transformers import PreTrainedTokenizer


class COCOCaptionDataset(Dataset):
    """
    Dataset for loading COCO captions dataset.
    """

    def __init__(
        self,
        root_dir: str,
        annotation_file: str,
        image_dir: str,
        tokenizer: PreTrainedTokenizer,
        transform: Optional[Callable] = None,
        max_length: int = 50,
        is_training: bool = True
    ):
        """
        Initialize the COCO dataset.

        Args:
            root_dir: Root directory for dataset
            annotation_file: Path to annotation file (relative to root_dir)
            image_dir: Directory containing images (relative to root_dir)
            tokenizer: Tokenizer for processing captions
            transform: Optional transform to apply to images
            max_length: Maximum caption length
            is_training: Whether this is a training dataset
        """
        self.root_dir = root_dir
        self.image_dir = os.path.join(root_dir, image_dir)
        self.annotation_path = os.path.join(root_dir, annotation_file)
        self.tokenizer = tokenizer
        self.transform = transform
        self.max_length = max_length
        self.is_training = is_training

        # Load annotations
        with open(self.annotation_path, 'r') as f:
            self.annotations = json.load(f)

        # Process annotations
        self._process_annotations()

    def _process_annotations(self):
        """Process COCO annotations into a more convenient format."""
        # Create image_id to filename mapping
        self.image_id_to_filename = {}
        for image in self.annotations['images']:
            self.image_id_to_filename[image['id']] = image['file_name']

        # Create list of (image_id, caption) pairs
        self.examples = []
        for annotation in self.annotations['annotations']:
            image_id = annotation['image_id']
            caption = annotation['caption']

            # Skip if image doesn't exist
            if image_id not in self.image_id_to_filename:
                continue

            self.examples.append({
                'image_id': image_id,
                'filename': self.image_id_to_filename[image_id],
                'caption': caption
            })

        # During training, duplicate examples to have one for each caption
        # During evaluation, group examples by image_id
        if not self.is_training:
            # Group captions by image_id
            self.image_examples = {}
            for example in self.examples:
                image_id = example['image_id']
                if image_id not in self.image_examples:
                    self.image_examples[image_id] = {
                        'filename': example['filename'],
                        'captions': []
                    }
                self.image_examples[image_id]['captions'].append(
                    example['caption'])

            # Convert to list of image examples
            self.examples = [
                {
                    'image_id': image_id,
                    'filename': data['filename'],
                    'captions': data['captions']
                }
                for image_id, data in self.image_examples.items()
            ]

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """Get a single example from the dataset."""
        example = self.examples[idx]

        # Load and process image
        image_path = os.path.join(self.image_dir, example['filename'])
        image = Image.open(image_path).convert('RGB')

        if self.transform:
            image = self.transform(image)

        # Process captions
        if self.is_training:
            # For training, return a single caption
            caption = example['caption']

            # Tokenize caption
            encoding = self.tokenizer(
                caption,
                padding='max_length',
                truncation=True,
                max_length=self.max_length,
                return_tensors='pt'
            )

            # Remove batch dimension
            caption_tokens = encoding.input_ids.squeeze(0)
            attention_mask = encoding.attention_mask.squeeze(0)

            return {
                'image': image,
                'caption_tokens': caption_tokens,
                'attention_mask': attention_mask,
                'caption': caption
            }
        else:
            # For evaluation, return all captions
            captions = example['captions']

            # Tokenize all captions
            caption_tokens_list = []
            attention_mask_list = []

            for caption in captions:
                encoding = self.tokenizer(
                    caption,
                    padding='max_length',
                    truncation=True,
                    max_length=self.max_length,
                    return_tensors='pt'
                )

                caption_tokens_list.append(encoding.input_ids.squeeze(0))
                attention_mask_list.append(encoding.attention_mask.squeeze(0))

            # Stack captions if there are multiple
            if caption_tokens_list:
                caption_tokens = torch.stack(caption_tokens_list)
                attention_mask = torch.stack(attention_mask_list)
            else:
                # Fallback for images with no captions
                caption_tokens = torch.zeros(
                    (1, self.max_length), dtype=torch.long)
                attention_mask = torch.zeros(
                    (1, self.max_length), dtype=torch.long)

            return {
                'image': image,
                'caption_tokens': caption_tokens,
                'attention_mask': attention_mask,
                'captions': captions,
                'image_id': example['image_id']
            }


class ObjectDetectionFeaturesDataset(Dataset):
    """
    Dataset for loading pre-extracted object detection features.
    (Bottom-Up Top-Down Attention approach).
    """

    def __init__(
        self,
        features_dir: str,
        annotation_file: str,
        tokenizer: PreTrainedTokenizer,
        max_objects: int = 36,
        max_length: int = 50,
        is_training: bool = True
    ):
        """
        Initialize the dataset with pre-extracted object features.

        Args:
            features_dir: Directory containing object features
            annotation_file: Path to annotation file
            tokenizer: Tokenizer for processing captions
            max_objects: Maximum number of objects per image
            max_length: Maximum caption length
            is_training: Whether this is a training dataset
        """
        self.features_dir = features_dir
        self.annotation_path = annotation_file
        self.tokenizer = tokenizer
        self.max_objects = max_objects
        self.max_length = max_length
        self.is_training = is_training

        # Load annotations
        with open(self.annotation_path, 'r') as f:
            self.annotations = json.load(f)

        # Process annotations
        self._process_annotations()

    def _process_annotations(self):
        """Process annotations into a more convenient format."""
        # Create image_id to filename mapping
        self.image_id_to_filename = {}
        for image in self.annotations['images']:
            # Features are typically stored with image IDs as filenames
            self.image_id_to_filename[image['id']] = f"{image['id']}.npz"

        # Create list of (image_id, caption) pairs
        self.examples = []
        for annotation in self.annotations['annotations']:
            image_id = annotation['image_id']
            caption = annotation['caption']

            # Skip if image doesn't exist
            if image_id not in self.image_id_to_filename:
                continue

            self.examples.append({
                'image_id': image_id,
                'filename': self.image_id_to_filename[image_id],
                'caption': caption
            })

        # During training, duplicate examples to have one for each caption
        # During evaluation, group examples by image_id
        if not self.is_training:
            # Group captions by image_id
            self.image_examples = {}
            for example in self.examples:
                image_id = example['image_id']
                if image_id not in self.image_examples:
                    self.image_examples[image_id] = {
                        'filename': example['filename'],
                        'captions': []
                    }
                self.image_examples[image_id]['captions'].append(
                    example['caption'])

            # Convert to list of image examples
            self.examples = [
                {
                    'image_id': image_id,
                    'filename': data['filename'],
                    'captions': data['captions']
                }
                for image_id, data in self.image_examples.items()
            ]

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """Get a single example from the dataset."""
        example = self.examples[idx]

        # Load pre-extracted features
        features_path = os.path.join(self.features_dir, example['filename'])

        try:
            features_data = np.load(features_path, allow_pickle=True)
            # Shape: [num_objects, feature_dim]
            obj_features = features_data['features']
            # Shape: [num_objects, 4] (x, y, width, height)
            obj_boxes = features_data['boxes']

            # Pad or truncate to max_objects
            num_objects = obj_features.shape[0]

            if num_objects > self.max_objects:
                # Truncate
                obj_features = obj_features[:self.max_objects]
                obj_boxes = obj_boxes[:self.max_objects]
                obj_mask = np.ones(self.max_objects, dtype=np.bool)
            else:
                # Pad
                pad_features = np.zeros(
                    (self.max_objects, obj_features.shape[1]), dtype=np.float32)
                pad_boxes = np.zeros((self.max_objects, 4), dtype=np.float32)
                obj_mask = np.zeros(self.max_objects, dtype=np.bool)

                pad_features[:num_objects] = obj_features
                pad_boxes[:num_objects] = obj_boxes
                obj_mask[:num_objects] = True

                obj_features = pad_features
                obj_boxes = pad_boxes
        except Exception as e:
            # If loading fails, create empty features
            feature_dim = 2048  # Default feature dimension for R-CNN
            obj_features = np.zeros(
                (self.max_objects, feature_dim), dtype=np.float32)
            obj_boxes = np.zeros((self.max_objects, 4), dtype=np.float32)
            obj_mask = np.zeros(self.max_objects, dtype=np.bool)

            print(f"Error loading features for {example['filename']}: {e}")

        # Convert to torch tensors
        obj_features = torch.from_numpy(obj_features)
        obj_boxes = torch.from_numpy(obj_boxes)
        obj_mask = torch.from_numpy(obj_mask)

        # Process captions
        if self.is_training:
            # For training, return a single caption
            caption = example['caption']

            # Tokenize caption
            encoding = self.tokenizer(
                caption,
                padding='max_length',
                truncation=True,
                max_length=self.max_length,
                return_tensors='pt'
            )

            # Remove batch dimension
            caption_tokens = encoding.input_ids.squeeze(0)
            attention_mask = encoding.attention_mask.squeeze(0)

            return {
                'region_features': obj_features,
                'region_boxes': obj_boxes,
                'region_mask': obj_mask,
                'caption_tokens': caption_tokens,
                'attention_mask': attention_mask,
                'caption': caption
            }
        else:
            # For evaluation, return all captions
            captions = example['captions']

            # Tokenize all captions
            caption_tokens_list = []
            attention_mask_list = []

            for caption in captions:
                encoding = self.tokenizer(
                    caption,
                    padding='max_length',
                    truncation=True,
                    max_length=self.max_length,
                    return_tensors='pt'
                )

                caption_tokens_list.append(encoding.input_ids.squeeze(0))
                attention_mask_list.append(encoding.attention_mask.squeeze(0))

            # Stack captions if there are multiple
            if caption_tokens_list:
                caption_tokens = torch.stack(caption_tokens_list)
                attention_mask = torch.stack(attention_mask_list)
            else:
                # Fallback for images with no captions
                caption_tokens = torch.zeros(
                    (1, self.max_length), dtype=torch.long)
                attention_mask = torch.zeros(
                    (1, self.max_length), dtype=torch.long)

            return {
                'region_features': obj_features,
                'region_boxes': obj_boxes,
                'region_mask': obj_mask,
                'caption_tokens': caption_tokens,
                'attention_mask': attention_mask,
                'captions': captions,
                'image_id': example['image_id']
            }


def build_coco_dataloaders(
    config,
    tokenizer: PreTrainedTokenizer,
    transform_train=None,
    transform_val=None,
    use_curriculum: bool = None
):
    """
    Build dataloaders for COCO dataset.

    Args:
        config: Configuration object
        tokenizer: Tokenizer for processing captions
        transform_train: Transform for training images
        transform_val: Transform for validation images
        use_curriculum: Whether to use curriculum learning (overrides config if provided)

    Returns:
        Tuple of (train_loader, val_loader, curriculum_sampler)
    """
    # Create datasets
    train_dataset = COCOCaptionDataset(
        root_dir=config.data_root,
        annotation_file=config.train_json,
        image_dir=config.train_image_dir,
        tokenizer=tokenizer,
        transform=transform_train,
        max_length=config.model.decoder.max_length,
        is_training=True
    )

    val_dataset = COCOCaptionDataset(
        root_dir=config.data_root,
        annotation_file=config.val_json,
        image_dir=config.val_image_dir,
        tokenizer=tokenizer,
        transform=transform_val,
        max_length=config.model.decoder.max_length,
        is_training=False
    )

    # Check if curriculum learning should be used
    if use_curriculum is None:
        use_curriculum = config.training.use_curriculum

    # Create curriculum sampler if requested
    curriculum_sampler = None
    if use_curriculum:
        try:
            from ..train.curriculum import create_curriculum_sampler
            curriculum_sampler = create_curriculum_sampler(train_dataset, config)
        except ImportError:
            print("Warning: Curriculum learning module not available, using standard sampling")

    # Create dataloaders
    if curriculum_sampler is not None:
        # Use curriculum sampler (no shuffle when using custom sampler)
        train_loader = DataLoader(
            train_dataset,
            batch_size=config.training.batch_size,
            sampler=curriculum_sampler,
            num_workers=config.num_workers,
            pin_memory=True
        )
    else:
        # Standard random sampling
        train_loader = DataLoader(
            train_dataset,
            batch_size=config.training.batch_size,
            shuffle=True,
            num_workers=config.num_workers,
            pin_memory=True
        )

    val_loader = DataLoader(
        val_dataset,
        batch_size=config.inference.num_candidates,  # Use num_candidates for validation
        shuffle=False,
        num_workers=config.num_workers,
        pin_memory=True
    )

    return train_loader, val_loader, curriculum_sampler
