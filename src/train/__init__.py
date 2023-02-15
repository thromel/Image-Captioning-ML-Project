"""Training module for image captioning."""

from .trainer import CaptioningTrainer
from .losses import ContrastiveLoss, ImageTextMatchingLoss, CombinedLoss
from .curriculum import CurriculumSampler, create_curriculum_sampler

__all__ = [
    'CaptioningTrainer',
    'ContrastiveLoss',
    'ImageTextMatchingLoss',
    'CombinedLoss',
    'CurriculumSampler',
    'create_curriculum_sampler',
]
