import os
import json
import numpy as np
from typing import Dict, List, Tuple, Optional, Any, Union, Callable

# Optional: Import metrics from pycocoevalcap if available
try:
    from pycocoevalcap.tokenizer.ptbtokenizer import PTBTokenizer
    from pycocoevalcap.bleu.bleu import Bleu
    from pycocoevalcap.meteor.meteor import Meteor
    from pycocoevalcap.rouge.rouge import Rouge
    from pycocoevalcap.cider.cider import Cider
    from pycocoevalcap.spice.spice import Spice
    PYCOCOEVALCAP_AVAILABLE = True
except ImportError:
    PYCOCOEVALCAP_AVAILABLE = False
    print("Warning: pycocoevalcap not available. Using placeholder metrics.")


def calculate_metrics(
    generated_captions: List[str],
    reference_captions: List[List[str]],
    image_ids: Optional[List[int]] = None
) -> Dict[str, float]:
    """
    Calculate evaluation metrics for generated captions.

    Args:
        generated_captions: List of generated caption strings
        reference_captions: List of lists of reference caption strings
        image_ids: Optional list of image IDs

    Returns:
        Dictionary of metric scores
    """
    if PYCOCOEVALCAP_AVAILABLE:
        return calculate_metrics_pycocoevalcap(
            generated_captions, reference_captions, image_ids
        )
    else:
        return calculate_metrics_placeholder(
            generated_captions, reference_captions, image_ids
        )


def calculate_metrics_pycocoevalcap(
    generated_captions: List[str],
    reference_captions: List[List[str]],
    image_ids: Optional[List[int]] = None
) -> Dict[str, float]:
    """
    Calculate metrics using pycocoevalcap.

    Args:
        generated_captions: List of generated caption strings
        reference_captions: List of lists of reference caption strings
        image_ids: Optional list of image IDs

    Returns:
        Dictionary of metric scores
    """
    # Create image IDs if not provided
    if image_ids is None:
        image_ids = list(range(len(generated_captions)))

    # Prepare data for pycocoevalcap
    gts = {}
    res = {}

    for i, (gen_caption, ref_captions) in enumerate(zip(generated_captions, reference_captions)):
        image_id = image_ids[i]

        # Add references
        gts[image_id] = []
        for ref_caption in ref_captions:
            gts[image_id].append({"caption": ref_caption})

        # Add generated caption
        res[image_id] = [{"caption": gen_caption}]

    # Tokenize captions
    tokenizer = PTBTokenizer()
    gts = tokenizer.tokenize(gts)
    res = tokenizer.tokenize(res)

    # Calculate scores
    scorers = [
        (Bleu(4), ["Bleu_1", "Bleu_2", "Bleu_3", "Bleu_4"]),
        (Meteor(), "METEOR"),
        (Rouge(), "ROUGE_L"),
        (Cider(), "CIDEr"),
    ]

    # Add SPICE if needed (it's slower)
    if os.environ.get("CALCULATE_SPICE", "0") == "1":
        scorers.append((Spice(), "SPICE"))

    # Calculate scores
    scores = {}

    for scorer, method in scorers:
        score, scores_per_image = scorer.compute_score(gts, res)

        if isinstance(method, list):
            for sc, m in zip(score, method):
                scores[m] = sc
        else:
            scores[method] = score

    return scores


def calculate_metrics_placeholder(
    generated_captions: List[str],
    reference_captions: List[List[str]],
    image_ids: Optional[List[int]] = None
) -> Dict[str, float]:
    """
    Placeholder metrics when pycocoevalcap is not available.
    This provides dummy implementations for basic understanding.

    Args:
        generated_captions: List of generated caption strings
        reference_captions: List of lists of reference caption strings
        image_ids: Optional list of image IDs

    Returns:
        Dictionary of metric scores
    """
    scores = {
        "Bleu_1": calculate_bleu(generated_captions, reference_captions, 1),
        "Bleu_4": calculate_bleu(generated_captions, reference_captions, 4),
        "METEOR": 0.0,  # Placeholder
        "ROUGE_L": 0.0,  # Placeholder
        "CIDEr": 0.0,    # Placeholder
    }

    return scores


def calculate_bleu(
    generated_captions: List[str],
    reference_captions: List[List[str]],
    n: int
) -> float:
    """
    Simple BLEU-N implementation for when pycocoevalcap is not available.

    Args:
        generated_captions: List of generated caption strings
        reference_captions: List of lists of reference caption strings
        n: N-gram size

    Returns:
        BLEU-N score
    """
    scores = []

    for gen_caption, ref_captions in zip(generated_captions, reference_captions):
        # Tokenize
        gen_tokens = gen_caption.split()
        ref_token_lists = [ref.split() for ref in ref_captions]

        # Calculate n-gram precision
        gen_ngrams = _get_ngrams(gen_tokens, n)
        ref_ngrams_list = [_get_ngrams(ref_tokens, n)
                           for ref_tokens in ref_token_lists]

        # Count how many generated n-grams match with references
        matches = 0
        for gen_ngram in gen_ngrams:
            # Check if this n-gram exists in any reference
            if any(gen_ngram in ref_ngrams for ref_ngrams in ref_ngrams_list):
                matches += 1

        # Calculate precision
        precision = matches / max(1, len(gen_ngrams))
        scores.append(precision)

    # Return average score
    return sum(scores) / max(1, len(scores))


def _get_ngrams(tokens: List[str], n: int) -> List[Tuple[str, ...]]:
    """
    Get n-grams from a list of tokens.

    Args:
        tokens: List of tokens
        n: N-gram size

    Returns:
        List of n-grams (as tuples)
    """
    return [tuple(tokens[i:i+n]) for i in range(len(tokens) - n + 1)]


class COCOEvalCap:
    """
    Wrapper class for COCO evaluation using pycocoevalcap.
    Use this for formal evaluation against COCO test set.
    """

    def __init__(self, coco, cocoRes):
        """
        Initialize with COCO and COCO results objects.

        Args:
            coco: COCO dataset object
            cocoRes: COCO results object
        """
        self.evalImgs = []
        self.eval = {}
        self.imgToEval = {}
        self.coco = coco
        self.cocoRes = cocoRes
        self.params = {'image_id': self.coco.getImgIds()}

    def evaluate(self):
        """Evaluate using COCO metrics."""
        # SPICE evaluation is slow, so you may want to make it optional
        use_spice = os.environ.get("CALCULATE_SPICE", "0") == "1"

        # Set up scorers
        scorers = [
            (Bleu(4), ["Bleu_1", "Bleu_2", "Bleu_3", "Bleu_4"]),
            (Meteor(), "METEOR"),
            (Rouge(), "ROUGE_L"),
            (Cider(), "CIDEr"),
        ]

        if use_spice:
            scorers.append((Spice(), "SPICE"))

        # Prepare captions
        gts = self.coco.loadImgs(self.params['image_id'])
        gts = {img['id']: self.coco.imgToAnns[img['id']] for img in gts}
        res = {img_id: self.cocoRes.imgToAnns[img_id] for img_id in gts.keys(
        ) if img_id in self.cocoRes.imgToAnns}

        # Tokenize
        tokenizer = PTBTokenizer()
        gts = tokenizer.tokenize(gts)
        res = tokenizer.tokenize(res)

        # Evaluate
        for scorer, method in scorers:
            print('Computing %s score...' % (method))
            score, scores = scorer.compute_score(gts, res)

            if isinstance(method, list):
                for sc, m in zip(score, method):
                    self.setEval(sc, m)
                    print("%s: %0.3f" % (m, sc))
            else:
                self.setEval(score, method)
                print("%s: %0.3f" % (method, score))

        # Save image-level scores
        for img_id in gts.keys():
            self.imgToEval[img_id] = {'image_id': img_id}

        self.params['image_id'] = list(self.imgToEval.keys())

        return self.eval

    def setEval(self, score, method):
        """Set evaluation scores."""
        self.eval[method] = score


def evaluate_model_on_coco(
    model,
    coco_dataloader,
    tokenizer,
    device: str = 'cuda',
    annotation_file: str = 'captions_val2014.json'
):
    """
    Evaluate model on COCO dataset using official metrics.

    Args:
        model: Image captioning model
        coco_dataloader: DataLoader for COCO images
        tokenizer: Tokenizer for decoding captions
        device: Device to run model on
        annotation_file: Path to COCO annotation file

    Returns:
        Evaluation metrics
    """
    # Import COCO API
    try:
        from pycocotools.coco import COCO
        from pycocotools.cocoeval import COCOeval
    except ImportError:
        print("Error: pycocotools is required for COCO evaluation")
        return None

    # Set model to evaluation mode
    model.eval()

    # Initialize COCO API with ground truth annotations
    coco = COCO(annotation_file)

    # Collect generated captions
    results = []

    for batch in coco_dataloader:
        # Move batch to device
        batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v
                 for k, v in batch.items()}

        # Generate captions
        with torch.no_grad():
            captions, _ = model.generate(
                images=batch['image'],
                max_length=50  # Adjust as needed
            )

        # Decode captions
        decoded_captions = [tokenizer.decode(caption, skip_special_tokens=True)
                            for caption in captions]

        # Add to results
        for i, caption in enumerate(decoded_captions):
            image_id = batch['image_id'][i].item()
            results.append({
                'image_id': image_id,
                'caption': caption
            })

    # Save results to a JSON file
    results_file = 'results.json'
    with open(results_file, 'w') as f:
        json.dump(results, f)

    # Load results with COCO API
    cocoRes = coco.loadRes(results_file)

    # Create evaluator
    cocoEval = COCOEvalCap(coco, cocoRes)

    # Evaluate
    metrics = cocoEval.evaluate()

    return metrics
