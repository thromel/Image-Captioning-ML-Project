# Modern Image Captioning System

This repository contains an implementation of a modern image captioning system with various architecture options including transformers, self-attention mechanisms, and pretrained language models.

## Features

- **Modular Architecture**: Mix and match encoders, decoders, and attention mechanisms
- **Multiple Vision Encoders**: ResNet, ViT, Swin Transformer, CLIP
- **Multiple Decoders**: LSTM, Transformer, GPT-2
- **Advanced Attention**: Soft Attention, Multi-Head Attention, Adaptive Attention, Attention-on-Attention
- **Reinforcement Learning**: Self-Critical Sequence Training support
- **Modern Techniques**: BLIP-2 style Q-Former, contrastive learning, curriculum learning

## Installation

Clone the repository and install the required dependencies:

```bash
git clone https://github.com/your-username/Image-Captioning-ML-Project.git
cd Image-Captioning-ML-Project
pip install -r requirements.txt
```

## Dataset Preparation

The system expects the COCO dataset in the following structure:

```
data/
  ├── train2014/       # Training images
  ├── val2014/         # Validation images
  └── annotations/
      ├── captions_train2014.json  # Training annotations
      └── captions_val2014.json    # Validation annotations
```

Download the COCO dataset from [the official website](https://cocodataset.org/#download) and extract it to the `data` directory.

## Training

To train the model with default settings:

```bash
python src/main.py --mode train --data_root /path/to/data
```

To use a specific encoder, decoder, and attention mechanism:

```bash
python src/main.py --mode train --data_root /path/to/data \
                   --encoder_type vit --decoder_type gpt2 --attention_type multi_head
```

To use reinforcement learning:

```bash
python src/main.py --mode train --data_root /path/to/data --use_rl
```

To save and load a configuration:

```bash
python src/main.py --mode train --data_root /path/to/data --save_config config.json
python src/main.py --mode train --config config.json
```

## Evaluation

To evaluate a trained model:

```bash
python src/main.py --mode eval --data_root /path/to/data --checkpoint /path/to/checkpoint.pth
```

## Demo

To run a demo with a single image:

```bash
python src/main.py --mode demo --checkpoint /path/to/checkpoint.pth --image_path /path/to/image.jpg
```

## Configuration

The model can be customized through a variety of configuration options. See `src/config.py` for all available options.

Key configuration categories:
- Encoder options (type, feature dimension, pretrained model)
- Decoder options (type, number of layers, hidden dimensions)
- Attention options (type, number of heads, temperature)
- Training options (batch size, learning rate, reinforcement learning)
- Inference options (beam size, decoding strategy, reranking)

## Architecture Details

The system follows a modular encoder-decoder architecture:

1. **Vision Encoder**: Extracts features from images using modern vision models
2. **Query-Former (Optional)**: Transforms visual features into a fixed set of query tokens
3. **Attention Mechanism**: Computes attention between decoder states and visual features
4. **Text Decoder**: Generates captions based on visual features

## Citation

If you use this code in your research, please cite:

```
@misc{modern-image-captioning-2023,
  author = {Your Name},
  title = {Modern Image Captioning System},
  year = {2023},
  publisher = {GitHub},
  journal = {GitHub repository},
  howpublished = {\url{https://github.com/your-username/Image-Captioning-ML-Project}}
}
```

## License

This project is licensed under the MIT License - see the LICENSE file for details.
