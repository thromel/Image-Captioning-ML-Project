# Image Captioning using "Show, Attend and Tell"

This repository contains the implementation of an Image Captioning model based on the paper ["Show, Attend and Tell"](https://arxiv.org/abs/1502.03044) by Kelvin Xu et al. The model is trained to generate captions for given images using an attention-based encoder-decoder network.

## About the Model

"Show, Attend and Tell" is a neural network model for image captioning that uses an attention-based encoder-decoder architecture. The model first encodes the image using a Convolutional Neural Network (CNN), which is followed by a recurrent neural network (RNN) that generates the caption word by word. Additionally, the model uses an attention mechanism that allows it to focus on different parts of the image at different time steps, resulting in more accurate and detailed captions.

### Architecture Overview

The project implements:

1. **Encoder**: A CNN-based feature extractor (ResNet-101) that processes input images
2. **Attention Mechanism**: Allows the model to focus on different parts of the image
3. **Decoder**: An LSTM-based sequence generator with optional BERT embeddings

## Dataset

The model is trained on the MS-COCO (Microsoft Common Objects in Context) dataset, which consists of images and corresponding captions. The dataset contains over 330,000 images with 5 captions each, resulting in a total of 1.6 million captions. We have used a preprocessed version of the dataset that has been converted into numerical representations.

## Features

- **Attention Visualization**: Visualize which parts of the image the model focuses on
- **BERT Integration**: Option to use BERT embeddings for enhanced semantic understanding
- **Checkpoint System**: Ability to resume training from checkpoints
- **Comprehensive Evaluation**: BLEU score metrics for caption quality assessment

## Dependencies

The code is implemented in Python 3.7 using the following libraries:

- PyTorch 1.9.0
- torchvision 0.10.0
- NumPy 1.19.5
- NLTK 3.6.3
- Pillow 8.3.2
- pycocotools
- pytorch-pretrained-bert

## Project Structure

```
Project Structure
├── models/
│   ├── encoder.py - ResNet-based image encoder
│   ├── decoder.py - LSTM-based caption decoder with attention
│   ├── loss.py - Loss tracking class
│   └── constants.py - Token constants (PAD, START, END, UNK)
├── data_loader.py - Data loading and processing utilities
├── processData.py - Dataset preparation and vocabulary handling
├── train.py - Training script
├── validate.py - Validation script
├── demo.py - Inference script for caption generation
└── demo.ipynb - Jupyter notebook for interactive demonstration
```

## Usage

### Training the Model

To train the model, run the following command:

```bash
python train.py
```

This will start the training process and save the model weights after each epoch in the checkpoints folder. The training process includes:

- Gradient clipping at 5.0
- Batch size of 16
- Learning rate (decoder) of 0.0004
- 4 training epochs
- Adam optimizer
- Learning rate decay of 0.8 every 1000 batches
- Checkpoint saving

### Validation

To validate the model on the validation set, run:

```bash
python validate.py
```

This will evaluate the model on the validation set and report BLEU scores.

### Generating Captions

To generate captions for new images, run:

```bash
python demo.py --image_path <path_to_image_file>
```

Replace `<path_to_image_file>` with the path to the image file for which you want to generate the caption. The script will:

1. Preprocess the image
2. Feed it to the trained model
3. Generate a caption for the image
4. Optionally visualize attention weights

## Technical Documentation

For comprehensive technical details about the architecture, implementation, and workflows, please see the [Technical Architecture Documentation](docs/technical_architecture.md).

## Performance Considerations

- GPU acceleration recommended for training
- BERT embeddings provide better semantic understanding but increase computational requirements
- Attention mechanism adds computational overhead but significantly improves caption quality
- Batch size may need adjustment based on available GPU memory

## Future Enhancements

1. Ensemble Models
2. Transformer-based architectures
3. Fine-tuning on domain-specific datasets
4. Beam search for inference
5. REST API service integration

## Acknowledgments

We would like to thank Kelvin Xu et al. for their paper "Show, Attend and Tell" and the MS-COCO dataset for providing the data for this project. We have also used some code snippets from the PyTorch tutorial on Image Captioning.

## References

1. Xu, K., Ba, J., Kiros, R., Cho, K., Courville, A., Salakhudinov, R., Zemel, R., & Bengio, Y. (2015). Show, Attend and Tell: Neural Image Caption Generation with Visual Attention. *Proceedings of the 32nd International Conference on Machine Learning*.
2. MS-COCO Dataset: https://cocodataset.org/
3. BERT: Devlin, J., Chang, M.W., Lee, K., & Toutanova, K. (2018). BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding. *arXiv preprint arXiv:1810.04805*.
