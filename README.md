# Building Small Language Models from Scratch

<a href="https://www.linkedin.com/in/prashant-lakhera-696119b/"><img src="https://img.shields.io/badge/Follow%20Prashant%20Lakhera-blue.svg?logo=linkedin"></a>


Welcome! In this repository you will find the code for all examples throughout the book **Building Small Language Models from Scratch: A Practical Guide** written by [Prashant Lakhera](https://www.linkedin.com/in/prashant-lakhera-696119b/) which we call: <br> 

<p align="center"><b><i>"The Illustrated Guide to Building LLMs from Scratch"</i></b></p>

Through hands-on implementation and detailed explanations, learn how to build a complete language model from the ground up, implementing every component yourself,from attention mechanisms to training loops.

<a href="https://github.com/ideaweaver-ai/Building-Small-Language-Model-from-Scratch-A-Practical-Guide-Book"><img src="https://raw.githubusercontent.com/ideaweaver-ai/Building-Small-Language-Model-from-Scratch-A-Practical-Guide-Book/main/images/Book_image.jpeg" width="50%" ></a>

<br>

The book is available on:

* [Amazon](https://www.amazon.com/) *(Coming soon)*
* [Gumroad](https://gumroad.com/) *(https://plakhera.gumroad.com/l/BuildingASmallLanguageModelfromScratch)*
* [LeanPub](https://leanpub.com/) *(https://leanpub.com/buildingasmalllanguagemodelfromscratch )*


## About This Book

This book is designed for anyone who wants to understand language models at a fundamental level. Unlike books that show you how to use existing models, this guide teaches you how to build them yourself. By the end, you'll have implemented a working 283M parameter Qwen3 model trained on the TinyStories dataset.

<details>
<summary><b>💡 Tip: A Note on Pace and Foundation</b></summary>

This is a comprehensive book of approximately **854 pages**. We strongly encourage you to take your time going through each chapter and ensure you complete the initial chapters to build a strong foundation before rushing into building the model. 

The early chapters on neural networks, PyTorch, and fundamental concepts are essential for understanding everything that follows. Building a solid foundation will make the later chapters on attention mechanisms, transformers, and model implementation much clearer and more meaningful.

</details>

### Key Features

* **Complete Implementation**: Build every component from scratch: attention mechanisms, positional encodings, feed-forward networks, and more
* **Modern Architecture**: Implement Qwen3 with Grouped Query Attention (GQA), RoPE, RMSNorm, and SwiGLU
* **Practical Training**: Train a real model on the TinyStories dataset with full training loops and optimization
* **Comprehensive Coverage**: From neural network basics to advanced topics like quantization and Mixture of Experts
* **Real Code**: All examples are production-ready and can be run on Google Colab or your local GPU

## Table of Contents

We recommend running all examples through Google Colab for the easiest setup. All examples were developed and tested using Google Colab, ensuring stability and compatibility.

| Chapter | Notebook |
|---|---|
| Chapter 0: Building from Scratch | |
| Chapter 1: Understanding Neural Networks: The Foundations of Modern AI | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1i1hacZZUGzoRE3luDE2KtS--honPnoa8?usp=sharing)|
| Chapter 2: PyTorch Fundamentals: The Building Blocks of Deep Learning | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1tfuMwnzsfZQ4ptFb7rxjLPowviyGZOKw?usp=sharing) |
| Chapter 3: GPUs: The Computational Engine Behind LLM Training | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1ku7QjGNO_ZuksHXd6UidKySS-TgtgWC4?usp=sharing) |
| Chapter 4: Where Intelligence Comes From: A Deep Look at Data | |
| Chapter 5: Understanding Language Models: From Foundations to Small-Scale Design | |
| Chapter 6: Tokenizer: How Language Models Break Text into Meaningful Units | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/13o8x0AVXUgiMsr85kI9pGGTqLuY4JUOZ?usp=sharing) |
| Chapter 7: Understanding Embeddings, Positional Encodings, and RoPE | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/13o8x0AVXUgiMsr85kI9pGGTqLuY4JUOZ?usp=sharing) |
| Chapter 8: Understanding Attention: From Self-Attention to Multi-Head Attention | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1Ux1qrHL5DII8088tmTc4tCJfHqt2zvlw?usp=sharing) |
| Chapter 9: Making Inference Fast: KV Cache, Multi-Query, and Grouped-Query Attention | |
| Chapter 10: Inside the Transformer Block: RMSNorm, SwiGLU, and Residual Connections | |
| Chapter 11: Building Qwen from Scratch | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/16IyYGf_z5IRjcVKwxa5yiXDEMiyf0u1d?usp=sharing) |
| Chapter 12: Quantization | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1cY_DSNZUErhgFAFapCd7pGjh4rBzQcIV?usp=sharing) |
| Chapter 13: Mixture of Experts | |
| Chapter 14: Training Small Language Models: A Practical Journey | |

> [!TIP]
> You can check the repository for setup guides and installation instructions. All code examples are designed to work with Python 3.12+ and PyTorch. For GPU training, we recommend at least 8GB VRAM, though training is also possible on CPU (significantly slower).

## Quick Start

### Prerequisites

* **Hardware**: A computer with at least 16GB RAM. For GPU training (recommended), you'll need a GPU with at least 8GB VRAM (e.g., RTX 3060, RTX 3070, RTX 4090, or A100)
* **Software**: Python 3.12 or higher (tested with Python 3.12.12)
* **Libraries**: PyTorch, transformers, datasets, and other dependencies
* **Knowledge**: Basic familiarity with Python programming and fundamental machine learning concepts

### Training the Model

The complete implementation for building and training the Qwen3-283M model is available:

* **GitHub**: [qwen3-283M-model.py](https://github.com/ideaweaver-ai/Building-Small-Language-Model-from-Scratch-A-Practical-Guide-Book/blob/main/qwen3-283M-model.py)
* **Google Colab**: [Interactive Training Notebook](https://colab.research.google.com/drive/16IyYGf_z5IRjcVKwxa5yiXDEMiyf0u1d?usp=sharing)

**Expected Training Time** (on NVIDIA A100 GPU):
* Training Time: ~5-6 hours for a complete training run
* Memory Usage: ~8GB VRAM
* Final Loss: Typically reaches 2.5-3.0 on TinyStories

## Model Architecture

The book guides you through building a Qwen3-based model with the following specifications:

* **Model Size**: ~283M parameters
* **Architecture**: Qwen3 with GQA, SwiGLU, RoPE
* **Context Length**: 32,768 tokens
* **Vocabulary**: 151,646 tokens (Qwen3 vocabulary)
* **Layers**: 18 transformer blocks
* **Attention Heads**: 4 heads with 1 KV group (Grouped Query Attention)
* **Hidden Dimension**: 640
* **Feed Forward Dimension**: 2,048

## Dataset

We use the **TinyStories dataset** for training:

* **Total Size**: 2.14 million examples (2.12M training, 22k validation)
* **Dataset Size**: Approximately 1GB
* **Content**: Synthetically generated short stories designed for 3-4 year old reading level
* **Vocabulary**: Small, controlled vocabulary of approximately 1,500 basic words
* **Source**: [Hugging Face Dataset](https://huggingface.co/datasets/roneneldan/TinyStories)
* **Paper**: [arXiv:2305.07759](https://arxiv.org/abs/2305.07759)

## What You'll Learn

By working through this book, you'll gain:

* **Deep Understanding**: Understand how transformers work at a fundamental level, not just how to use them
* **Practical Skills**: Learn to collect and process training data, implement training loops, handle tokenization, and manage memory constraints
* **Real Implementation**: Build every component yourself—attention, normalization, feed-forward networks, and the complete architecture
* **Modern Techniques**: Implement state-of-the-art components like GQA, RoPE, RMSNorm, and SwiGLU
* **Training Expertise**: Understand the complete training pipeline from data preprocessing to model evaluation

## Contributing

We welcome contributions! If you find errors, have suggestions, or want to improve the code, please:

1. Open an issue describing the problem or enhancement
2. Submit a pull request with your changes
3. Ensure all code follows the existing style and includes appropriate comments

## Feedback and Support

If you have any questions, need corrections, or have feedback to share, please reach out to us at **help@ideaweaver.ai**. We value your feedback and are always looking to improve.

## Citation

If you find this book useful for your research or learning, please consider citing it:

```bibtex
@book{building-small-llm-from-scratch,
  title        = {Building Small Language Models from Scratch: A Practical Guide},
  author       = {Prashant Lakhera },
  year         = {2025},
  url          = {https://github.com/ideaweaver-ai/Building-Small-Language-Model-from-Scratch-A-Practical-Guide-Book},
  note         = {A comprehensive guide to building language models from scratch}
}
```

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

Thank you to all the readers who have taken the time to work through this comprehensive guide. Building language models from scratch is a challenging journey, and we hope this book has provided you with the knowledge and skills to build your own models.

**Happy Building!** 🚀

