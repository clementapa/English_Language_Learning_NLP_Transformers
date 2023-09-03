# Feedback Prize - English Language Learning

Evaluating language knowledge of ELL students from grades 8-12

> Authors: [Apavou Clément](https://github.com/clementapa)

![Python](https://img.shields.io/badge/Python-green.svg?style=plastic)
![PyTorch](https://img.shields.io/badge/PyTorch-orange.svg?style=plastic)
![PyTorch Lightning](https://img.shields.io/badge/PyTorch-Lightning-blueviolet.svg?style=plastic)

## :mag_right: Introduction
This code was build for this kaggle competition: https://www.kaggle.com/competitions/feedback-prize-english-language-learning/overview

***Goal of the Competition***
> The goal of this competition is to assess the language proficiency of 8th-12th grade English Language Learners (ELLs). Utilizing a dataset of essays written by ELLs will help to develop proficiency models that better supports all students.
> Your work will help ELLs receive more accurate feedback on their language development and expedite the grading cycle for teachers. These outcomes could enable ELLs to receive more appropriate learning tasks that will help them improve their English language proficiency.

:dart: ***Goal of this project for me*** \
For me, the goal of this project was: 
- to train myself to respond to an NLP task (here it is a NLP multi-regression task) 
- to handle transformers fine tuning by implementing and applying optimization training techniques using limited computational resources. 
- to see all the methods used to facilitate and optimize transformers fine tuning. Such as :
  - Freezing
  - Gradient Accumulation
  - Automatic Mixed Precision
  - Gradient checkpointing
  - etc ... (see the category *useful resources*)
- Understanding and manipulating transformer pretrained / fine tuned embeddings 

## :chart_with_upwards_trend: Weights and Biases training experiments

Experiments are available on wandb: [![](https://github.com/wandb/assets/blob/main/wandb-github-badge-gradient.svg)](https://wandb.ai/clementapa/ELL?workspace=user-clementapa)

## :books: Useful resources 
### Transformers optimization fine tuning and inference
- https://www.kaggle.com/code/rhtsingh/utilizing-transformer-representations-efficiently/notebook
- https://www.kaggle.com/code/vad13irt/optimization-approaches-for-transformers
- https://www.kaggle.com/code/rhtsingh/on-stability-of-few-sample-transformer-fine-tuning/notebook
- https://www.kaggle.com/code/rhtsingh/all-the-ways-you-can-compress-transformers
- https://www.kaggle.com/code/rhtsingh/speeding-up-transformer-w-optimization-strategies/notebook
- https://www.kaggle.com/code/rhtsingh/swa-apex-amp-interpreting-transformers-in-torch/notebook
- https://www.kaggle.com/code/rhtsingh/guide-to-huggingface-schedulers-differential-lrs/notebook

### Training notebooks
- https://www.kaggle.com/code/vslaykovsky/lb-0-43-ensemble-of-top-solutions
- https://www.kaggle.com/code/yasufuminakama/fb3-deberta-v3-base-baseline-train
- https://www.kaggle.com/code/yasufuminakama/fb3-deberta-v3-base-baseline-inference

### Great visualisation
- https://www.kaggle.com/code/lextoumbourou/feedback3-eda-hf-custom-trainer-sift#Text-Examples

### :tada: Implementation To DO / done
- [x] metrics
- [x] parser
- [x] test option

- [x] Deberta v3 base: BERT with disentangled attention better than BERT and RoBERTa models. 1) disentangled attention mechanism, where each word is represented using two vectors that encode its content and position, respectively, and the attention weights among words are computed using disentangled matrices on their contents and relative positions, respectively. 2) an enhanced mask decoder is used to incorporate absolute positions in the decoding layer to predict the masked tokens in model pre-training.

- [x] different way to use pretrained transformers embeddings
- [x] SmoothL1Loss
- [x] Layer normalization
- [x] Mean pooling 
- [x] MultilabelStratifiedKFold split of the data
- [x] Last layer reinitialization or partially reinitialzation -> retrain last encoder layers more task specific
- [x] Layer-wise LR decay to reduce overfitting (https://www.kaggle.com/code/rhtsingh/on-stability-of-few-sample-transformer-fine-tuning/notebook) -> LLRD is a method that applies higher learning rates for top layers and lower learning rates for bottom layers. This is accomplished by setting the learning rate of the top layer and using a multiplicative decay rate to decrease the learning rate layer-by-layer from top to bottom. The goal is to modify the lower layers that encode more general information less than the top layers that are more specific to the pre-training task. This method is adopted in fine-tuning several recent pre-trained models, including XLNet and ELECTRA.
- [x] gradient checkpointing -> increases computation time but it helps to fit larger batch size in single pass.
- [x] StochasticWeightAveraging -> make your models generalize better at virtually no additional cost. The SWA procedure smooths the loss landscape thus making it harder to end up in a local minimum during optimization, Leads to Wider Optima and Better Generalization, SWA produces an ensemble by combining weights of the same network at different stages of training and then uses this model with combined weights to make predictions.
- [ ] concatenate embeddings of several models
- [X] take the last 4 layers hidden states of DeBERTa, take MeanPool of them to gather information along the sequence axis, then take WeightedLayerPool with a set of trainable weights to gather information along the depth axis of the model,
