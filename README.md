# To DO
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

https://www.kaggle.com/code/rhtsingh/utilizing-transformer-representations-efficiently/notebook
https://www.kaggle.com/code/vad13irt/optimization-approaches-for-transformers
https://www.kaggle.com/code/rhtsingh/on-stability-of-few-sample-transformer-fine-tuning/notebook
https://www.kaggle.com/code/rhtsingh/all-the-ways-you-can-compress-transformers
https://www.kaggle.com/code/rhtsingh/speeding-up-transformer-w-optimization-strategies/notebook
https://www.kaggle.com/code/rhtsingh/swa-apex-amp-interpreting-transformers-in-torch/notebook
https://www.kaggle.com/code/rhtsingh/guide-to-huggingface-schedulers-differential-lrs/notebook

https://www.kaggle.com/code/vslaykovsky/lb-0-43-ensemble-of-top-solutions
https://www.kaggle.com/code/yasufuminakama/fb3-deberta-v3-base-baseline-train
https://www.kaggle.com/code/yasufuminakama/fb3-deberta-v3-base-baseline-inference
Great visualisation:
https://www.kaggle.com/code/lextoumbourou/feedback3-eda-hf-custom-trainer-sift#Text-Examples

solve layer_wise_lr_decay overfit
