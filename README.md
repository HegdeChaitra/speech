# Attention model for document classification

## About this project
This repo is for training the attention model for classification task. It's the implementation of the paper - https://www.cs.cmu.edu/~./hovy/papers/16HLT-hierarchical-attention-networks.pdf . Instead of considering the Hierarchical attention, only word attention is implemented. 

## Dataset
The input to the model is Bombora's Hollow dataset. Before feeding the model to training, pre-processing is necessary. 
Step by step instriction on how to pre-process the Hollow dataset can be seen in `initial_data_processing.ipynb` notebook.
- The Hollow train dataset is split into train and val in the ration of 0.8 and 0.2 for the purpose of training and validating the model. 
- label is encoded between 0 to n_class-1. 
- The tokens are lower cased, stop words are removed and tokens are mappend to index and this mapping is saved in a pickle file for future reference.

## Description of files
1. def_attn_model_word_eff.py : Defines the word-level attention model
2. define_process_hollow.py : Defines the dunction to preprocess the dataset
3. define_word_training.py : Define the train and evaluation functions
4. train_attention.py : Script to do end-to-end training and evaluation

## Training
The script `train_attention.py` contains script to train and evaluate the attention model. The input to the script is the processed hollow dataset. 
Use `python train_attention.py --help` to see options for training and evaluating the model. 

```
usage: train_attention.py [-h] [--max_len MAX_LEN] [--bs BS]
                          [--n_vocab N_VOCAB] [--lr LR] [--n_epoch N_EPOCH]
                          [--do_train] [--do_eval] [--source_path SOURCE_PATH]
                          [--train_data TRAIN_DATA] [--val_data VAL_DATA]
                          [--token_id TOKEN_ID] [--emb_dim EMB_DIM]
                          [--save_path SAVE_PATH] [--model_load MODEL_LOAD]
                          --model_save MODEL_SAVE

optional arguments:
  -h, --help            show this help message and exit
  --max_len MAX_LEN     Maximum length of the sentence. default=500
  --bs BS               Batch size. default=32
  --n_vocab N_VOCAB     Vocabulary size. default=50000
  --lr LR               Learning rate. default=5e-4
  --n_epoch N_EPOCH     Number of epochs. default=30
  --do_train            Whether to run training. default=False
  --do_eval             Whether to run eval on the dev set. default=False
  --source_path SOURCE_PATH
                        Path to dataset. default=./processed_data/
  --train_data TRAIN_DATA
                        Name of train dataset. default=idized_hollow_train.csv
  --val_data VAL_DATA   Name of val dataset. default=idized_hollow_val.csv
  --token_id TOKEN_ID   Name of token&id pickle file. default=token_and_id.pk
  --emb_dim EMB_DIM     embedding dimension. default=200
  --save_path SAVE_PATH
                        Directory where models are to be saved
  --model_load MODEL_LOAD
                        Complete path to the Model to load for continued
                        training/ evaluation. default=None
  --model_save MODEL_SAVE
                        Name of the model to be saved. required=True
```

### Example of model training 
```
python train_attention.py \
--do_train \
--model_save "sample_model" \
--n_epoch 20
```

## Model Evaluation
- `train_attention.py` script can be used for model evaluation aswell. The model evaluation outputs top-1, top-5 and top-10 accuracy with time taken to run the script.
- `Evaluating_all_model.ipynb` compares top-1, top-5, top-10 accuracy and time complexity of the attention model with fasttext and convolution model.
- `printing_attentions.py` prints sample output of the attention model. It also displays top 10 attended words for that particular examples and corresponding attention weights.

Hollow validation accuracy is as follows:
```
time taken= 106.20869135856628
top1_acc 69.29092671973872
top5_acc 83.40732802612779
top10_acc 86.71284956113493
```

The hollow test accuracy is as follows:
```
time taken= 15.06
top1_acc 74.96
top5_acc 85.14
top10_acc 87.51
```

### Example of model evaluation
``` 
python train_attention.py \
--do_eval \
--model_load "./saved_models/model_att5"
```
