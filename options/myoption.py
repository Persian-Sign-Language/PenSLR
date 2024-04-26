from models.cnnlstm import M2 as M
from path_managers import PathManager

import torch
import os

# dataset directory
dataset_directory = os.path.join("data/dataset1-3")


# features list
features_list = [
  "acc_x",
  "acc_y",
  "acc_z",
  "line_acc_x",
  "line_acc_y",
  "line_acc_z",
  "gyro_x",
  "gyro_y",
  "gyro_z",
  "gravity_x",
  "gravity_y",
  "gravity_z",
  "flex_0",
  "flex_1",
  "flex_2",
  "flex_3",
  "flex_4",
]

# mode: ['train', 'test', 'feature']
mode = 'train'

# all the recorders
recorders = ['Subject5', 'Subject3', 'Subject2', 'Subject1', 'Subject4']

# random seed
random_seed = 42

# who do you want to train with
recorders_train = recorders[:-1]

# who do you want to test with
recorders_test = recorders[-1:]

# train val ratio (if 0.2 -> 20% of data will be used for validation)
train_val_ratio = 0.2

# k-fold ratio (if you want k-fold validation set it to 1/k)
k_fold_ratio = 0.2

# fold index 
fold_index = 4

# learning rate
learning_rate = 1e-4

# regularization factor
regularization_factor = 0.01

# train batch size
train_batch_size = 9

# epoch
epoch = 1000

# optimizer
optim = torch.optim.AdamW

# verbose (Epochs to update tqdm bar stats)
epoch_verbose = 10

# eval (Epochs for perform evaluation and updating tensorboard data)
epoch_eval = 1

# A custom lr scheduler (default is constant learning rate)
def lr_scheduler(lr, ep, losses): 
    return lr

# loss threshold (if the model's loss reaches below this the learning process stops)
loss_thresh = 0.05

# save_epoch (epochs to save model and matplotlib diagrams)
save_epoch = 10 

# name of the experiment (if None a new experiment will be created)
exp_name = None
model_name = M.__name__

# Visualization Tool
visualization = ['tensorboard', 'matplotlib'][1] #values: ['tensorboard', 'matplotlib']

# train min and max word length
train_min_word_length = 1
train_max_word_length = 3

# test min and max word length
test_min_word_length = 1
test_max_word_length = 3

path_manager = PathManager(
    exp_dir='exp',
    exp_path=exp_name,
    model_name=model_name
)
