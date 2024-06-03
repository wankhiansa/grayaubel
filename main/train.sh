#!/bin/bash

# task=classification  # target is a binary value (e.g., drug or not).
# dataset=hiv

task=regression
dataset=coronary_artery_disease

radius=1
dim=50
layer_hidden=4
layer_output=4

batch_train=40
batch_test=40
lr=1e-3
lr_decay=0.99
decay_interval=10
iteration=100

homo_lumo_dim=3
mlp_hidden_dim=64
mlp_output_dim=32


setting=$dataset--radius$radius--dim$dim--layer_hidden$layer_hidden--layer_output$layer_output--batch_train$batch_train--batch_test$batch_test--lr$lr--lr_decay$lr_decay--decay_interval$decay_interval--iteration$iteration--homo_lumo_dim$homo_lumo_dim--mlp_hidden_dim$mlp_hidden_dim--mlp_output_dim$mlp_output_dim
python train.py $task $dataset $radius $dim $layer_hidden $layer_output $batch_train $batch_test $lr $lr_decay $decay_interval $weight_decay $iteration $homo_lumo_dim $mlp_hidden_dim $mlp_output_dim $setting
