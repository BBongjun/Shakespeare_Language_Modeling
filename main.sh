#!/usr/bin/env bash

for epoch in 100; do
        for n_layers in 2 4; do
        log_file="./log/ln_layers_${n_layers}_epochs_${epoch}.txt"
        
        python main.py --n_layers $n_layers --num_epochs $epoch | tee $log_file
        done
done 
