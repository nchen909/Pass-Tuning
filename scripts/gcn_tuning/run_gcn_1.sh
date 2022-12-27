#!/bin/bash

model_lst=(plbart)
#(roberta codebert graphcodebert codet5 plbart unixcoder)
# cuda=0
cd ../..
for model in "${model_lst[@]}"; do

    #NLU
    # CUDA_VISIBLE_DEVICES=0 bash run_prefix.sh $model defect
    # CUDA_VISIBLE_DEVICES=0 bash run_prefix.sh $model clone

    # # #you can indeed run defect in main.py(run_gen.py),
    # # # in which 'em' refers to eval_acc * 100, and bleu=codebleu=0

    # # #NLG
    # CUDA_VISIBLE_DEVICES=0 bash run_prefix.sh $model translate java-cs
    # CUDA_VISIBLE_DEVICES=0 bash run_prefix.sh $model translate cs-java
    # CUDA_VISIBLE_DEVICES=0 bash run_prefix.sh $model refine small
    # CUDA_VISIBLE_DEVICES=0 bash run_prefix.sh $model refine medium
    # CUDA_VISIBLE_DEVICES=0 bash run_prefix.sh $model generate

    
    CUDA_VISIBLE_DEVICES=0 bash run_gcn_tuning.sh $model summarize ruby
    # CUDA_VISIBLE_DEVICES=0 bash run_prefix.sh $model summarize javascript
    # CUDA_VISIBLE_DEVICES=0 bash run_prefix.sh $model summarize go
    # CUDA_VISIBLE_DEVICES=0 bash run_prefix.sh $model summarize python
    # CUDA_VISIBLE_DEVICES=0 bash run_prefix.sh $model summarize java
    # CUDA_VISIBLE_DEVICES=0 bash run_prefix.sh $model summarize php

done