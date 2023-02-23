#!/bin/bash

model_lst=(codet5)
#(roberta codebert graphcodebert codet5 plbart unixcoder)
# cuda=0
cd ../..
for model in "${model_lst[@]}"; do

    #NLU
    # CUDA_VISIBLE_DEVICES=0 bash run_pass_tuning.sh $model clone

    # CUDA_VISIBLE_DEVICES=0 bash run_pass_tuning.sh $model defect
    # # #you can indeed run defect in main.py(run_gen.py),
    # # # in which 'em' refers to eval_acc * 100, and bleu=codebleu=0

    # # #NLG
    # CUDA_VISIBLE_DEVICES=0 bash run_pass_tuning.sh $model translate java-cs
    # CUDA_VISIBLE_DEVICES=0 bash run_pass_tuning.sh $model translate cs-java
    CUDA_VISIBLE_DEVICES=0 bash run_pass_tuning.sh $model refine small
    # CUDA_VISIBLE_DEVICES=0 bash run_pass_tuning.sh $model refine medium
    # CUDA_VISIBLE_DEVICES=0 bash run_pass_tuning.sh $model generate

    
    
    # CUDA_VISIBLE_DEVICES=0 bash run_pass_tuning.sh $model summarize ruby
    # CUDA_VISIBLE_DEVICES=0 bash run_pass_tuning.sh $model summarize javascript
    # CUDA_VISIBLE_DEVICES=0 bash run_pass_tuning.sh $model summarize go
    # CUDA_VISIBLE_DEVICES=0 bash run_pass_tuning.sh $model summarize python
    # CUDA_VISIBLE_DEVICES=0 bash run_pass_tuning.sh $model summarize java
    # CUDA_VISIBLE_DEVICES=0 bash run_pass_tuning.sh $model summarize php

done