#!/bin/bash

model_lst=(roberta codebert graphcodebert codet5 plbart unixcoder)

cuda=0

for model in "${model_lst[@]}"; do
    #NLU
    bash run.sh $model clone

    bash run.sh $model defect
    #you can indeed run defect in main.py(run_gen.py),
    # in which 'em' refers to eval_acc * 100, and bleu=codebleu=0

    #NLG
    bash run.sh $model summarize ruby
    bash run.sh $model summarize javascript
    bash run.sh $model summarize go
    bash run.sh $model summarize python
    bash run.sh $model summarize java
    bash run.sh $model summarize php

    
    bash run.sh $model translate java-cs
    bash run.sh $model translate cs-java
    bash run.sh $model refine small
    bash run.sh $model refine medium
    bash run.sh $model generate

done



# for model in "${model_lst[@]}"; do
#     #NLG
#     echo "${model} summarize"
#     bash run.sh $model summarize python
#     bash run.sh $model summarize java
#     bash run.sh $model summarize javascript
#     bash run.sh $model summarize go
#     bash run.sh $model summarize ruby
#     bash run.sh $model summarize php
#     echo "${model} translate"
#     bash run.sh $model translate java-cs
#     bash run.sh $model translate cs-java
#     echo "${model} refine"
#     bash run.sh $model refine small
#     bash run.sh $model refine medium
#     echo "${model} generate"
#     bash run.sh $model generate

#     #NLU
#     echo "${model} defect"
#     bash run.sh $model defect
#     #you can indeed run defect in main.py(run_gen.py),
#     # in which 'em' refers to eval_acc * 100, and bleu=codebleu=0
#     echo "${model} clone"
#     bash run.sh $model clone
# done
