#!/bin/bash

model_lst=(roberta codebert graphcodebert unixcoder)
# 'bart', 'plbart', 't5', 'codet5'
alg=L-BFGS-B
cuda=0

for model in "${model_lst[@]}"; do
    bash run.sh $model summarize python
    bash run.sh $model summarize java
    bash run.sh $model summarize javascript
    bash run.sh $model summarize go
    bash run.sh $model translate cs-java
    bash run.sh $model translate java-cs
    bash run.sh $model refine
    bash run.sh $model generate
    bash run.sh $model defect
    bash run.sh $model clone
done
