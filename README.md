# CodePrompt

## Environment & Preparing

```shell
conda create --name cat python=3.7
conda activate cat
pip install -r requirements.txt
git clone https://github.com/nchen909/CodePrompt
cd CodePrompt/evaluator/CodeBLEU/parser
bash build.sh
cd ../../../
cp evaluator/CodeBLEU/parser/my-languages.so build/
#make sure git-lfs installed like 'apt-get install git-lfs'
bash get_models.sh
```

## Preparing data

The dataset comes from [CodeSearchNet](https://github.com/microsoft/CodeXGLUE/tree/main/Code-Text/code-to-text) .

### Preparing local path

Direct WORKDIR in run.sh, run_few_shot.sh to your path.

## Finetune

```bash
export MODEL_NAME=
export TASK=
export SUB_TASK=
# to run one task
bash run.sh $MODEL_NAME $TASK $SUB_TASK
# to run few shot
bash run_few_shot.sh $MODEL_NAME $TASK $SUB_TASK
# to run multi task
bash run_multi_task.sh
```

  `MODEL_NAME` can be any one of `["roberta", "codebert", "graphcodebert", "unixcoder","t5","codet5","bart","plbart"]`.

  `TASK` can be any one of `['summarize', 'translate', 'refine', 'generate', 'defect', 'clone']`. (generate refers concode in codexglue, and we don't consider complete)

  `SUB_TASK` can be in picture below

![image-20221014233118653](https://pic.mathskiller909.com/img/20221027202855.png?x-oss-process=style/nchen909)
