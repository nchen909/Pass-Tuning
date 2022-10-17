# CodePrompt

## Train

```bash
export MODEL_NAME=
export TASK=
export SUB_TASK=
bash run.sh $MODEL_NAME $TASK $SUB_TASK
```

  `MODEL_NAME` can be any one of `["roberta", "codebert", "graphcodebert", "unixcoder"]`.

  `TASK` can be any one of `['summarize', 'translate', 'refine', 'generate', 'defect', 'clone']`.

  `SUB_TASK` can be in picture below

![image-20221014233118653](https://pic.mathskiller909.com/img/20221014233118.png?x-oss-process=style/mathskiller)
