WORKDIR="Pass-Tuning"
HUGGINGFACE_LOCALS="Pass-Tuning/data/huggingface_models/"

export PYTHONPATH=$WORKDIR


MODEL_NAME=${1}
#codebert
TASK=${2}
#summarize
SUB_TASK=${3:-none}
#python

DATA_NUM=-1
MODEL_DIR=save_models
SUMMARY_DIR=tensorboard
FULL_MODEL_TAG=${MODEL_NAME}

if [[ ${SUB_TASK} == none ]]; then
  OUTPUT_DIR=${MODEL_DIR}/${TASK}/${FULL_MODEL_TAG}
  RES_DIR=results/${TASK}/${FULL_MODEL_TAG}
  RES_FN=results/${TASK}/${FULL_MODEL_TAG}.txt
else
  OUTPUT_DIR=${MODEL_DIR}/${TASK}/${SUB_TASK}/${FULL_MODEL_TAG}
  RES_DIR=results/${TASK}/${SUB_TASK}/${FULL_MODEL_TAG}
  RES_FN=results/${TASK}/${SUB_TASK}/${FULL_MODEL_TAG}.txt
fi

CACHE_DIR=${WORKDIR}/.cache/${TASK}/${SUB_TASK}/${FULL_MODEL_TAG}
LOG=${OUTPUT_DIR}/train.log
mkdir -p ${OUTPUT_DIR}
mkdir -p ${CACHE_DIR}
mkdir -p ${RES_DIR}

RUN_FN=${WORKDIR}/main.py

if [[ ${SUB_TASK} == none ]]; then
  # CUDA_VISIBLE_DEVICES=0 \
  TOKENIZERS_PARALLELISM=false \
    python ${RUN_FN} ${MULTI_TASK_AUG} \
    --do_test --do_train --do_eval --do_eval_bleu --save_last_checkpoints --always_save_model \
    --task ${TASK} --model_name ${MODEL_NAME} --data_num ${DATA_NUM}  \
    --output_dir ${OUTPUT_DIR}  --summary_dir ${SUMMARY_DIR} --huggingface_locals ${HUGGINGFACE_LOCALS}\
    --data_dir ${WORKDIR}/data  --prefix_tuning 'GCN' --knowledge_usage 'separate' --old_prefix_dir ${WORKDIR}/data_prefix --cache_path ${CACHE_DIR} --res_dir ${RES_DIR} --res_fn ${RES_FN} \
    2>&1 | tee ${LOG}
else
  # CUDA_VISIBLE_DEVICES=0 \
  TOKENIZERS_PARALLELISM=false \
    python ${RUN_FN} ${MULTI_TASK_AUG} \
    --do_test --do_train --do_eval --do_eval_bleu --save_last_checkpoints --always_save_model \
    --task ${TASK} --sub_task ${SUB_TASK} --model_name ${MODEL_NAME} --data_num ${DATA_NUM}  \
    --output_dir ${OUTPUT_DIR}  --summary_dir ${SUMMARY_DIR} --huggingface_locals ${HUGGINGFACE_LOCALS}\
    --data_dir ${WORKDIR}/data  --prefix_tuning 'GCN' --knowledge_usage 'separate' --old_prefix_dir ${WORKDIR}/data_prefix --cache_path ${CACHE_DIR} --res_dir ${RES_DIR} --res_fn ${RES_FN} \
    2>&1 | tee ${LOG}
fi
