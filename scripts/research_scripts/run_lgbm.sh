#!/bin/bash
set -e

echo "===== RUN LGBM REGRESSION ====="

DATA_PATH="/home_ext/zzx/work_file/ATEL-Correction-Analysis-of-Recipe/raw_data/df_filtered_outlier.csv"
MODEL_DIR="/home_ext/zzx/work_file/ATEL-Correction-Analysis-of-Recipe/models_result/lgbm"
RESULT_DIR="/home_ext/zzx/work_file/ATEL-Correction-Analysis-of-Recipe/results/lgbm_eval"
LOG_DIR="/home_ext/zzx/work_file/ATEL-Correction-Analysis-of-Recipe/logs/260210"

python -m src.experiments.run_regression \
  --model_type lgbm \
  --mode train \
  --raw_data_path "${DATA_PATH}" \
  --model_dir "${MODEL_DIR}" \
  --result_dir "${RESULT_DIR}" \
  --log_dir "${LOG_DIR}" \
  --n_jobs 5

echo "===== LGBM DONE ====="