#!/bin/bash
set -e

echo "===== RUN XGBOOST REGRESSION ====="

DATA_PATH="/home_ext/zzx/work_file/ATEL-Correction-Analysis-of-Recipe/raw_data/df_filtered_outlier.csv"
MODEL_DIR="/home_ext/zzx/work_file/ATEL-Correction-Analysis-of-Recipe/outputs/models/xgb"
RESULT_DIR="/home_ext/zzx/work_file/ATEL-Correction-Analysis-of-Recipe/outputs/analysis"
LOG_DIR="/home_ext/zzx/work_file/ATEL-Correction-Analysis-of-Recipe/logs/xgb"

python -m src.tasks.current_prediction.run_regression \
  --model_type xgb \
  --mode train \
  --raw_data_path "${DATA_PATH}" \
  --model_dir "${MODEL_DIR}" \
  --result_dir "${RESULT_DIR}" \
  --log_dir "${LOG_DIR}" \
  --n_jobs 20

echo "===== XGBOOST DONE ====="
