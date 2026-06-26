#!/bin/bash
set -e

echo "===== RUN GANDALF REGRESSION ====="

DATA_PATH="/home_ext/zzx/work_file/ATEL-Correction-Analysis-of-Recipe/raw_data/df_filtered_outlier.csv"
MODEL_DIR="/home_ext/zzx/work_file/ATEL-Correction-Analysis-of-Recipe/models_result/gandalf"
RESULT_DIR="/home_ext/zzx/work_file/ATEL-Correction-Analysis-of-Recipe/results"
LOG_DIR="/home_ext/zzx/work_file/ATEL-Correction-Analysis-of-Recipe/logs/gandalf"

python -m src.tasks.current_prediction.run_regression \
  --model_type gandalf \
  --mode train \
  --date_cols hc_chamber_day:ME:%Y-%m-%d \
  --raw_data_path "${DATA_PATH}" \
  --model_dir "${MODEL_DIR}" \
  --result_dir "${RESULT_DIR}" \
  --log_dir "${LOG_DIR}" \
  --n_jobs 1 

echo "===== GANDALF DONE ====="