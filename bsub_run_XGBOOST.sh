#!/bin/bash

job_name=$1
log_folder="$HOME/AI_workshop/PlantCaduceus/my_logs/XGBOOST_logs"
mkdir -p ${log_folder}
mem=50
threads=1
project_folder="/home/labs/alevy/petrzhu/AI_workshop/PlantCaduceus/datasets/TAIR"

bsub -q medium -R rusage[mem=${mem}GB] -R affinity[thread*${threads}] \
    -J ${job_name} -e ${log_folder}/${job_name}_%J.err -o ${log_folder}/${job_name}_%J.out \
    "source activate PlantCad2; python /home/labs/alevy/petrzhu/AI_workshop/PlantCaduceus/PZ_scripts/Run_XGBOOST.py \
    -embeddings \"${project_folder}/TAIR10_FT_pop_data_ref_alt_npz/combined_embeddings.npz\" \
    -npz_key \"test\" \
    -output_name \"TAIR_Chr1_bed\" \
    -model_path \"${project_folder}/TAIR10_FT_neutral_vs_simulated_RESULT1/seed_42_XGBoost.json\""
