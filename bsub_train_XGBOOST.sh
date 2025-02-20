#!/bin/bash

job_name=$1
log_folder="$HOME/AI_workshop/PlantCaduceus/my_logs/XGBOOST_logs"
mkdir -p ${log_folder}
mem=50
threads=20
project_folder="/home/labs/alevy/petrzhu/AI_workshop/PlantCaduceus/datasets/TAIR/TAIR10_FT_neutral_vs_simulated"

bsub -q medium -R rusage[mem=${mem}GB] -R affinity[thread*${threads}] \
    -J ${job_name} -e ${log_folder}/${job_name}_%J.err -o ${log_folder}/${job_name}_%J.out \
    "source activate PlantCad2; python /home/labs/alevy/petrzhu/AI_workshop/PlantCaduceus/PZ_scripts/Train_XGBOOST.py \
    -train_emb \"${project_folder}_train_npz/combined_embeddings.npz\" \
    -valid_emb \"${project_folder}_val_npz/combined_embeddings.npz\" \
    -test_emb \"${project_folder}_test_npz/combined_embeddings.npz\" \
    -train_labels \"${project_folder}_train.txt\" \
    -valid_labels \"${project_folder}_val.txt\" \
    -test_labels \"${project_folder}_test.txt\" \
    -output \"${project_folder}_RESULT1\""

# 40GB for arabidopsis, 10 threads

# train_emb=""
# valid_emb=""
# test_emb=""
# train_labels=""
# valid_labels=""
# test_labels=""
# output_folder=""
# seed=""

# log_folder="$HOME/AI_workshop/PlantCaduceus/my_logs/XGBOOST_logs"
# mkdir -p ${log_folder}

# while [[ $# -gt 0 ]]; do
#   case $1 in
#     -train_emb) train_emb="$2"; shift 2 ;;
#     -valid_emb) valid_emb="$2"; shift 2 ;;
#     -test_emb) test_emb="$2"; shift 2 ;;
#     -train_labels) train_labels="$2"; shift 2 ;;
#     -valid_labels) valid_labels="$2"; shift 2 ;;
#     -test_labels) test_labels="$2"; shift 2 ;;
#     -output) output_folder="$2"; shift 2 ;;
#     -seed) seed="$2"; shift 2 ;;
#     *) echo "Unknown option $1"; exit 1 ;;
#   esac
# done

# # Проверка обязательных параметров
# if [[ -z "$train_emb" || -z "$valid_emb" || -z "$train_labels" || -z "$valid_labels" || -z "$output_folder" ]]; then
#   echo "Error: Missing required arguments"
#   echo "Usage: $0 -train_emb <path> -valid_emb <path> -train_labels <path> -valid_labels <path> -output <path> [-test_emb <path>] [-test_labels <path>] [-seed <value>]"
#   exit 1
# fi

# bsub -q long -R rusage[mem=${mem}GB] -R affinity[thread*${threads}] \
#     -J ${job_name} -e ${log_folder}/${job_name}_%J.err -o ${log_folder}/${job_name}_%J.out \
#     "source activate PlantCad2; python /home/labs/alevy/petrzhu/AI_workshop/PlantCaduceus/PZ_scripts/Train_XGBOOST.py \
#     -train_emb "$train_emb" \
#     -valid_emb "$valid_emb" \
#     ${test_emb:+-test_emb "$test_emb"} \
#     -train_labels "$train_labels" \
#     -valid_labels "$valid_labels" \
#     ${test_labels:+-test_labels "$test_labels"} \
#     -output "$output_folder" \
#     ${seed:+-seed "$seed"}"



