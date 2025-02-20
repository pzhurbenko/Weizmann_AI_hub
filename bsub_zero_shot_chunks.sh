#!/bin/bash

# Вводные параметры
job_name=$1
inp_folder=$2
output_folder="${inp_folder/_chunks/_chunk_results}"
log_folder="$HOME/AI_workshop/PlantCaduceus/my_logs/zero_shot_logs"

# Display help message if needed
if [[ $1 == "--help" || $1 == "-h" ]]; then
  echo "Usage: bash bsub_save_embeddings.sh job_name inp_folder_chunks"
  echo "Log folder is ${log_folder}"
  exit 0
fi
mkdir -p ${log_folder}
mkdir -p ${output_folder}
echo "$output_folder/$i"
# Итерация по файлам в папке
for i in $(ls $inp_folder); do
    # Формируем команду с подставленными параметрами
#    bsub -q long-gpu -gpu num=1:j_exclusive=no:gmem=2G:gmodel=NVIDIAA40 -R rusage[mem=1GB] -R affinity[thread*1] \
    bsub -q long-gpu -gpu "num=1:j_exclusive=no:gmem=2G" -m best_gpu_hosts -R rusage[mem=3GB] -R affinity[thread*1] \
        -J ${job_name} -e ${log_folder}/${job_name}_%J.err -o ${log_folder}/${job_name}_%J.out \
        "source activate PlantCad2; python $HOME/AI_workshop/PlantCaduceus/src/zero_shot_score.py \
        -input \"$inp_folder/$i\" \
        -output \"$output_folder/$i\" \
        -model \"kuleshov-group/PlantCaduceus_l32\" \
        -device \"cuda:0\""
done

# 1 GB for 100k chunks