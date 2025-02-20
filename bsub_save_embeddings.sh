#!/bin/bash

# Вводные параметры
job_name=$1
inp_folder=$2
data_type=$3
output_folder="${inp_folder/_chunks/_npz}"
log_folder="$HOME/AI_workshop/PlantCaduceus/my_logs/fine_tune_logs"

# Display help message if needed
if [[ $1 == "--help" || $1 == "-h" ]]; then
  echo "Usage: bash bsub_save_embeddings.sh job_name inp_folder_chunks data_type(test/train/valid)"
  echo "Log folder is ${log_folder}"
  exit 0
fi
mkdir -p ${log_folder}
# output folder will be created by Save_embeddings.py 
# Итерация по файлам в папке
for i in $(ls $inp_folder); do
    # Формируем команду с подставленными параметрами
    bsub -q long-gpu -gpu num=1:j_exclusive=yes:gmem=42G:gmodel=NVIDIAA40 -R rusage[mem=3GB] -R affinity[thread*1] \
        -J ${job_name} -e ${log_folder}/${job_name}_%J.err -o ${log_folder}/${job_name}_%J.out \
        "source activate PlantCad2; python $HOME/AI_workshop/PlantCaduceus/PZ_scripts/Save_embeddings.py \
        -input_chunk \"$inp_folder/$i\" \
        -output_folder \"$output_folder\" \
        -model \"kuleshov-group/PlantCaduceus_l32\" \
        -device \"cuda:0\" \
        -dataset_type \"$data_type\""
done

# 42GB for 100k chunks 
