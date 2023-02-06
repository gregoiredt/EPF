#!/bin/bash

#SBATCH --job-name=ACI_CQRoj
#SBATCH --output=Travail/code/code/logs/%j.txt
#SBATCH --error=Travail/code/code/logs/err_%j.txt
#SBATCH -t 71:59:59
#SBATCH --output=res.txt
#SBATCH --array 0-5
#SBATCH --nodes=6
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=36
#SBATCH --mem-per-cpu=3500
#SBATCH --partition=cn
#SBATCH --wckey=P11YQ:StOpt
source  py37/bin/activate
cd Travail/code/code


name_file_arr=(aci_cqroj_lasso2.csv aci_cqroj_lasso1.csv aci_cqroj_linear.csv aci_cqroj_gb.csv)
name_model_arr=(lasso2 lasso1 linear gb)

for id in {0..3}
do
  printf $id"\n"
  if [ "$id" -eq "$SLURM_ARRAY_TASK_ID" ]
  
  then
    name_file=${name_file_arr[id]}
    name_model=${name_model_arr[id]}
    python main.py --num_cpus 34 --type_training conformal --filename name_file --time_mode day --hours final_choice --method ACI_CQRoj --n_div 1 --id_start 0 --id_stop 740 --preprocessing 1 --parallel 1 --models name_model
  fi
done




