#!/bin/bash

#SBATCH --job-name=gridsearch
#SBATCH --output=Travail/code/code/logs/%j.txt
#SBATCH --error=Travail/code/code/logs/err_%j.txt
#SBATCH -t 71:59:59
#SBATCH --output=res.txt
#SBATCH --array 0-1
#SBATCH --nodes=2
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=36
#SBATCH --mem-per-cpu=3500
#SBATCH --partition=cn
#SBATCH --wckey=P11YQ:StOpt
source  py37/bin/activate
cd Travail/code/code

name_file_arr=(lasso.csv lasso_preproc.csv)
name_model_arr=(lasso lasso)
preproc_arr=(0 1)

for id in {0..1}
do
  printf $id"\n"
  if [ "$id" -eq "$SLURM_ARRAY_TASK_ID" ]
  
  then
    name_file=${name_file_arr[id]}
    name_model=${name_model_arr[id]}
    preproc=${preproc_arr[id]}
    python main.py --num_cpus 34 --gridsearch 1 --type_training mreg --filename $name_file --time_mode day --hours final_choice --models $name_model --preprocessing $preproc
  fi
done

