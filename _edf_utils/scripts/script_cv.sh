#!/bin/bash

#SBATCH --job-name=gridsearch
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

name_file_arr=(xgblin_cv_res.csv xgbtree_cv_res.csv rf_cv_res.csv qrf_cv_res.csv qlasso_cv_res.csv qgb_csv_res.csv)
name_model_arr=(xgblin xgbtree rf qrf qlasso qgb)

for id in {0..5}
do
  printf $id"\n"
  if [ "$id" -eq "$SLURM_ARRAY_TASK_ID" ]
  
  then
    name_file=${name_file_arr[id]}
    name_model=${name_model_arr[id]}
    python main.py --num_cpus 34 --gridsearch 1 --type_training mreg --filename $name_file --time_mode day --hours final_choice --models $name_model --preprocessing 1 --fewer 1
  fi
done

