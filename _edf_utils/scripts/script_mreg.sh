#!/bin/bash

#SBATCH --job-name=mreg
#SBATCH --output=Travail/code/code/logs/%j.txt
#SBATCH --error=Travail/code/code/logs/err_%j.txt
#SBATCH -t 71:59:59
#SBATCH --output=res.txt
#SBATCH --array 0-3
#SBATCH --nodes=4
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=36
#SBATCH --mem-per-cpu=3500
#SBATCH --partition=cn
#SBATCH --wckey=P11YQ:StOpt
source  py37/bin/activate
cd Travail/code/code

name_file_arr=(lasso_cv_2019_0408_preproc.csv lasso1__2019_0408_preproc.csv xgblin_2019_0408_preproc.csv rf_2019_0408_preproc.csv)
name_model_arr=(lassocv lasso1 xgblin rf)

for id in {0..3}
do
  printf $id"\n"
  if [ "$id" -eq "$SLURM_ARRAY_TASK_ID" ]
  
  then
    name_file=${name_file_arr[id]}
    name_model=${name_model_arr[id]}
    #printf "main.py --num_cpus 34 --type_training mreg --filename $name_file --time_mode day --hours lasso_cv --models $name_model --n_div 2 --id_stop 5"\n
    python main.py --num_cpus 34 --type_training mreg --filename $name_file --time_mode day --hours all --models $name_model --n_div 5 --id_stop 740 --preprocessing 1
  fi
done



