#!/bin/bash

#SBATCH --job-name=qregs_week
#SBATCH --output=Travail/code/code/logs/%j.txt
#SBATCH --error=Travail/code/code/logs/err_%j.txt
#SBATCH -t 71:59:59
#SBATCH --output=res.txt
#SBATCH --array 0-3
#SBATCH --nodes=4
#SBATCH --ntasks=4
#SBATCH --cpus-per-task=36
#SBATCH --mem-per-cpu=3500
#SBATCH --partition=cn
#SBATCH --wckey=P11YQ:StOpt
source  py37/bin/activate
cd Travail/code/code

name_file_arr=(linear_week_preproc_0808.csv lasso1_week_preproc_0808.csv gb_week_preproc_0808.csv qrf_week_preproc_0808.csv)
name_model_arr=(linear lasso1 gb qrf)

for id in {0..3}
do
  printf $id"\n"
  if [ "$id" -eq "$SLURM_ARRAY_TASK_ID" ]
  
  then
    name_file=${name_file_arr[id]}
    name_model=${name_model_arr[id]}
    #printf "main.py --num_cpus 34 --type_training mreg --filename $name_file --time_mode day --hours lasso_cv --models $name_model --n_div 2 --id_stop 5"\n
    python main.py --num_cpus 34 --type_training qreg --filename $name_file --time_mode week --hours final_choice --models $name_model --n_div 1 --id_stop 740 --preprocessing 1
  fi
done

#lasso1 lasso2 linear gb qrf
#lasso1_0308_preproc.csv lasso2_0308_preproc.csv linear_0308_preproc.csv gb_0308_preproc.csv qrf_0308_preproc.csv
