#!/bin/bash

#SBATCH --job-name=cqr1cv
#SBATCH --output=Travail/code/code/logs/%j.txt
#SBATCH --error=Travail/code/code/logs/err_%j.txt
#SBATCH -t 71:59:59
#SBATCH --output=res.txt
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=36
#SBATCH --mem-per-cpu=3500
#SBATCH --partition=cn
#SBATCH --wckey=P11YQ:StOpt
source  py37/bin/activate
cd Travail/code/code
python main.py --num_cpus 34 --type_training conformal --filename cqr_cv_test_0808_week_preproc.csv --time_mode week --no_conformal 1 --hours final_choice --method CQRval --n_div 10 --id_start 0 --id_stop 740 --preprocessing 1 --cqr_cv 1

