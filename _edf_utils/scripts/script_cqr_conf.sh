#!/bin/bash

#SBATCH --job-name=cqr_oj
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
python main.py --num_cpus 34 --type_training conformal --filename qpred_cqroj_preproc.csv --time_mode day --hours final_choice --method CQRoj --n_div 1 --id_start 0 --id_stop 740 --preprocessing 1


