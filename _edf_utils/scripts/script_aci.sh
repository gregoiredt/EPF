#!/bin/bash

#SBATCH --job-name=aci
#SBATCH --output=logs/simu.out%j
#SBATCH --error=logs/simu.err%j
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
python main.py --num_cpus 34 --type_training conformal --filename qpred_aci_0808_week_preproc.csv --time_mode week --hours final_choice --method ACI --n_div 1 --preprocessing 1


