#!/bin/bash 

#$ -V
#$ -j y
#$ -S /bin/bash
#$ -P anl1

. activate /project/anl1/anaconda_envs/mne_anlffr_mkl
python /projectnb/anl1/lv/fishgame_rerun_models/v3.py $@
