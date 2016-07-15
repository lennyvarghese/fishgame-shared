#!/bin/bash 

#$ -V
#$ -j y
#$ -S /bin/bash
#$ -P anl1

. activate /project/anl1/anaconda_envs/mne_anlffr_mkl
python /projectnb/anl1/lv/fishgame-shared/v3.py $@
