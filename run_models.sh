#!/bin/bash

whichType=$1
parentDir=/projectnb/anl1/lv/fishgame_rerun_models
nSamps=20000
nBurn=10000
nRun=2

for q in `seq 1 $nRun`; do
    for t in "visual" "audio"; do
        for s in "accuracy"; do
            # C = depends on congruence
            # S = use stim coding on this variable
            for p in "vC zSC aC t" "vC zSC a t" "vC zS a t" "vC zS aC t" "v zSC aC t" "v zSC a t" "v zS aC t" "v zS a t";  do
                jobstr="$t"_"$s"_"$q"_"$whichType"_`echo "$p" | sed 's/ /_/g'`_"$nSamps"
                qsub -l h_rt=300:00:00 -N $jobstr "$parentDir"/qsub_script.sh $q $s $t $nSamps $nBurn $whichType $p
            done
        done
    done
done
