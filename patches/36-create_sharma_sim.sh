#!/usr/bin/bash

mkdir -p computed/tmp/

for LANG in 'en' 'am' 'bn' 'uz' 'pl' 'es' 'sw'; do
    SIGNATURE="sharma_sim_${LANG}"
    sbatch --time=00-10 --ntasks=70 --mem-per-cpu=4G \
        --job-name="${SIGNATURE}" \
        --output="logs/${SIGNATURE}.log" \
        --wrap="\
            python3 ./models/sharma/create_sim.py \
                --lang ${LANG} \
                --ntasks 60 \
            ;"

done;