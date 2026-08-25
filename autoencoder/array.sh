#!/bin/bash

DATASET="motor"

SUBJECT_COUNTS=(all)
REPS=(0 1 2 3 4 5 6 7 8 9)

for N in "${SUBJECT_COUNTS[@]}"; do

    # "all" only has one split: splits_nall.pkl
    if [ "$N" = "all" ]; then

        echo "Submitting $DATASET n=all"

        sbatch \
            --job-name="${DATASET}_nall" \
            --export=ALL,N_SUBJECTS=all,REP=0 \
            /home1/amadapur/projects/eeg_trait_state_geometry/autoencoder/ae.sh "$DATASET"

    else

        # Numeric subject counts have reps 0-9
        for REP in "${REPS[@]}"; do

            echo "Submitting $DATASET n=$N rep=$REP"

            sbatch \
                --job-name="${DATASET}_n${N}_rep${REP}" \
                --export=ALL,N_SUBJECTS=$N,REP=$REP \
                /home1/amadapur/projects/eeg_trait_state_geometry/autoencoder/ae.sh "$DATASET"

        done
    fi
done