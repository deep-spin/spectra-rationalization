GPU=0
SEEDS=(38 39 40 41 42)
TRANSITIONS=(0.001 0.01 0.1 0 0.5 1 1.5)

for TRANSITION in "${TRANSITIONS[@]}" 
do
    for SEED in "${SEEDS[@]}"
    do
        CUDA_VISIBLE_DEVICES=$GPU python3 -W ignore rationalizers train \
            --config /home/sophia/spectra-rationalization/configs/beer/beer_mixedspectra.yaml \
            --seed $SEED \
            --transition $TRANSITION \
            --default_root_dir /home/sophia/spectra-rationalization/experiments/beer-mixed-tuning/hadamard/$TRANSITION/$SEED/ 
    done
done
