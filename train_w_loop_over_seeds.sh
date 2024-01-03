GPU=0
SEEDS=(38 39 40 41 42)
for SEED in "${SEEDS[@]}"
do
    CUDA_VISIBLE_DEVICES=$GPU python3 -W ignore rationalizers train --config /home/nunomg/spectra-rationalization/configs/beer/beer_spectra.yaml --seed $SEED
done