source ~/anaconda3/etc/profile.d/conda.sh
conda activate traffic_flow

cd model_DGCRN1

# InfoNCE Loss
for t in 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 1.0 2.0 3.0; do
for lambda in 0.01 0.05 0.1 0.5 1.0 2.0; do
for lambda1 in 0.01 0.05 0.1 0.5 1.0 2.0; do
    python traintorch_MDGCRN.py \
        --contra_type=True \
        --temp=$t \
        --lamb=$lambda \
        --lamb1=$lambda1 \
        --gpu=1

done
done
done

# Triplet Loss
for lambda in 0.01 0.05 0.1 0.5 1.0 2.0; do
for lambda1 in 0.01 0.05 0.1 0.5 1.0 2.0; do
    python traintorch_MDGCRN.py \
        --contra_type=False \
        --lamb=$lambda \
        --lamb1=$lambda1 \
        --gpu=1

done
done
  

