dset_name=OCD_90_200_fMRI
results_root=results
seed=2018
exp_id=seed_${seed}

#### training
bsz=32
model=TCN


PYTHONPATH=$PYTHONPATH:. python train.py \
--dset_name ${dset_name} \
--model ${model} \
--bsz ${bsz} \
--results_root ${results_root} \
--exp_id ${exp_id} \
--seed ${seed} \
${@:1}
