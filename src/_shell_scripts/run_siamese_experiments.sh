export PYTHONPATH=$PYTHONPATH:/projects/StealthAttackDetection

for i in $(seq 0 9); do
    python src/main_siamese_network.py --attack-type slowdos --apply-normalization True --batch-size 16 --n_exps siamese --n_runs $i
done
