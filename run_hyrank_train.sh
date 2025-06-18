nohup /opt/conda/bin/python main_train.py --dataset_name 'hyrank' --halfwidth 1 --spe_patch_size 11 --spe_dim 196 --spa_dim 196 --lr 0.03 --num_epoch 60 --warmup 20 > ./checkpoints/hyrank_results.out 2>&1 &










