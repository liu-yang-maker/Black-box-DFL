# python experiments_singlegraph_zeroth.py \
# --objective modularity_zeroth \
# --hidden 50 --embed_dim 50 \
# --weight_decay 5e-4 --dropout 0.2 \
# --train_iters 1001 --clustertemp 50 \
# --num_cluster_iter 1 --lr 0.01 \
# --dataset cora

python experiments_singlegraph_kcenter_zeroth.py \
--objective kcenter_zeroth \
--hidden 50 --embed_dim 50 \
--weight_decay 5e-4 --dropout 0.2 \
--train_iters 1001 --clustertemp 30 \
--num_cluster_iter 1 --lr 0.01 \
--dataset cora_connected
