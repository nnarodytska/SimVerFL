
FedAvg training run
python3 federated_cnn.py  --dataset square --clients 8  --clients_per_round 8 --client_batch_size 128 --compression_scheme none --num_rounds 2000 --lr 0.05 --net SimpleFC --gpu 0 --test_every 20 --client_train_steps 1 --nbits 1 --data_per_client synthetic-data-split --personalization_rounds 100 --data_variant default   --ps fedavg_w_optim --ps_lr 0.05  --p_lr 0.001

FedPRox
python3 federated_cnn.py  --dataset square --clients 8  --clients_per_round 8 --client_batch_size 128 --compression_scheme none --num_rounds 2000 --lr 0.05 --net SimpleFC --gpu 1 --test_every 20 --client_train_steps 2 --nbits 1 --data_per_client synthetic-data-split --personalization_rounds 100 --data_variant default   --ps fedavg_w_optim --ps_lr 0.05 --client fedprox --fedprox_mu 0.5 --p_lr 0.001




