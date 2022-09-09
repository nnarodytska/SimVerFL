# SimVerFL


### Run with synthetic datasets, by federated_cnn.py:

python3.8 read_models.py --path  ./trained_models/[YYY]/square/results_SimpleFC_square_sds_8_8_pr_[XXX]_mu_0.001_cts_1_data_default.fedavg.2000.pt

where 

    XXX is peronalization round, possible values are [1,2,...,100]
    YYY is traingg variant, possible values are  [fedavg, fedprox]

### Examples 

Ex1: personalization round = 0, training = fedavg (no personalization)

python3.8 read_models.py --path  ./trained_models/fedavg/square/results_SimpleFC_square_sds_8_8_pr_0_mu_0.001_cts_1_data_default.fedavg.2000.pt

Ex2: peronalization round = 100, training = fedavg

python3.8 read_models.py --path  ./trained_models/fedavg/square/results_SimpleFC_square_sds_8_8_pr_100_mu_0.001_cts_1_data_default.fedavg.2000.pt



Ex3: personalization round = 0, training = fedprox (no personalization)

python3.8 read_models.py --path  ./trained_models/fedprox/square/results_SimpleFC_square_sds_8_8_pr_0_mu_0.5_cts_2_data_default.fedprox.2000.pt

Ex4: personalization round = 100, training = fedprox

python3.8 read_models.py --path  ./trained_models/fedprox/square/results_SimpleFC_square_sds_8_8_pr_100_mu_0.5_cts_2_data_default.fedprox.2000.pt