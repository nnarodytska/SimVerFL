import torch
import torchvision
import torchvision.transforms as transforms
import numpy as np
from sklearn.datasets import load_svmlight_files
from torch.utils.data import TensorDataset
from .datasets_setup import DATASET_BASE_PATHS, DATASET_CORR_PATHS, DATASET_DEFAULT_PATHS, DATASET_EXPAND_PATHS, DATASET_FLIP_PATHS, DATASET_PATHS, download_UCI_dataset 

import pandas as pd

class datasets():

    def __init__(self, dataset, num_clients, params={'data_per_client': 'sequential', 'dataset_variant': 'default'}):

        if dataset == 'CIFAR10':

            transform_train = transforms.Compose([
                transforms.RandomCrop(32, padding=4),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
            ])

            transform_test = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
            ])

            self.trainset = torchvision.datasets.CIFAR10(
                root='./data', train=True, download=True, transform=transform_train)

            self.testset = torchvision.datasets.CIFAR10(
                root='./data', train=False, download=True, transform=transform_test)

            self.num_classes = 10

        elif dataset == 'CIFAR100':

            transform_train = transforms.Compose([
                transforms.RandomCrop(32, padding=4),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
            ])

            transform_test = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
            ])

            self.trainset = torchvision.datasets.CIFAR100(
                root='./data', train=True, download=True, transform=transform_train)

            self.testset = torchvision.datasets.CIFAR100(
                root='./data', train=False, download=True, transform=transform_test)

            self.num_classes = 100

        elif dataset == 'MNIST':

            transform_train = transforms.Compose([
                torchvision.transforms.Resize((32, 32)),
                torchvision.transforms.ToTensor(),
                torchvision.transforms.Normalize((0.1307,), (0.3081,))
            ])

            transform_test = transforms.Compose([
                torchvision.transforms.Resize((32, 32)),
                torchvision.transforms.ToTensor(),
                torchvision.transforms.Normalize((0.1307,), (0.3081,))
            ])

            self.trainset = torchvision.datasets.MNIST(
                root='./data', train=True, download=True, transform=transform_train)

            self.testset = torchvision.datasets.MNIST(
                root='./data', train=False, download=True, transform=transform_test)

            self.num_classes = 10
        elif dataset == 'a4a' or dataset == 'a9a':
            paths = DATASET_PATHS[dataset]
            if not (paths['train'].exists() and paths['test'].exists()):
                download_UCI_dataset(dataset)
            train_x, train_y, test_x, test_y = load_svmlight_files([str(paths['train']), str(paths['test'])])

            train_x = train_x[:-1]
            train_y = train_y[:-1]

            def convert_UCI_to_torch(x, y):
                return (torch.from_numpy(x.toarray().astype(np.float32)),
                        torch.reshape(torch.from_numpy(((y + 1) / 2).astype(np.float32)), [-1, 1]))

            self.trainset = TensorDataset(*convert_UCI_to_torch(train_x, train_y))
            self.testset = TensorDataset(*convert_UCI_to_torch(test_x, test_y))

            self.num_classes = 2

        elif dataset in ['square', 'flipcross', 'cross', 'adult', 'credit', 'recidivism', 'lending', 'heloc']:
            if params['dataset_variant'] == 'base':
                paths = DATASET_BASE_PATHS[dataset]
            elif params['dataset_variant'] == 'default':
                paths = DATASET_DEFAULT_PATHS[dataset]
            elif params['dataset_variant'] == 'flip':
                paths = DATASET_FLIP_PATHS[dataset]
            elif params['dataset_variant'] == 'corr':
                paths = DATASET_CORR_PATHS[dataset]    
            elif params['dataset_variant'] == 'expand':
                paths = DATASET_EXPAND_PATHS[dataset]                                
            else:
                assert False, "Unknown dataset type"
        
            if not (paths['train'].exists() and paths['test'].exists()):
                assert False,  f"Dataset files are missing in {paths['train']} and/or {paths['test']}"

            #train_x, train_y, test_x, test_y = load_svmlight_files([str(paths['train']), str(paths['test'])])

            train = pd.read_csv(str(paths['train']))
            test = pd.read_csv(str(paths['test']))



            features = train.columns[:-2].astype(str)

        
            train_x         = (train.iloc[:, 0:-2]).to_numpy()
            train_y         = (train.iloc[:, -2:-1]).to_numpy()
            train_clients   = (train.iloc[:, -1]).to_numpy()

            test_x          = (test.iloc[:,  0:-2]).to_numpy()
            test_y          = (test.iloc[:, -2:-1]).to_numpy()
            test_clients    = (test.iloc[:, -1]).to_numpy()

            def convert_SDS_to_torch(x, y):
                return (torch.from_numpy(x.astype(np.float32)),
                        torch.from_numpy(y.astype(np.int).reshape((-1,))))


            self.features = features
            self.trainset = TensorDataset(*convert_SDS_to_torch(train_x, train_y))
            self.testset = TensorDataset(*convert_SDS_to_torch(test_x, test_y))
            self.train_clients  = train_clients
            self.test_clients  = test_clients

            self.n_features  = len(train_x[0])
            self.num_classes = 2

        else:
            raise Exception('Unsupported dataset {}'.format(dataset))

        self.num_clients = num_clients
        self.trainset_len = len(self.trainset)
        self.testset_len = len(self.testset)
        self.params = params
        self.indices = {}

    def get_client_train_data(self, client_id):

        if self.params['data_per_client'] == 'sequential':

            # assume that self.trainset_len >> self.num_clients
            k, m = divmod(self.trainset_len, self.num_clients)
            indices = list(range(client_id * k + min(client_id, m), (client_id + 1) * k + min(client_id + 1, m)))

            # debug
            subset = torch.utils.data.Subset(self.trainset, indices)
            print('(train) client id {} ({})'.format(client_id, len(subset.indices)))

            return torch.utils.data.Subset(self.trainset, indices)

        elif self.params['data_per_client'] == 'label_per_client':

            # Assuming that classes start from 0, similarly to client_id. Holds for CIFAR10 and CIFAR100. ### TODO: why not for MNIST?
            indices = np.where(np.array(self.trainset.targets) == (client_id % self.num_classes))[0]

            # debug
            subset = torch.utils.data.Subset(self.trainset, indices.tolist())
            print('(train) client id {} with {} label ({})'.format(client_id, client_id % self.num_classes, len(subset.indices)))

            unique_targets = np.unique([subset.dataset[idx][1] for idx in subset.indices])
            if not (len(unique_targets) == 1 and unique_targets[0] == client_id % self.num_classes):
                raise Exception('Error, label_per_client (train) misbehaves, expecting class {} for client {}, '
                                'but getting class(es) {}'.format(client_id % self.num_classes, client_id,
                                                                  unique_targets))

            return torch.utils.data.Subset(self.trainset, indices.tolist())

        elif self.params['data_per_client'] == 'label_per_client1':
            iid_trainset, per_label_trainset = torch.utils.data.random_split(self.trainset, [int(0.2*len(self.trainset)), int(0.8*len(self.trainset))], generator=torch.Generator().manual_seed(42))
            # Assuming that classes start from 0, similarly to client_id. Holds for CIFAR10 and CIFAR100. ### TODO: why not for MNIST?

            # assume that self.trainset_len >> self.num_clients
            k, m = divmod(len(iid_trainset), self.num_clients)
            iid_indices = list(range(client_id * k + min(client_id, m), (client_id + 1) * k + min(client_id + 1, m)))
            per_label_indices = np.where(np.array(per_label_trainset.targets) == (client_id % self.num_classes))[0]

            return torch.utils.data.ConcatDataset(torch.utils.data.Subset(per_label_trainset, per_label_indices.tolist()), torch.utils.data.Subset(iid_trainset, iid_indices))

        elif self.params['data_per_client'] == 'synthetic-data-split':
            indices = list(np.where(self.train_clients == client_id))
            indices = indices[0]
            return torch.utils.data.Subset(self.trainset, indices)
        else:

            raise Exception('Unknown params[\'data_per_client\']: {}'.format(self.params['data_per_client']))

    def get_client_test_data(self, client_id):


        if self.params['data_per_client'] == 'sequential':

            # assume that self.testset_len >> self.num_clients
            k, m = divmod(self.testset_len, self.num_clients)
            indices = list(range(client_id * k + min(client_id, m), (client_id + 1) * k + min(client_id + 1, m)))
            self.indices[client_id] = indices

            # debug
            subset = torch.utils.data.Subset(self.testset, indices)
            print('(test) client id {} ({})'.format(client_id, len(subset.indices)))


            return torch.utils.data.Subset(self.testset, indices)

        elif self.params['data_per_client'] == 'label_per_client':

            # Assuming that classes start from 0, similarly to client_id. Holds for CIFAR10 and CIFAR100.
            indices = np.where(np.array(self.testset.targets) == (client_id % self.num_classes))[0]
            self.indices[client_id] = indices
            # debug
            subset = torch.utils.data.Subset(self.testset, indices.tolist())
            print('(test) client id {} with {} label ({})'.format(client_id, client_id % self.num_classes, len(subset.indices)))

            unique_targets = np.unique([subset.dataset[idx][1] for idx in subset.indices])
            if not (len(unique_targets) == 1 and unique_targets[0] == client_id % self.num_classes):
                raise Exception('Error, label_per_client (test) misbehaves, expecting class {} for client {}, '
                                'but getting class(es) {}'.format(client_id % self.num_classes, client_id,
                                                                  unique_targets))


            return torch.utils.data.Subset(self.testset, indices.tolist())

        elif self.params['data_per_client'] == 'synthetic-data-split':
            indices = list(np.where(self.test_clients == client_id))
            indices = indices[0]
            return torch.utils.data.Subset(self.testset, indices)            

        else:

            raise Exception('Unknown params[\'data_per_client\']: {}'.format(self.params['data_per_client']))

    def get_data_np(self, set):
        set_np = []
        target_np = []
        for t in set: 
            sample = list(t[0].cpu().detach().numpy())
            set_np.append(sample)
            target = list(t[0].cpu().detach().numpy())
            target_np.append(target)
        return np.asarray(set_np), np.asarray(target_np)


    def get_train_data(self):
        return self.trainset

    def get_test_data(self):
        return self.testset

    def get_indices(self):
        return self.indices