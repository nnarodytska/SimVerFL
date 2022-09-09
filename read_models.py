from ast import arg
import torch
from defs import *

from client import model_factories
from client.dataset_factories import datasets
from client.model_factories import MODEL_TWO_PARAMS

import argparse

import random
import numpy as np
from torch.autograd import Variable
import copy
import pickle




def read_model_path(fn, id):
    data = torch.load(fn, map_location='cpu')
    if id == -1:
        # return global model
        return data['checkpoint']['args'], data['checkpoint']['params']
    elif id >= 0:
        return data['checkpoint']['args'], data['client_models'][id]
    else:
        raise Exception ('Invalid id value: {}'.format(id))

def update_net(layered_parameters, net, device):
    # update net with given params
    pytorch_total_params = sum(p.numel() for p in net.parameters())
    initial_params = torch.zeros(pytorch_total_params, device=device)

    offset = 0
    for params, updated_params in zip(net.parameters(), layered_parameters):

        layer_size = len(params.data.view(-1))
        initial_params[offset:offset + layer_size] = updated_params.data.view(-1).detach().clone()
        offset += layer_size
        params.data = updated_params.data.to(device).detach().clone()

def fix_seeds(seed, device):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.random.manual_seed(seed)
    if device == "cuda":
        torch.cuda.manual_seed_all(seed)
        torch.cuda.manual_seed(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

def read_models(path, debug = False):
    nb_clients = 1000
    id = -1
    
    trained_models = {}
    while(True): # -1 stands for the federated model
        if debug:  print(f"id {id}")
        if (id >= nb_clients):
            break
        try:
            trainining_args, params = read_model_path(path, id)
        except:
            assert False, f"Model {path} is missing"


        if id == -1: # load once for all models
            dataloader = datasets(trainining_args.dataset, trainining_args.clients, params={'data_per_client': trainining_args.data_per_client,  'dataset_variant': trainining_args.data_variant})

            trainset = dataloader.get_train_data()
            trainset_np, traintarget_np = dataloader.get_data_np(trainset)
            np.random.shuffle(trainset_np)            
            trainloader = torch.utils.data.DataLoader(trainset, batch_size=trainining_args.client_batch_size, shuffle=True, num_workers=0, pin_memory=True)
            
            testset = dataloader.get_test_data()
            testloader = torch.utils.data.DataLoader(testset, batch_size=trainining_args.client_batch_size, shuffle=True, num_workers=0, pin_memory=True)         

            nb_clients = trainining_args.clients            

        else:
            trainset = dataloader.get_client_train_data(id)
            trainset_np, traintarget_np = dataloader.get_data_np(trainset)
            np.random.shuffle(trainset_np)            
            trainloader = torch.utils.data.DataLoader(trainset, batch_size=trainining_args.client_batch_size, shuffle=True, num_workers=0, pin_memory=True)
            
            testset = dataloader.get_client_test_data(id)
            testloader = torch.utils.data.DataLoader(testset, batch_size=trainining_args.client_batch_size, shuffle=True, num_workers=0, pin_memory=True)         


        # initiate the net
        # # TODO: need to modify to support num_features as in input
        if not (trainining_args.net in MODEL_TWO_PARAMS):
            net = getattr(model_factories, trainining_args.net)(dataloader.num_classes)
        else:
            net = getattr(model_factories, trainining_args.net)(dataloader.n_features, dataloader.num_classes)
        
        local_model = net.to(device)        
        local_model.eval()        
        # update the net with the loaded params
        update_net(params, net, device)
        if id == -1: # load once for all models]
            global_model = copy.deepcopy(net)
            global_model.eval()
                
        trained_models[id] = copy.deepcopy(local_model)
    
        id +=1
    return trained_models, dataloader, trainining_args

if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    ### if debug, only prints the command to run, without execution
    parser.add_argument('--path', '-p', required=True, help='path to checkpoint')
    parser.add_argument('--debug', '-d',  required=False, default=0, help='  ')
    input_args = parser.parse_args()

    ################ Constants #############
    fix_seeds(42, "gpu")
    gpu = 0

   
    device, device_ids = get_device(gpu)        
    debug = bool(input_args.debug)
    path = input_args.path


    #**************** Read models ****************# 
    trained_models, dataloader, trainining_args = read_models(path, debug)

    #**************** Test models ****************# 
    for id, net in trained_models.items():
        # load data
        if (id != -1):
            #clients data
            testset = dataloader.get_client_test_data(id)
            testloader = torch.utils.data.DataLoader(testset, batch_size=trainining_args.client_batch_size, shuffle=True, num_workers=0, pin_memory=True)         
        else:
            # global model, so test data contains all data. We can use it for debugging 
            testset = dataloader.get_test_data()
            testloader = torch.utils.data.DataLoader(testset, batch_size=trainining_args.client_batch_size, shuffle=True, num_workers=0, pin_memory=True)     

        # test model
        test_correct = 0
        test_total = 0

        with torch.no_grad():
            for inputs, targets in testloader:
                inputs, targets = inputs.to(device), targets.to(device)

                outputs = net(inputs)
                _, predicted = outputs.max(1)
                test_total += targets.size(0)
                test_correct += predicted.eq(targets).sum().item()
        # print the model accuracy
        if id == -1:
            print('global model: acc = {}'.format(100. * test_correct / test_total))
        else:
            print('client {} model: acc = {}'.format(id, 100. * test_correct / test_total))

        

