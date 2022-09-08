# -*- coding: utf-8 -*-
"""
Created on Tue Aug 18 13:51:34 2020

@author: HOME-PC
"""

import torch

DEF_CONSTANT_MODEL_GLOBAL   = "global"
DEF_CONSTANT_MODEL_LOCAL    = "local"
DEF_CONSTANT_MISSCLASS      = "missclassifed"
DEF_CONSTANT_CORRECTCLASS   = "correctclassifed"

DEF_EXPLAINER_LIME          = "lime"
DEF_EXPLAINER_ANCOR          = "anchor"
DEF_EXPLAINER_SHAP          = "shap"




def get_device(gpu):
    if gpu == -1: # use all available GPUs
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        device_ids = [x for x in range(torch.cuda.device_count())]
    else:
        if gpu > torch.cuda.device_count()-1:
            raise Exception('got gpu index={}, but there are only {} GPUs'.format(gpu, torch.cuda.device_count()))
        if torch.cuda.is_available():
            device = 'cuda:{}'.format(gpu)
            device_ids = [gpu]
        else:
            device = 'cpu'

    if device == 'cpu':
        print('*** Warning: No GPU was found, running over CPU.')

    print('*** Set device to {}'.format(device))
    if device == 'cuda' and torch.cuda.device_count() > 1:
        print('*** Running on multiple GPUs ({})'.format(torch.cuda.device_count()))

    return device, device_ids