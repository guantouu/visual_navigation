from torchvision.models import resnet50, ResNet50_Weights
from resnet50 import ResNet50

import time
import torch
import torch.nn as nn
import numpy as np

if __name__ == '__main__': 
    model = ResNet50([3, 4, 6, 3])

    # state_dict() 回傳一個OrderDict, 儲存網路架構的名子與對應的參數    
    model_state = model.state_dict()

    weights=ResNet50_Weights.DEFAULT
    pretrained_model = resnet50(weights)
    pretrained_model_state = pretrained_model.state_dict()

    pretrained_dict = {k: v for k, v in pretrained_model_state.items() if k in model_state}

    model_state.update(pretrained_dict)
    model.load_state_dict(model_state)

    struct_time = time.localtime()
    time_stamp = int(time.mktime(struct_time))
    torch.save(model, '/home/brianchen/Documents/visual_navigation/data/resnet50_{}.pt'.format(time_stamp))