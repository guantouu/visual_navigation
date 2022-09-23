from xml.etree.ElementTree import tostring
import torch
import torch.nn as nn
from torchvision.io import read_image
from torchvision.models import resnet50, ResNet50_Weights


weights = ResNet50_Weights.DEFAULT
model = resnet50(weights=weights)
model.eval()

# state_dict() 回傳一個OrderDict, 儲存網路架構的名子與對應的參數
model_state = model.state_dict()

print("model_state type: ", type(weights))

for param_tensor in model.state_dict():
    print("{0:<50} \t {1:>30}".format(param_tensor, str(model.state_dict()[param_tensor].size())))

