from PIL import Image
from resnet50 import ResNet50
from sklearn import manifold

import torch
import torchvision.transforms as trns
import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt

color_dict = {
        'door': 1,
        'sofa': 2,
        'table': 3,
        'TV': 4,
        'flower': 5,
        'laptop': 6
    }

def class_type():
    classType = ['']*40
    for i in range(0 ,40):
        if i < 20:
            classType[i] = 'door'
        # elif i <= 20 and i > 10:
        #     classType[i] = 'sofa'
        # elif i <= 32 and i > 20:
        #     classType[i] = 'table'
        elif i < 40 and i >= 20:
            classType[i] = 'TV'
        # elif i <= 50 and i > 40:
        #     classType[i] = 'flower'
        # elif i <= 60 and i > 50:
        #     classType[i] = 'laptop'
    return classType

if __name__ == '__main__':
    model = torch.load('/home/brianchen/Documents/visual_navigation/data/resnet50_1664008061.pt')
    image_dir = '/home/brianchen/Documents/visual_navigation/data/image'
    pred_list = np.empty((0, 2048))

    transforms = trns.Compose(
        [
            trns.Resize((224, 224)),
            trns.ToTensor(),
            trns.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ]
    )

    for path in os.listdir(image_dir):
        image_path = os.path.join(image_dir, path)
        if os.path.isfile(image_path):
            image = Image.open(image_path).convert("RGB")
            image_tensor = transforms(image)
            image_tensor = image_tensor.unsqueeze(0)

            prediction = model(image_tensor).squeeze()
            prediction_array = prediction.detach().numpy()

            pred_list = np.append(pred_list, np.array([prediction_array]), axis=0)

    pd.DataFrame(pred_list).to_csv("/home/brianchen/Documents/visual_navigation/visual_navigation.csv")

    X_tsne = manifold.TSNE(n_components=2, init='random', random_state=5, perplexity=5).fit_transform(pred_list)
    x_min, x_max = X_tsne.min(0), X_tsne.max(0)
    X_norm = (X_tsne - x_min) / (x_max - x_min)
    classtype = class_type()

    plt.figure(figsize=(8, 8))
    for i in range(0, (X_norm.shape[0] - 1)):
        plt.text(
            X_norm[i, 0], 
            X_norm[i, 1], 
            str(classtype[i]), 
            color=plt.cm.Set1(color_dict.get(classtype[i])),
            fontdict={'weight': 'bold', 'size': 9}
        )
    plt.xticks([])
    plt.yticks([])
    plt.show()             
    