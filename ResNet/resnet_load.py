from PIL import Image
from resnet50 import ResNet50

import torch
import torchvision.transforms as trns

if __name__ == '__main__':
    model = torch.load('/home/brianchen/Documents/visual_navigation/data/resnet50_1664008061.pt')
    image_path = '/home/brianchen/Documents/visual_navigation/data/FloorPlan212_goal.png'

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

    image = Image.open(image_path).convert("RGB")
    image_tensor = transforms(image)
    image_tensor = image_tensor.unsqueeze(0)

    prediction = model(image_tensor).squeeze()

