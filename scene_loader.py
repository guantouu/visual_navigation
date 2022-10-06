from symbol import term
from turtle import position
from ai2thor.controller import Controller
from ResNet.resnet50 import ResNet50


from constants import ACTION_SIZE
from constants import SCREEN_WIDTH
from constants import SCREEN_HEIGHT
from constants import HISTORY_LENGTH
from constants import VISIBILITYDISTANCE

import numpy as np
import torchvision.transforms as trns
import torch
import pandas as pd
import random
import re

GRID_SIZE = 0.25

class THORDiscreteEnvironment(object):

    def __init__(self, config=dict()):

        self.scene_name         = 'FloorPlan212'
        self.terminal_id        = 'Laptop|+01.80|+00.47|+00.50'
        self.model              = torch.load('/home/brianchen/Documents/visual_navigation/data/resnet50_1664008061.pt')
        self.transforms         = trns.Compose([trns.Resize((224, 224)), 
                                                trns.ToTensor(), 
                                                trns.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
                                              ])

        self.controller         = Controller(scene=self.scene_name,
                                            gridSize=GRID_SIZE,
                                            width=SCREEN_WIDTH,
                                            height=SCREEN_HEIGHT,
                                            visibilityDistance=VISIBILITYDISTANCE)

        self.positions          = self.controller.step(action="GetReachablePositions").metadata["actionReturn"]

        self.history_length     = HISTORY_LENGTH
        self.regex              = re.compile(r'is blocking Agent')


        self.s_t                = np.zeros([2048])
        self.s_t1               = np.zeros_like(self.s_t)
        
        self.reset()

    def reset(self):
        self.controller.reset(scene=self.scene_name)
        position = random.choice(self.positions)
        self.controller.step(
            action="Teleport",
            position=position
        )
        
        self.current_state = self.controller.step(action='Done')
        self.s_t = self._tiled_state(self.current_state.frame)
        
        self.reward   = 0
        self.collided = False
        self.terminal = False
    
    def step(self, action):
        assert not self.terminal, 'step() called in terminal state'
        self.current_state = self.controller.step(action=action)
        for i in self.current_state.metadata["objects"]:
            if i['objectId'] == self.terminal_id:
                self.terminal = True
        if self.regex.search(self.current_state.metadata['errorMessage']):
            self.collided = True
        else:
            self.collided = False
        self.reward = self._reward(self.terminal, self.collided)
        self.s_t1 = np.append(self.s_t[:,1:], self.state, axis=1)

    def update(self):
        self.s_t = self.s_t1

    def _tiled_state(self, frame):
        f = self._resnet_feature(frame=frame)
        return np.tile(f, (1, self.history_length))

    def _reward(self, terminal, collided):
        if terminal: return 10.0
        return -0.1 if collided else -0.01
    
    def _resnet_feature(self, frame):
        image_tensor = self.transforms(frame)
        image_tensor = image_tensor.unsqueeze(0)

        prediction = self.model(image_tensor).squeeze()
        prediction_array = prediction.detach().numpy()
        return prediction_array[:,np.newaxis]

    @property
    def state(self):
        return self._resnet_feature(self.current_state.frame)