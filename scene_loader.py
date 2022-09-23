from ai2thor.controller import Controller

from constants import ACTION_SIZE
from constants import SCREEN_WIDTH
from constants import SCREEN_HEIGHT
from constants import HISTORY_LENGTH

GRID_SIZE = 0.25

class THORDiscreteEnvironment(object):

    def __init__(self, config=dict()):

        self.scene_name         = config.get('scene_name', 'badroom_04')
        self.random_statrt      = config.get('random_start', True)
        self.n_feat_per_locaiton = config.get('n_feat_per_locaiton', 1)
        self.terminal_state_id   = config.get('terminal_state_id', 0)

        controller = Controller(
            scene=self.scene_name,
            gridSize=GRID_SIZE,
            width=SCREEN_HEIGHT,
            height=SCREEN_HEIGHT,
            renderDepthImage=True
        )

        