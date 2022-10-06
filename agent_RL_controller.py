class THORDiscreteEnvironment(object):
    def __init__(self, config=dict()):
        self.scene_name          = config.get('scene_name', 'bedroom_04')
        self.random_start        = config.get('random_start', True)
        self.n_feat_per_location = config.get('n_feat_per_location', 1)
        self.terminal_state_id   = config.get('terminal_state_id', 0)

        
