class ActorCriticNetwork(object):
    def __init__(self, action_size, device="/cpu:0"):
        self.device = device
        self.action_size = action_size

    def prepare_loss(self, entropy_beta, scopes):
        scope_key = self._get_key(scopes[:-1])



    def run_policy_and_value():
        raise NotImplementedError()

    def run_policy():
        raise NotImplementedError()
    
    def run_value():
        raise NotImplementedError()

    def get_vars():
        raise NotImplementedError()

    def sync_from():
        pass

    def _local_var_name():
        pass

    def _fc_weight_variable():
        pass

    def _fc_bias_variable():
        pass

    def _conv_weight_variable():
        pass

    def _conv_bias_variable():
        pass

    def _conv2d():
        pass

    def _get_key(self, scopes):
        return  '/'.join(scopes)

class ActorCriticFFNetwork(ActorCriticNetwork):
    """
        Implementation of the target-driven deep siamese actor-critic network from [Zhu et al., ICRA 2017]
        We use tf.variable_scope() to define domains for parameter sharing
    """    
    def __init__(self,
                 action_size,
                 device="/cpu:0",
                 network_scope="network",
                 scene_scopes=["scene"]):
        super().__init__(action_size, device)

    def run_policy_and_value():
        pass

    def run_policy():
        pass

    def run_value():
        pass

    def get_vars():
        pass