from network import ActorCriticFFNetwork

from constants import ACTION_SIZE
from constants import ENTROPY_BETA

class A3CTrainingThread(object):
    def __init__(self,
                 thread_index,
                 global_network,
                 initial_learning_rate,
                 learning_rate_input,
                 grad_applier,
                 max_global_time_step,
                 device,
                 network_scope="network",
                 scene_scope="scene",
                 task_scope="task"):
        self.thread_index = thread_index
        self.learning_rate_input = learning_rate_input
        self.max_global_time_step = max_global_time_step

        self.network_scope = network_scope
        self.scene_scope = scene_scope
        self.task_scope = task_scope
        self.scopes = [network_scope, scene_scope, task_scope]

        self.local_network = ActorCriticFFNetwork(
                                action_size=ACTION_SIZE,
                                device=device,
                                network_scope=network_scope,
                                scene_scopes=[scene_scope])

        self.local_network.prepare_loss(ENTROPY_BETA, self.scene_scope)
