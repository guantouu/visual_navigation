import os
import threading
import numpy as np

import signal
import random
import os

from network import ActorCriticFFNetwork
from training_thread import A3CTrainingThread

from utils.ops import log_uniform
from utils.rmsprop_applier import RMSPropApplier

from constants import ACTION_SIZE
from constants import PARALLEL_SIZE
from constants import INITIAL_ALPHA_LOW
from constants import INITIAL_ALPHA_HIGH
from constants import INITIAL_ALPHA_LOG_RATE
from constants import MAX_TIME_STEP
from constants import CHECKPOINT_DIR
from constants import LOG_FILE
from constants import RMSP_EPSILON
from constants import RMSP_ALPHA
from constants import GRAD_NORM_CLIP
from constants import USE_GPU
from constants import TASK_TYPE
from constants import TASK_LIST

if __name__ == '__main__':
    device = "/gpu:0" if USE_GPU else "/cup:0"
    network_scope = TASK_TYPE
    list_of_tasks = TASK_LIST
    scene_scopes = list_of_tasks.keys()
    global_t = 0
    stop_requested = False

    if not os.path.exists(CHECKPOINT_DIR):
        os.mkdir(CHECKPOINT_DIR)

    inital_learning_rate = log_uniform(INITIAL_ALPHA_LOW,
                                       INITIAL_ALPHA_HIGH,
                                       INITIAL_ALPHA_LOG_RATE)

    global_network = ActorCriticFFNetwork(action_size=ACTION_SIZE,
                                          device=device,
                                          network_scope=network_scope,
                                          scene_scopes=scene_scopes)

    branches = []
    for scene in scene_scopes:
        for task in list_of_tasks[scene]:
            branches.append((scene, task))

    NUM_TASKS = len(branches)
    assert PARALLEL_SIZE >= NUM_TASKS, \
        "Not enough threads for multitasking: at least {} threads needed.".format(NUM_TASKS)

    learning_rate_input = False
    grad_applier = RMSPropApplier(learning_rate=learning_rate_input,
                                  decay=RMSP_ALPHA,
                                  momentum=0.0,
                                  epsilon=RMSP_EPSILON,
                                  clip_norm=GRAD_NORM_CLIP,
                                  device=device)

    training_threads = []
    for i in range(PARALLEL_SIZE):
        scene, task = branches[i%NUM_TASKS]
        training_thread = A3CTrainingThread(i, global_network, inital_learning_rate,
                                            learning_rate_input,
                                            grad_applier, MAX_TIME_STEP,
                                            device=device,
                                            network_scope="thread-%d"%(i+1),
                                            scene_scope=scene,
                                            task_scope=task)
        training_threads.append(training_thread)

    
    def train_function(parallel_index):
        global global_t
        training_thread = training_threads[parallel_index]
        last_global_t = 0

        scene, task = branches[parallel_index % NUM_TASKS]
        key = scene + "-" + task

        while global_t < MAX_TIME_STEP and not stop_requested:
            diff_global_t = training_thread.process()

            gloabl_t += diff_global_t

            if parallel_index == 0 and global_t - last_global_t > 1000000:
                print('Save checkpoint at timestamp %d' % global_t)
                last_global_t = global_t


    def signal_handler(signal, frame):
        global stop_requested
        print('You pressed Ctrl+C')
        stop_requested = True

    train_threads = []
    for i in range(PARALLEL_SIZE):
        train_threads.append(threading.Thread(target=train_function, args=(i,)))

    signal.signal(signal.SIGINT, signal_handler)

    for t in train_threads:
        t.join()
    