import time
import json
import numpy as np
import cv2

from ai2thor.controller import Controller
from utils.tools import SimpleImageViewer

GRID_SIZE = 0.25

def key_press(key, mod):
    global human_agent_action, human_wants_restart, stop_requested, take_picture, invert_view

    if key == ord('R') or key == ord('r'):
        human_wants_restart = True
    if key == ord('F') or key == ord('f'):
        stop_requested = True

    if key == 119:
        human_agent_action = "MoveAhead"
    if key == 100:
        human_agent_action = "MoveRight"
    if key == 97:
        human_agent_action = "MoveLeft"
    if key == 115:
        human_agent_action = "MoveBack"
    if key == 120:
        human_agent_action = "Stand"
    if key == 122:
        human_agent_action = "Crouch"

    if key == 101:
        human_agent_action = "RotateRight"
    if key == 113:
        human_agent_action = "RotateLeft"

    if key == 112:
        take_picture = True
    if key == 105:
        invert_view = not invert_view

def rollout(event, controller, viewer, scene_name):
    global human_agent_action, human_wants_restart, stop_requested, take_picture, invert_view

    human_agent_action = None
    human_wants_restart = False
    stop_requested = False
    take_picture = False
    invert_view = False

    while True:
        time.sleep(1)
        if human_agent_action is not None:
            event = controller.step(action=human_agent_action)
            human_agent_action = None
        
        if human_wants_restart:
            controller.reset(scene=scene_name)
            human_wants_restart = False
        
        if event.metadata['collided']:
            print('Collision occurs.')
            event.collided = False

        if stop_requested: break
        
        if take_picture:
            current_image = event.frame
            cv2.imwrite("data/{}_goal.png".format(scene_name), current_image)
            json_dict = {}
            agent_position = event.metadata["agent"]["position"]
            agent_rotation = event.metadata["agent"]["rotation"]
            json_dict["grid_size"] = GRID_SIZE
            json_dict["agent_position"] = agent_position
            json_dict["agent_rotation"] = agent_rotation

            with open('data/{}_goal.json'.format(scene_name), "w") as outfile:
                json.dump(json_dict, outfile)

            take_picture = False

        if invert_view and event.depth_frame is not None:
            depth_image = event.depth_frame
            depth_image /= np.max(depth_image)
            depth_image *= 255
            viewer.imshow(depth_image.astype("uint8"))
        else:
            viewer.imshow(event.frame)

if __name__ == '__main__':
    scene_name = 'FloorPlan212'

    controller = Controller(
        scene=scene_name,
        gridSize=0.25,
        width=1000,
        height=1000,
        grid_size=GRID_SIZE,
        renderDepthImage=True
    )
    event = controller.step(action='Done')

    human_agent_action = None
    human_wants_restart = False
    stop_requested = False
    take_picture = False
    invert_view = False

    viewer = SimpleImageViewer()
    viewer.imshow(event.frame)
    viewer.window.on_key_press = key_press

    print("Use WASD keys to move the agent.")
    print("Use QE keys to move the camera.")
    print("Press I to switch between RGB and Depth views.")
    print("Press P to save an image of the current view.")
    print("Press R to reset agent\'s location.")
    print("Press F to quit.")

    rollout(event, controller, viewer, scene_name)

    print("Closing")
    viewer.close()
    time.sleep(3)
    print("Goodbye.")