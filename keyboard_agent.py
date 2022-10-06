import time
import json
import numpy as np
import re
import cv2

from ai2thor.controller import Controller
from constants import SCREEN_HEIGHT
from constants import SCREEN_WIDTH
from utils.tools import SimpleImageViewer

GRID_SIZE = 0.25
VISIBILITYDISTANCE = 0.5

def key_press(key, mod):
    global human_agent_action, human_wants_restart, stop_requested, take_picture, invert_view, label_text

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

def mouse_press(X, Y, button, modifiers):
    global label_text
    query = controller.step(
        action="GetObjectInFrame",
        x=float(X / SCREEN_WIDTH),
        y=float((SCREEN_HEIGHT - Y) / SCREEN_HEIGHT)
    )
    object_id = query.metadata["actionReturn"]
    # label_text = "X: {}, Y:{}".format(float(X / SCREEN_WIDTH), float((SCREEN_HEIGHT - Y) / SCREEN_HEIGHT))
    label_text = object_id
    

def rollout(event, controller, viewer, scene_name):
    global human_agent_action, human_wants_restart, stop_requested, take_picture, invert_view, label_text, pic_num

    human_agent_action = None
    human_wants_restart = False
    stop_requested = False
    take_picture = False
    invert_view = False
    regex  = re.compile(r'is blocking Agent')

    while True:
        if human_agent_action is not None:
            event = controller.step(action=human_agent_action)
            if regex.search(event.metadata['errorMessage']):
                print('blocking')
            for i in event.metadata["objects"]:
                if i['objectId'] == 'Laptop|+01.80|+00.47|+00.50':
                    print("visible is {}".format(i['visible'] if 'True' else 'False'))
            human_agent_action = None
        
        if human_wants_restart:
            controller.reset(scene=scene_name)
            human_wants_restart = False

        if stop_requested: break
        
        if take_picture:
            current_image = event.cv2img
            cv2.imwrite("/home/brianchen/Documents/visual_navigation/data/image/_{}.png".format(pic_num), current_image)
            json_dict = {}
            agent_position = event.metadata["agent"]["position"]
            agent_rotation = event.metadata["agent"]["rotation"]
            json_dict["grid_size"] = GRID_SIZE
            json_dict["agent_position"] = agent_position
            json_dict["agent_rotation"] = agent_rotation
            json_dict["object_id"] = label_text

            with open('data/{}_goal.json'.format(scene_name), "w") as outfile:
                json.dump(json_dict, outfile)
            pic_num += 1

            take_picture = False

        if invert_view and event.depth_frame is not None:
            depth_image = event.depth_frame
            depth_image /= np.max(depth_image)
            depth_image *= 255
            viewer.imshow(depth_image.astype("uint8"), label_text)
        else:
            viewer.imshow(event.frame, label_text)

if __name__ == '__main__':
    scene_name = 'FloorPlan212'

    controller = Controller(
        scene=scene_name,
        gridSize=0.25,
        width=SCREEN_HEIGHT,
        height=SCREEN_WIDTH,
        grid_size=GRID_SIZE,
        renderDepthImage=True,
        visibilityDistance=VISIBILITYDISTANCE
    )
    event = controller.step(action='Done')

    human_agent_action = None
    human_wants_restart = False
    stop_requested = False
    take_picture = False
    invert_view = False
    label_text = ""
    pic_num = 0

    viewer = SimpleImageViewer()
    viewer.imshow(event.frame, label_text)
    viewer.window.on_key_press = key_press
    viewer.window.on_mouse_press = mouse_press

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