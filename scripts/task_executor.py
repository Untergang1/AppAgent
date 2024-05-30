import ast
import datetime
import json
import os
import re
import sys
import time

from . import prompts
from .and_controller import chose_device, AndroidController, traverse_tree_all, strip_xml
from .model import parse_explore_rsp, parse_grid_rsp, chose_model
from .utils import print_with_color, draw_bbox_multi, draw_grid

def task_executor(configs):
    mllm = chose_model(configs)
    if mllm == None:
        print_with_color(f"ERROR: Unsupported model type {configs['model']}!", "red")
        sys.exit()

    app = configs["app"]
    root_dir = configs["root_dir"]
    detail = configs["detail"]
    device = configs["DEVICE"]
    task_desc = configs["desc"]
    no_doc = configs["nodoc"]
    max_rounds = configs["MAX_ROUNDS"]
    min_dist = configs["MIN_DIST"]
    request_interval = configs["REQUEST_INTERVAL"]
    dark_mode = configs["DARK_MODE"]

    app_dir = os.path.join(os.path.join(root_dir, "apps"), app)
    work_dir = os.path.join(root_dir, "tasks")
    if not os.path.exists(work_dir):
        os.mkdir(work_dir)
    auto_docs_dir = os.path.join(app_dir, "auto_docs")
    demo_docs_dir = os.path.join(app_dir, "demo_docs")
    task_timestamp = int(time.time())
    dir_name = datetime.datetime.fromtimestamp(task_timestamp).strftime(f"task_{app}_%Y-%m-%d_%H-%M-%S")
    task_dir = os.path.join(work_dir, dir_name)
    os.mkdir(task_dir)
    log_path = os.path.join(task_dir, f"log_{app}_{dir_name}.txt")

    if no_doc:
        print_with_color("proceed without docs.", "yellow")
    elif not os.path.exists(auto_docs_dir) and not os.path.exists(demo_docs_dir):
        print_with_color(f"No documentations found for the app {app}. Do you want to proceed with no docs? Enter y or n",
                         "red")
        user_input = ""
        while user_input != "y" and user_input != "n":
            user_input = input().lower()
        if user_input == "y":
            no_doc = True
        else:
            sys.exit()
    elif os.path.exists(auto_docs_dir) and os.path.exists(demo_docs_dir):
        print_with_color(f"The app {app} has documentations generated from both autonomous exploration and human "
                         f"demonstration. Which one do you want to use? Type 1 or 2.\n1. Autonomous exploration\n2. Human "
                         f"Demonstration",
                         "blue")
        user_input = ""
        while user_input != "1" and user_input != "2":
            user_input = input()
        if user_input == "1":
            docs_dir = auto_docs_dir
        else:
            docs_dir = demo_docs_dir
    elif os.path.exists(auto_docs_dir):
        print_with_color(f"Documentations generated from autonomous exploration were found for the app {app}. The doc base "
                         f"is selected automatically.", "yellow")
        docs_dir = auto_docs_dir
    else:
        print_with_color(f"Documentations generated from human demonstration were found for the app {app}. The doc base is "
                         f"selected automatically.", "yellow")
        docs_dir = demo_docs_dir

    if not device:
        device = chose_device()

    controller = AndroidController(device)
    width, height = controller.get_device_size()
    if not width and not height:
        print_with_color("ERROR: Invalid device size!", "red")
        sys.exit()
    if detail:
        print_with_color(f"Screen resolution of {device}: {width}x{height}", "yellow")

    if not task_desc:
        print_with_color("Please enter the description of the task you want me to complete in a few sentences:", "blue")
        task_desc = input()

    round_count = 0
    last_act = "None"
    task_complete = False
    grid_on = False
    rows, cols = 0, 0


    def area_to_xy(area, subarea):
        area -= 1
        row, col = area // cols, area % cols
        x_0, y_0 = col * (width // cols), row * (height // rows)
        if subarea == "top-left":
            x, y = x_0 + (width // cols) // 4, y_0 + (height // rows) // 4
        elif subarea == "top":
            x, y = x_0 + (width // cols) // 2, y_0 + (height // rows) // 4
        elif subarea == "top-right":
            x, y = x_0 + (width // cols) * 3 // 4, y_0 + (height // rows) // 4
        elif subarea == "left":
            x, y = x_0 + (width // cols) // 4, y_0 + (height // rows) // 2
        elif subarea == "right":
            x, y = x_0 + (width // cols) * 3 // 4, y_0 + (height // rows) // 2
        elif subarea == "bottom-left":
            x, y = x_0 + (width // cols) // 4, y_0 + (height // rows) * 3 // 4
        elif subarea == "bottom":
            x, y = x_0 + (width // cols) // 2, y_0 + (height // rows) * 3 // 4
        elif subarea == "bottom-right":
            x, y = x_0 + (width // cols) * 3 // 4, y_0 + (height // rows) * 3 // 4
        else:
            x, y = x_0 + (width // cols) // 2, y_0 + (height // rows) // 2
        return x, y


    while round_count < max_rounds:
        # grid_on=True
        round_count += 1
        print_with_color(f"Round {round_count}", "yellow")
        screenshot_path = controller.get_screenshot(f"{dir_name}_{round_count}", task_dir)
        xml_path = controller.get_xml(f"{dir_name}_{round_count}", task_dir)
        if screenshot_path == "ERROR" or xml_path == "ERROR":
            break
        if grid_on:
            rows, cols = draw_grid(screenshot_path, os.path.join(task_dir, f"{dir_name}_{round_count}_grid.png"))
            image = os.path.join(task_dir, f"{dir_name}_{round_count}_grid.png")
            prompt = prompts.task_template_grid
        else:
            # clickable_list = []
            # focusable_list = []
            # traverse_tree(xml_path, clickable_list, "clickable", True)
            # traverse_tree(xml_path, focusable_list, "focusable", True)
            # elem_list = clickable_list.copy()
            # for elem in focusable_list:
            #     bbox = elem.bbox
            #     center = (bbox[0][0] + bbox[1][0]) // 2, (bbox[0][1] + bbox[1][1]) // 2
            #     close = False
            #     for e in clickable_list:
            #         bbox = e.bbox
            #         center_ = (bbox[0][0] + bbox[1][0]) // 2, (bbox[0][1] + bbox[1][1]) // 2
            #         dist = (abs(center[0] - center_[0]) ** 2 + abs(center[1] - center_[1]) ** 2) ** 0.5
            #         if dist <= min_dist:
            #             close = True
            #             break
            #     if not close:
            #         elem_list.append(elem)
            elem_list = []
            traverse_tree_all(xml_path, elem_list, True)
            print(elem_list)
            draw_bbox_multi(screenshot_path, os.path.join(task_dir, f"{dir_name}_{round_count}_labeled.png"), elem_list,
                            dark_mode=dark_mode)
            image = os.path.join(task_dir, f"{dir_name}_{round_count}_labeled.png")
            if no_doc:
                prompt = re.sub(r"<ui_document>", "", prompts.task_template)
            else:
                ui_doc = ""
                for i, elem in enumerate(elem_list):
                    doc_path = os.path.join(docs_dir, f"{elem.uid}.txt")
                    if not os.path.exists(doc_path):
                        continue
                    ui_doc += f"Documentation of UI element labeled with the numeric tag '{i + 1}':\n"
                    doc_content = ast.literal_eval(open(doc_path, "r").read())
                    if doc_content["tap"]:
                        ui_doc += f"This UI element is clickable. {doc_content['tap']}\n\n"
                    if doc_content["text"]:
                        ui_doc += f"This UI element can receive text input. The text input is used for the following " \
                                  f"purposes: {doc_content['text']}\n\n"
                    if doc_content["long_press"]:
                        ui_doc += f"This UI element is long clickable. {doc_content['long_press']}\n\n"
                    if doc_content["v_swipe"]:
                        ui_doc += f"This element can be swiped directly without tapping. You can swipe vertically on " \
                                  f"this UI element. {doc_content['v_swipe']}\n\n"
                    if doc_content["h_swipe"]:
                        ui_doc += f"This element can be swiped directly without tapping. You can swipe horizontally on " \
                                  f"this UI element. {doc_content['h_swipe']}\n\n"
                print_with_color(f"Documentations retrieved for the current interface:\n{ui_doc}", "magenta")
                ui_doc = """
                You also have access to the following documentations that describes the functionalities of UI 
                elements you can interact on the screen. These docs are crucial for you to determine the target of your 
                next action. You should always prioritize these documented elements for interaction:""" + ui_doc
                prompt = re.sub(r"<ui_document>", ui_doc, prompts.task_template)
        xml_string = strip_xml(xml_path)
        # print(xml_string)
        prompt = re.sub(r"<task_description>", xml_string, prompt)
        prompt = re.sub(r"<task_description>", task_desc, prompt)
        prompt = re.sub(r"<last_act>", last_act, prompt)
        if detail:
            print_with_color("Thinking about what to do in the next step...", "yellow")
        rsp = mllm.invoke(prompt, images=[image])
        with open(log_path, "a") as logfile:
            log_item = {"step": round_count, "prompt": prompt, "image": f"{dir_name}_{round_count}_labeled.png",
                        "response": rsp}
            logfile.write(json.dumps(log_item) + "\n")
        if grid_on:
            res = parse_grid_rsp(rsp, detail)
        else:
            res = parse_explore_rsp(rsp, detail)
        act_name = res[0]
        if act_name == "FINISH":
            task_complete = True
            break
        if act_name == "ERROR":
            break
        last_act = res[-1]
        res = res[:-1]
        if act_name == "tap":
            _, area = res
            # print(area)
            tl, br = elem_list[area - 1].bbox
            x, y = (tl[0] + br[0]) // 2, (tl[1] + br[1]) // 2
            ret = controller.tap(x, y)
            if ret == "ERROR":
                print_with_color("ERROR: tap execution failed", "red")
                break
        elif act_name == "text":
            _, input_str = res
            ret = controller.text(input_str)
            if ret == "ERROR":
                print_with_color("ERROR: text execution failed", "red")
                break
        elif act_name == "long_press":
            _, area = res
            tl, br = elem_list[area - 1].bbox
            x, y = (tl[0] + br[0]) // 2, (tl[1] + br[1]) // 2
            ret = controller.long_press(x, y)
            if ret == "ERROR":
                print_with_color("ERROR: long press execution failed", "red")
                break
        elif act_name == "swipe":
            _, area, swipe_dir, dist = res
            tl, br = elem_list[area - 1].bbox
            x, y = (tl[0] + br[0]) // 2, (tl[1] + br[1]) // 2
            ret = controller.swipe(x, y, swipe_dir, dist)
            if ret == "ERROR":
                print_with_color("ERROR: swipe execution failed", "red")
                break
        elif act_name == "grid":
            grid_on = True
        elif act_name == "tap_grid" or act_name == "long_press_grid":
            _, area, subarea = res
            x, y = area_to_xy(area, subarea)
            if act_name == "tap_grid":
                ret = controller.tap(x, y)
                if ret == "ERROR":
                    print_with_color("ERROR: tap execution failed", "red")
                    break
            else:
                ret = controller.long_press(x, y)
                if ret == "ERROR":
                    print_with_color("ERROR: tap execution failed", "red")
                    break
        elif act_name == "swipe_grid":
            _, start_area, start_subarea, end_area, end_subarea = res
            start_x, start_y = area_to_xy(start_area, start_subarea)
            end_x, end_y = area_to_xy(end_area, end_subarea)
            ret = controller.swipe_precise((start_x, start_y), (end_x, end_y))
            if ret == "ERROR":
                print_with_color("ERROR: tap execution failed", "red")
                break
        if act_name != "grid":
            grid_on = False
        time.sleep(request_interval)

    if task_complete:
        return True, "success"
    else:
        if round_count == max_rounds:
            msg = "max_rounds"
        else:
            msg = "error"
        return False, msg
