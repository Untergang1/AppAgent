import ast
import datetime
import json
import os
import re
import sys
import time

from . import prompts
from .and_controller import chose_device, AndroidController, traverse_tree
from .model import parse_explore_rsp, parse_grid_rsp, chose_model
from .utils import print_with_color, draw_bbox_multi, draw_grid
from .graph_database import GraphDatabase, calculate_hash

def task_executor(configs):
    mllm = chose_model(configs)
    if mllm == None:
        print_with_color(f"ERROR: Unsupported model type {configs['model']}!", "red")
        sys.exit()

    app = configs["app"]
    root_dir = configs["root_dir"]
    device = configs["DEVICE"]
    task_desc = configs["desc"]
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

    print_with_color("proceed with graph database.", "yellow")
    db = GraphDatabase()

    if not device:
        device = chose_device()

    controller = AndroidController(device)
    width, height = controller.get_device_size()
    if not width and not height:
        print_with_color("ERROR: Invalid device size!", "red")
        sys.exit()
    print_with_color(f"Screen resolution of {device}: {width}x{height}", "yellow")

    if not task_desc:
        print_with_color("Please enter the description of the task you want me to complete in a few sentences:", "blue")
        task_desc = input()

    round_count = 0
    # last_act = "None"
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

    pre_node_id = None
    cur_node_id = None
    act_desc = None
    intermediate_goal = None

    while round_count < max_rounds:
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
            clickable_list = []
            focusable_list = []
            traverse_tree(xml_path, clickable_list, "clickable", True)
            traverse_tree(xml_path, focusable_list, "focusable", True)
            elem_list = clickable_list.copy()
            for elem in focusable_list:
                bbox = elem.bbox
                center = (bbox[0][0] + bbox[1][0]) // 2, (bbox[0][1] + bbox[1][1]) // 2
                close = False
                for e in clickable_list:
                    bbox = e.bbox
                    center_ = (bbox[0][0] + bbox[1][0]) // 2, (bbox[0][1] + bbox[1][1]) // 2
                    dist = (abs(center[0] - center_[0]) ** 2 + abs(center[1] - center_[1]) ** 2) ** 0.5
                    if dist <= min_dist:
                        close = True
                        break
                if not close:
                    elem_list.append(elem)
            draw_bbox_multi(screenshot_path, os.path.join(task_dir, f"{dir_name}_{round_count}_labeled.png"), elem_list,
                            dark_mode=dark_mode)
            image = os.path.join(task_dir, f"{dir_name}_{round_count}_labeled.png")

            tar = task_desc if intermediate_goal is None else intermediate_goal
            related_paths = db.query(cur_node_id, tar)
            if len(related_paths) == 0:
                paths_doc = ""
            else:
                paths_doc = "Here are some possible action paths:\n"
                for i, path in enumerate(related_paths, start=1):
                    paths_doc += f"{i}: {path}\n"
            prompt = re.sub(r"<related_paths>", paths_doc, prompts.task_template)

        prompt = re.sub(r"<task_description>", task_desc, prompt)
        # prompt = re.sub(r"<last_act>", last_act, prompt)
        print_with_color("Thinking about what to do in the next step...", "yellow")
        status, rsp = mllm.get_model_response(prompt, [image])
        if status:
            with open(log_path, "a") as logfile:
                log_item = {"step": round_count, "prompt": prompt, "image": f"{dir_name}_{round_count}_labeled.png",
                            "response": rsp}
                logfile.write(json.dumps(log_item) + "\n")
            if grid_on:
                res = parse_grid_rsp(rsp)
            else:
                observation, intermediate_goal, action, parsed_function = parse_explore_rsp(rsp)
            act_name = parsed_function[0]
            if act_name == "FINISH":
                task_complete = True
                break
            if act_name == "ERROR":
                break

            if act_name == "tap":
                _, area = parsed_function
                tl, br = elem_list[area - 1].bbox
                x, y = (tl[0] + br[0]) // 2, (tl[1] + br[1]) // 2
                ret = controller.tap(x, y)
                if ret == "ERROR":
                    print_with_color("ERROR: tap execution failed", "red")
                    break
            elif act_name == "text":
                _, input_str = parsed_function
                ret = controller.text(input_str)
                if ret == "ERROR":
                    print_with_color("ERROR: text execution failed", "red")
                    break
            elif act_name == "long_press":
                _, area = parsed_function
                tl, br = elem_list[area - 1].bbox
                x, y = (tl[0] + br[0]) // 2, (tl[1] + br[1]) // 2
                ret = controller.long_press(x, y)
                if ret == "ERROR":
                    print_with_color("ERROR: long press execution failed", "red")
                    break
            elif act_name == "swipe":
                _, area, swipe_dir, dist = parsed_function
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
            elif act_name == "grid":
                grid_on = True

            fun_desc = observation
            pre_node_id = cur_node_id
            cur_node_id = calculate_hash(xml_path)
            db.create_or_update_node(cur_node_id, fun_desc, xml_path)

            if pre_node_id is not None and act_desc is not None:
                db.create_or_update_relationship(pre_node_id, cur_node_id, act_desc)

            act_desc = action

            time.sleep(request_interval)
        else:
            print_with_color(str(rsp), "red")
            break

    if task_complete:
        return True, "success"
    else:
        if round_count == max_rounds:
            msg = "max_rounds"
        else:
            msg = "error"
        return False, msg
