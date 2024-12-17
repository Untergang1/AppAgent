import os
import re
import sys
import time
from typing import Dict
import logging

from . import prompts
from .and_controller import chose_device, AndroidController, traverse_tree
from .model import parse_explore_rsp, parse_grid_rsp, chose_model
from .utils import print_with_color, draw_bbox_multi, draw_grid
from .graph_database import GraphDatabase, calculate_hash


def task_executor(task: Dict, log_dir: str, configs: Dict):
    mllm = chose_model(configs)
    if mllm == None:
        print_with_color(f"ERROR: Unsupported model type {configs['MODEL']}!", "red")
        sys.exit()

    task_desc = task["task_desc"]
    max_rounds = task["max_rounds"]
    task_num = task["task_num"]

    device = configs["DEVICE"]
    min_dist = configs["MIN_DIST"]
    request_interval = configs["REQUEST_INTERVAL"]
    dark_mode = configs["DARK_MODE"]
    freeze_db = configs["FREEZE_DB"]
    use_db = configs["USE_DB"]

    dir_name = f"task_{task_num}"
    task_dir = os.path.join(log_dir, dir_name)
    os.mkdir(task_dir)

    logger = logging.getLogger("Agent")
    logger.info(f"Task {task_num}: {task_desc}")

    if use_db:
        print_with_color("proceed WITH graph database.", "yellow")
    else:
        print_with_color("proceed WITHOUT graph database.", "yellow")

    db = GraphDatabase()

    if not device:
        device = chose_device()

    controller = AndroidController(device)
    width, height = controller.get_device_size()
    if not width and not height:
        print_with_color("ERROR: Invalid device size!", "red")
        sys.exit()
    print_with_color(f"Screen resolution of {device}: {width}x{height}", "yellow")

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
        logger.info(f"Round {round_count}")
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

            pre_node_id = cur_node_id
            cur_node_id = calculate_hash(xml_path)

            if use_db:
                tar = task_desc if intermediate_goal is None else intermediate_goal
                related_paths = db.query_realted_paths(cur_node_id, tar)
                if len(related_paths) == 0:
                    paths_doc = ""
                    logger.info("No related path found")
                else:
                    paths_doc = ""
                    for i, path in enumerate(related_paths, start=1):
                        paths_doc += f"{i}: {path}\n"
                    logger.info(f"Find related paths:\n{paths_doc}")
                    paths_doc = "Here are some possible action paths:\n" + paths_doc
            else:
                paths_doc = ""

            prompt = re.sub(r"<related_paths>", paths_doc, prompts.task_template)

        prompt = re.sub(r"<task_description>", task_desc, prompt)
        print_with_color("Thinking about what to do in the next step...", "yellow")

        status, rsp = mllm.get_model_response(prompt, [image])
        logger.info(f"RESPONSE:\n{rsp}")

        if status:
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

            if not freeze_db:
                db.create_or_update_node(cur_node_id, fun_desc, xml_path)
                if pre_node_id is not None and act_desc is not None:
                    db.create_or_update_relationship(pre_node_id, cur_node_id, act_desc)

            act_desc = action

            time.sleep(request_interval)
        else:
            print_with_color(str(rsp), "red")
            break

    if task_complete:
        logger.info(f"Task completed in {round_count} rounds")
        return True, "success"
    else:
        if round_count == max_rounds:
            msg = "up to max_rounds"
            logger.info("up to max_rounds")
        else:
            msg = "error"
            logger.error("Task exit with error")
        return False, msg
