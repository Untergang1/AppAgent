import subprocess
import argparse
import os
import datetime
import time

from scripts.utils import print_with_color, load_config
from tasks import train_tasks
from scripts.task_executor import task_executor
from scripts.graph_database import GraphDatabase
from scripts.and_controller import AndroidController
from scripts.logger import set_logger

os.environ["TOKENIZERS_PARALLELISM"] = "false"


parser = argparse.ArgumentParser(formatter_class=argparse.RawDescriptionHelpFormatter)
parser.add_argument("--model", default=None)
parser.add_argument("--start", default=0)
parser.add_argument('--reset_db', action='store_true', default=False)
args = vars(parser.parse_args())

timestamp = int(time.time())
dir_name = datetime.datetime.fromtimestamp(timestamp).strftime(f"train_%Y-%m-%d_%H-%M-%S")
log_dir = os.path.join(".", "task_logs", "train_logs", dir_name)
os.makedirs(log_dir)
set_logger("Agent", file_path=os.path.join(log_dir, "task.log"))

configs = load_config()

configs['FREEZE_DB'] = False

device = configs["DEVICE"]
controller = AndroidController(device)

if args['reset_db']:
    print_with_color("Resetting database...", color="yellow")
    db = GraphDatabase()
    db.clear_database()

subprocess.run(f"adb -s {device} shell input keyevent KEYCODE_HOME", shell=True)

start = int(args["start"])
for i in range(start, len(train_tasks)):
    task_desc, max_rounds = train_tasks[i]
    print_with_color(f"Run task{i}: \"{task_desc}\" within {max_rounds} rounds", color="green")
    task = {
        "max_rounds": max_rounds,
        "task_desc": task_desc,
        "task_num": i,
    }

    try:
        complete, msg = task_executor(task=task, log_dir=log_dir, configs=configs)

        if complete:
            print_with_color(msg, "yellow")
        else:
            print_with_color(msg, "red")

    except Exception as e:
        print_with_color(f"An  error occurred while executing task: \"{task_desc}\". Error: {e}", color="red")

    controller.force_stop_app()

