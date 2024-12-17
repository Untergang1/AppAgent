import subprocess
import argparse
import os
import datetime
import time

from scripts.utils import print_with_color, load_config
from tasks import eval_tasks
from scripts.task_executor import task_executor
from scripts.and_controller import AndroidController
from scripts.logger import set_logger

os.environ["TOKENIZERS_PARALLELISM"] = "false"

parser = argparse.ArgumentParser(formatter_class=argparse.RawDescriptionHelpFormatter)
parser.add_argument("--model", default=None)
parser.add_argument("--start", default=0)
parser.add_argument('--no_db', action='store_true', default=False)
args = vars(parser.parse_args())

configs = load_config()

if args['model'] is not None:
    configs["MODEL"] = args['model']

if args["no_db"]:
    configs["USE_DB"] = False

db_str = "no_db_" if args["no_db"] else "with_db_"
timestamp = int(time.time())
dir_name = datetime.datetime.fromtimestamp(timestamp).strftime(f"eval_{db_str}%Y-%m-%d_%H-%M-%S")
log_dir = os.path.join(".", "task_logs", "eval_logs", dir_name)
os.makedirs(log_dir)
set_logger("Agent", file_path=os.path.join(log_dir, "task.log"))

device = configs["DEVICE"]
controller = AndroidController(device)

subprocess.run(f"adb -s {device} shell input keyevent KEYCODE_HOME", shell=True)

start = int(args["start"])
for i in range(start, len(eval_tasks)):
    task_desc, max_rounds = eval_tasks[i]
    print_with_color(f"Evaluate task{i}: \"{task_desc}\" within {max_rounds} rounds", color="green")
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

