import os
import argparse
import time
import datetime

from scripts.utils import print_with_color, load_config
from scripts.task_executor import task_executor
from scripts.and_controller import AndroidController
from scripts.logger import set_logger

os.environ["TOKENIZERS_PARALLELISM"] = "false"


arg_desc = " Run AppAgent"
parser = argparse.ArgumentParser(formatter_class=argparse.RawDescriptionHelpFormatter, description=arg_desc)
parser.add_argument("--model", default=None, help="inference model")
parser.add_argument("--max_rounds", type=int, default=5, help="inference model")
parser.add_argument('--no_db', action='store_true', default=False)
parser.add_argument('--update_db', action='store_true', default=False)
parser.add_argument("--task")
args = vars(parser.parse_args())

configs = load_config()

if args["model"] is not None:
    configs["MODEl"] = args["model"]

if args["no_db"]:
    configs["USE_DB"] = False

if args["update_db"]:
    configs["FREEZE_DB"] = False


timestamp = int(time.time())
dir_name = datetime.datetime.fromtimestamp(timestamp).strftime(f"run_%Y-%m-%d_%H-%M-%S")
log_dir = os.path.join(".", "task_logs", "run_logs", dir_name)
os.makedirs(log_dir)
set_logger("Agent", file_path=os.path.join(log_dir, "task.log"))

device = configs["DEVICE"]
controller = AndroidController(device)

task = {
    "max_rounds": args["max_rounds"],
    "task_desc": args['task'],
    "task_num": 0,
}

complete, msg = task_executor(task=task, log_dir=log_dir, configs=configs)

controller.force_stop_app()

if complete:
    print_with_color(msg, "yellow")
else:
    print_with_color(msg, "red")
