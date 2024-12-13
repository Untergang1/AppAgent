import subprocess
import argparse
import os
import datetime
import time

from scripts.utils import print_with_color, load_config
from tasks import train_tasks
from scripts.task_executor import task_executor

os.environ["TOKENIZERS_PARALLELISM"] = "false"

configs = load_config()

parser = argparse.ArgumentParser(formatter_class=argparse.RawDescriptionHelpFormatter)
parser.add_argument("--model", default=configs["DEFAULT_MODEL"])
parser.add_argument("--app", default="system")
parser.add_argument("--start", default=0)
args = vars(parser.parse_args())
configs.update(args)

timestamp = int(time.time())
dir_name = datetime.datetime.fromtimestamp(timestamp).strftime(f"train_%Y-%m-%d_%H-%M-%S")
log_dir = os.path.join(".", "task_logs", "train_logs", dir_name)
os.makedirs(log_dir)
configs['log_dir'] = log_dir

configs['freeze_db'] = False

device = configs["DEVICE"]

subprocess.run(f"adb -s {device} shell input keyevent KEYCODE_HOME", shell=True)

start = int(args["start"])
for i in range(start, len(train_tasks)):
    task, max_rounds = train_tasks[i]
    print_with_color(f"Evaluate task{i}: \"{task}\" within {max_rounds} rounds", color="green")
    configs.update({
        "MAX_ROUNDS": max_rounds,
        "task": task,
        "task_num": i,
    })

    try:
        complete, msg = task_executor(configs)

        if complete:
            print_with_color(msg, "yellow")
        else:
            print_with_color(msg, "red")

    except Exception as e:
        print_with_color(f"An  error occurred while executing task: \"{task}\". Error: {e}", color="red")

    subprocess.run(f"adb -s {device} shell input keyevent KEYCODE_HOME", shell=True)

