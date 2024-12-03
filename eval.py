import subprocess
import argparse
import os

from scripts.utils import print_with_color, load_config
from scripts.eval_task import tasks
from scripts.task_executor import task_executor

os.environ["TOKENIZERS_PARALLELISM"] = "false"

configs = load_config()

arg_desc = "AppAgent Evaluate"
parser = argparse.ArgumentParser(formatter_class=argparse.RawDescriptionHelpFormatter, description=arg_desc)
parser.add_argument("--model", default=configs["DEFAULT_MODEL"])
parser.add_argument("--app", default="system")
parser.add_argument("--start", default=0)
args = vars(parser.parse_args())
configs.update(args)

device = configs["DEVICE"]

subprocess.run(f"adb -s {device} shell input keyevent KEYCODE_HOME", shell=True)

start = int(args["start"])
for i in range(start, len(tasks)):
    task, max_rounds = tasks[i]
    print_with_color(f"Evaluate task: \"{task}\" within {max_rounds} rounds", color="green")
    configs.update({
        "MAX_ROUNDS": max_rounds,
        "task": task,
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

