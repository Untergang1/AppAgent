from scripts.utils import print_with_color, load_config
from scripts.task_executor import task_executor
import argparse

import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"

def parse_args(configs):
    arg_desc = " Run AppAgent"
    parser = argparse.ArgumentParser(formatter_class=argparse.RawDescriptionHelpFormatter, description=arg_desc)
    parser.add_argument("--app", default="system")
    parser.add_argument("--model", default=configs["DEFAULT_MODEL"], help="inference model")
    parser.add_argument("--detail", "-d", action="store_true", help="show detailed process")
    parser.add_argument("--task")
    args = vars(parser.parse_args())
    return args

configs = load_config()
args = parse_args(configs)
configs.update(args)

complete, msg = task_executor(configs)

if complete:
    print_with_color(msg, "yellow")
else:
    print_with_color(msg, "red")