from scripts.utils import print_with_color, parse_args, load_config
from scripts.task_executor import task_excutor

configs = load_config()
args = parse_args(configs)

complete, msg = task_excutor(args, configs)

if complete:
    print_with_color(msg, "yellow")
else:
    print_with_color(msg, "red")