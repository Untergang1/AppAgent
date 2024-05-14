import subprocess
import argparse
from scripts.eval_task import tasks

arg_desc = "AppAgent Evaluate"
parser = argparse.ArgumentParser(formatter_class=argparse.RawDescriptionHelpFormatter, description=arg_desc)
parser.add_argument("--model")
parser.add_argument("--device")
args = vars(parser.parse_args())

model = args["model"]
device = args["device"]


for i in range(len(tasks)):
    subprocess.run(f'python run.py -nd --model "{model}" --desc "{tasks[i]}" --device "{device}"', shell=True)
    subprocess.run(f"adb -s {device} shell input keyevent KEYCODE_HOME", shell=True)

