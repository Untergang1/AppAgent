import time
import os
import subprocess
import argparse
import datetime
from http.server import BaseHTTPRequestHandler, HTTPServer

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

def get_ip_address(interface='en0'):
    result = subprocess.run(['ipconfig', 'getifaddr', interface], capture_output=True, text=True, check=True)
    ip_address = result.stdout.strip()
    return ip_address


device = configs["DEVICE"]
controller = AndroidController(device)


class SimpleHTTPRequestHandler(BaseHTTPRequestHandler):
    def do_POST(self):
        content_length = int(self.headers['Content-Length'])
        post_data = self.rfile.read(content_length)

        received_data = post_data.decode('utf-8')
        print_with_color(f"Received task description: {received_data}", "blue")

        if received_data:
            # 1. 设置日志目录
            timestamp = int(time.time())
            dir_name = datetime.datetime.fromtimestamp(timestamp).strftime(f"server_%Y-%m-%d_%H-%M-%S")
            log_dir = os.path.join(".", "task_logs", "run_logs", dir_name)
            os.makedirs(log_dir, exist_ok=True)
            set_logger("Agent", file_path=os.path.join(log_dir, "task.log"))

            # 2. 构造任务字典（新接口要求）
            task = {
                "max_rounds": configs.get("max_rounds", 5),
                "task_desc": received_data,
                "task_num": 0,
            }

            # 3. 回到主界面，准备执行
            subprocess.run(f"adb -s {device} shell input keyevent KEYCODE_HOME", shell=True)
            subprocess.run(f"adb -s {device} shell input keyevent KEYCODE_HOME", shell=True)
            time.sleep(0.5)

            # 4. 执行任务
            complete, msg = task_executor(task=task, log_dir=log_dir, configs=configs)
            controller.force_stop_app()

            if complete:
                print_with_color(msg, "yellow")
            else:
                print_with_color(msg, "red")

        # 返回 HTTP 响应
        self.send_response(200)
        self.send_header('Content-type', 'text/plain')
        self.end_headers()
        self.wfile.write(b'POST received successfully')


# 启动 HTTP Server
server_class = HTTPServer
handler_class = SimpleHTTPRequestHandler

address = get_ip_address()
port = 8000
server_address = (address, port)

httpd = server_class(server_address, handler_class)
print_with_color(f"Server started on http://{address}:{port}", "green")

httpd.serve_forever()
