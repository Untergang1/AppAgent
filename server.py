import time
from http.server import BaseHTTPRequestHandler, HTTPServer
import subprocess

from scripts.utils import print_with_color, load_config
from scripts.task_executor import task_executor
import argparse

def parse_args(configs):
    arg_desc = " Run AppAgent"
    parser = argparse.ArgumentParser(formatter_class=argparse.RawDescriptionHelpFormatter, description=arg_desc)
    parser.add_argument("--app", default="system")
    parser.add_argument("--root_dir", default="./")
    parser.add_argument("--model", default=configs["DEFAULT_MODEL"], help="inference model")
    parser.add_argument("--nodoc", "-n", action="store_true", help="proceed without docs")
    parser.add_argument("--detail", "-d", action="store_true", help="show detailed process")
    args = vars(parser.parse_args())
    return args

configs = load_config()
args = parse_args(configs)
configs.update(args)

device = configs["DEVICE"]

def get_ip_address(interface='en0'):
    result = subprocess.run(['ipconfig', 'getifaddr', interface], capture_output=True, text=True, check=True)
    ip_address = result.stdout.strip()
    return ip_address

class SimpleHTTPRequestHandler(BaseHTTPRequestHandler):
    def do_POST(self):
        content_length = int(self.headers['Content-Length'])
        post_data = self.rfile.read(content_length)

        received_data = post_data.decode('utf-8')
        print(received_data)

        if received_data != "":
            configs.update({"desc": received_data})
            subprocess.run(f"adb -s {device} shell input keyevent KEYCODE_HOME", shell=True)
            subprocess.run(f"adb -s {device} shell input keyevent KEYCODE_HOME", shell=True)
            time.sleep(0.5)
            task_executor(configs)

        self.send_response(200)
        self.send_header('Content-type', 'text/plain')
        self.end_headers()
        self.wfile.write(b'POST received successfully')

server_class=HTTPServer
handler_class=SimpleHTTPRequestHandler
address = get_ip_address()
port = 8000

server_address = (address, port)
httpd = server_class(server_address, handler_class)
print(f"Server started on {address}:{port}")

httpd.serve_forever()

