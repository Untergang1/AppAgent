import subprocess

interface = 'en0'
result = subprocess.run(['ipconfig', 'getifaddr', interface], capture_output=True, text=True, check=True)
print(result.stdout.strip())