# Need to kill python background process later according to your os.For linux run(pkill -9 python)
import os
from jsonrpclib import ServerProxy
import time

os.system("python -u preprocess_data.py --num_households 4 > logs/logs_preprocess_data.txt &")
time.sleep(2)
data_server_ip="http://127.0.0.1:8000"
data_server=ServerProxy(data_server_ip)
while not data_server.server_status():
    time.sleep(10)
    continue

print("Read completed")

os.system("python -u aggregator.py > logs/logs_aggregator.txt &")
time.sleep(5)
print("aggregator started")

with open("participants.txt","r") as file:
    for line in file:
        line = line.strip()
        if len(line) == 0:
            break
        ip, port,my_id = line.split(" ")
        os.system(f"python -u ecc.py --port {port} --id {my_id} > logs/logs_ecc_{my_id}.txt &")
        print(f"ecc-{my_id} started")

print("All eccs started")