import datetime
import time

import torch
from udpstream import UDPStream

stream = UDPStream(2300, torch.Size((640, 480)), "cuda:1")

interval = 0.5
t_0 = time.time()
while True:
    if t_0 + interval <= time.time():
        frame = stream.read()
        t_0 = time.time()
        print(
            f"Frame at {datetime.datetime.fromtimestamp(t_0).time()} with {frame.sum()} events"
        )

try:
    stream.stop_server()
except Exception as e:
    print(e)
