import time

import torch
from udpstream import UDPStream

stream = UDPStream(2300, torch.Size((640, 480)), "cuda:1")

interval = 0.003
frames = []
t_0 = time.time()
for i in range(100):
    while t_0 + interval <= time.time():
        pass
    frame = stream.read()
    frames.append(frame)
    t_0 = time.time()

print(frames[-1].shape)

try:
    stream.stop_server()
except Exception as e:
    print(e)
