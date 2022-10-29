## **NOTE**: This project is superceeded by [AEStream](https://github.com/norse/aestream/)

# <s>UDP DVS stream</s>

This project allows the streaming of dynamic vision system (DVS) 
event-camera data over UDP and into PyTorch.

It follows the UDP protocol used in AEStream.

## Installation

```bash
pip install -e [...]
```

## Usage
```python
import time
import torch
from udpstream import UDPStream

# This will immediately start a server in the background 
# and accumulate events
stream = UDPStream(2300, torch.Size((640, 480)), "cuda")

for i in range(100):
    frame = stream.read() # Read a single tensor
    time.sleep(0.010)     # Read frames every 10 ms
    ...
```

## Author

* Jens E. Pedersen <jeped@kth.se>

Thanks to Philipp Mondorff for inital code around the UDP socket.
