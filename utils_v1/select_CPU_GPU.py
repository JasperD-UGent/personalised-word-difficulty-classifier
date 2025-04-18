import sys
from tensorflow.python.client import device_lib
from typing import Optional


def select_device(prefer_gpu: bool, id_gpu: Optional[int]):
    local_device_protos = device_lib.list_local_devices()
    l_gpus = [x.name for x in local_device_protos if x.device_type == "GPU"]

    if (len(l_gpus) > 0) and prefer_gpu:
        return l_gpus[id_gpu]
    else:
        return [x.name for x in local_device_protos if x.device_type == "CPU"][0]
