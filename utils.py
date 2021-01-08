import torch

def todevice(tensor, device, non_blocking=False):
    if isinstance(tensor, list) or isinstance(tensor, tuple):
        assert isinstance(tensor[0], torch.Tensor)
        return [todevice(t, device, non_blocking=non_blocking)
        		for t in tensor]
    elif isinstance(tensor, torch.Tensor):
        return tensor.to(device, non_blocking=non_blocking)