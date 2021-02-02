import torch
import h5py
import pickle
import numpy as np


def todevice(tensor, device, non_blocking=False):
    if isinstance(tensor, list) or isinstance(tensor, tuple):
        assert isinstance(tensor[0], torch.Tensor)
        return [todevice(t, device, non_blocking=non_blocking)
                for t in tensor]
    elif isinstance(tensor, torch.Tensor):
        return tensor.to(device, non_blocking=non_blocking)


def pad(x, size):
    total_frames, feat_size = x.shape
    clip_start = total_frames - size // 2
    clip_end = total_frames + size // 2
    pad = 0
    if clip_start < 0:
        clip_start = 0
    if clip_end > total_frames:
        clip_end = total_frames
    x = x[clip_start:clip_end]
    pad = size - (clip_end - clip_start)
    if pad > 0:
        added_frames_left = x[0:1].repeat(pad // 2, 1)
        added_frames_right = x[-1:].repeat(pad // 2 + pad % 2, 1)
        x = torch.cat((added_frames_left, x, added_frames_right), dim=0)
    return x


def process_activitynet(appearance_feat_path, motion_feat_path, name2ids_path, appearance_feat_h5, motion_feat_h5, num_frames=16, num_clips=24):
    with open(name2ids_path, 'rb') as handle:
        name2ids = pickle.load(handle)
    appearance_feat_dict = torch.load(appearance_feat_path)
    names = list(appearance_feat_dict.keys())
    feat_size = appearance_feat_dict[names[0]].shape[-1]
    dataset_size = len(appearance_feat_dict)

    with h5py.File(appearance_feat_h5, 'w') as fd:
        appearance_feat = fd.create_dataset('resnet_features', shape=(dataset_size, num_clips, num_frames, feat_size), dtype=np.float16)
        video_ids_dset = fd.create_dataset('ids', shape=(dataset_size,), dtype=np.int)
        for i, name in enumerate(names):
            appearance_feat[i] = pad(appearance_feat_dict.pop(name), num_frames * num_clips).view(24, 16, 2048)
            video_ids_dset[i] = name2ids[name]

    motion_feat_dict = torch.load(motion_feat_path)
    with h5py.File(motion_feat_h5, 'w') as fd:
        motion_feat = fd.create_dataset('resnext_features', shape=(dataset_size, num_clips, feat_size), dtype=np.float16)
        video_ids_dset = fd.create_dataset('ids', shape=(dataset_size,), dtype=np.int)
        for i, name in enumerate(names):
            motion_feat[i] = pad(motion_feat_dict.pop(name), num_clips).view(24, 2048)
            video_ids_dset[i] = name2ids[name]
