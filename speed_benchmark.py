import timeit
import numpy as np
import torch
from tqdm import tqdm

import matplotlib.pyplot as plt

from utils import todevice
import model.HCRN as HCRN

def autolabel(rects):
    for rect in rects:
        height = rect.get_height()
        ax.annotate('{}'.format(height),
                    xy=(rect.get_x() + rect.get_width() / 2, height),
                    xytext=(0, 3),  # 3 points vertical offset
                    textcoords="offset points",
                    ha='center', va='bottom')

def time_model(models, model_kwargs, batch_size, num_clips, backward=False):
    times = []
    for hcrn_model in models:
        print("Starting", hcrn_model)
        model = HCRN.HCRNNetwork(**model_kwargs[hcrn_model], hcrn_model=hcrn_model).to(device)

        criterion = torch.nn.CrossEntropyLoss().to(device)

        answers = torch.zeros(batch_size).long().to(device)

        ans_candidates = torch.zeros(batch_size, 5, 1000)
        ans_candidates_len = torch.zeros(batch_size, 5)

        appearance_feat = torch.zeros(batch_size, num_clips, 16, model_kwargs[hcrn_model]['vision_dim'])
        motion_feat = torch.zeros(batch_size, num_clips, model_kwargs[hcrn_model]['vision_dim'])
        question = torch.zeros(batch_size, 10).long()
        question_len = 10*torch.ones(batch_size).long()

        data = (ans_candidates, ans_candidates_len, appearance_feat, motion_feat, question, question_len)

        data = [todevice(x, device) for x in data]

        #pbar = tqdm(total=nb_it)

        def forward():
            logits = model(*data)
            if backward:
                loss = criterion(logits, answers)
                loss.backward()
            #pbar.update()

        time = timeit.Timer(forward).timeit(number=nb_it)/nb_it
        print(hcrn_model, time)

        times.append(time)
    return times

vocab = {}
vocab['question_token_to_idx'] = np.zeros(10000)
vocab['answer_token_to_idx'] = np.zeros(10000)

model_kwargs = {
    'basic': {
        'vision_dim': 2048,
        'module_dim': 512,
        'word_dim': 300,
        'k_max_frame_level': 16,
        'k_max_clip_level': 8,
        'spl_resolution': 1,
        'vocab': vocab,
        'question_type': "none",
        #'subvids': 6,
    },
    'fast': {
        'vision_dim': 2048,
        'module_dim': 512,
        'word_dim': 300,
        'k_max_frame_level': 16,
        'k_max_clip_level': 8,
        'spl_resolution': 1,
        'vocab': vocab,
        'question_type': "none",
        #'subvids': 6,
    },
    'faster': {
        'vision_dim': 2048,
        'module_dim': 512,
        'word_dim': 300,
        'k_max_frame_level': 8,
        'k_max_clip_level': 4,
        'spl_resolution': 1,
        'vocab': vocab,
        'question_type': "none",
        #'subvids': 6,
    },
}

device = "cuda"

nb_it = 100

batch_size = 32
num_clips = 8

models = ["basic", "fast", "faster"]

x = np.arange(len(models))  # the label locations
width = 0.35  # the width of the bars

#################

times = time_model(models, model_kwargs, batch_size, num_clips)
times = np.round(times, 4)

fig, ax = plt.subplots()
rects1 = ax.bar(x, times, width)

ax.set_ylabel('Time (s)')
#plt.semilogy()
ax.set_xticks(x)
ax.set_xticklabels(models)
ax.legend()

autolabel(rects1)

fig.tight_layout()
plt.savefig('./img/time.png', dpi=200)

#################

times_back = time_model(models, model_kwargs, batch_size, num_clips, backward=True)
times_back = np.round(times_back, 4)

fig, ax = plt.subplots()
rects1 = ax.bar(x, times_back, width)

ax.set_ylabel('Time (s)')
#plt.semilogy()
ax.set_xticks(x)
ax.set_xticklabels(models)
ax.legend()

autolabel(rects1)

fig.tight_layout()
plt.savefig('./img/time_back.png', dpi=200)

#################

batch_size = 128

times_batch = time_model(models, model_kwargs, batch_size, num_clips)
times_batch = np.round(times_batch, 4)

fig, ax = plt.subplots()
rects2 = ax.bar(x - width/2, times, width, label='batch_size=32')
rects3 = ax.bar(x + width/2, times_batch, width, label='batch_size=128')

ax.set_ylabel('Time (s)')
#plt.semilogy()
ax.set_xticks(x)
ax.set_xticklabels(models)
ax.legend()

autolabel(rects2)
autolabel(rects3)

fig.tight_layout()
plt.savefig('./img/time_batch.png', dpi=200)
#################

batch_size = 32
num_clips = 24

times_clips = time_model(models, model_kwargs, batch_size, num_clips)
times_clips = np.round(times_clips, 4)

#################

for hcrn_model in models:
    model_kwargs[hcrn_model]['subvids'] = 6
num_clips = 24

times_levels = time_model(models, model_kwargs, batch_size, num_clips)
times_levels = np.round(times_levels, 4)

fig, ax = plt.subplots()

rects2 = ax.bar(x - width/2, times_clips, width, label='2 levels')
rects3 = ax.bar(x + width/2, times_levels, width, label='3 levels')

ax.set_ylabel('Time (s)')
#plt.semilogy()
ax.set_xticks(x)
ax.set_xticklabels(models)
ax.legend()

autolabel(rects2)
autolabel(rects3)

fig.tight_layout()
plt.savefig('./img/time_24clips.png', dpi=200)

################
