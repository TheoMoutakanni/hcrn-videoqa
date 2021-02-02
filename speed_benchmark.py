import timeit
import numpy as np
import torch
from tqdm import tqdm

import matplotlib.pyplot as plt
import matplotlib as mpl
import seaborn as sns
sns.set_style('ticks')
params = {
   'axes.labelsize': 8,
   #'text.fontsize': 8,
   'legend.fontsize': 10,
   'xtick.labelsize': 10,
   'ytick.labelsize': 10,
   'text.usetex': False,
   'figure.figsize': [3.5, 4.5],
   'axes.prop_cycle': mpl.cycler(color=sns.color_palette("coolwarm_r",3))
}
mpl.rcParams.update(params)

from utils import todevice
import model.HCRN as HCRN

def autolabel(rects):
    for rect in rects:
        height = rect.get_height()
        ax.annotate('{}'.format(height),
                    xy=(rect.get_x() + rect.get_width() / 2, height),
                    xytext=(0, 3),  # 3 points vertical offset
                    textcoords="offset points",
                    fontsize=8,
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
"""
fig, ax = plt.subplots()
ax.grid(axis='y', color="0.9", linestyle='-', linewidth=0.5)
rects1 = ax.bar(x, times, width)

ax.set_ylabel('Time (s)')
#plt.semilogy()
ax.set_xticks(x)
ax.set_xticklabels(models)

autolabel(rects1)

fig.tight_layout()
sns.despine(left=True)
plt.savefig('./img/time.png', dpi=200, transparent=False, bbox_inches='tight')
"""
#################

times_back = time_model(models, model_kwargs, batch_size, num_clips, backward=True)
times_back = np.round(times_back, 4)

fig, ax = plt.subplots()
ax.grid(axis='y', color="0.9", linestyle='-', linewidth=0.5)

rects1 = ax.bar(x - width/2, times_back, width, label='training w/ backprop')
ax.bar(0, 0, 0)
rects2 = ax.bar(x + width/2, times, width, label='inference')

ax.set_ylabel('Time (s)')
#plt.semilogy()
ax.set_xticks(x)
ax.set_xticklabels(models)
ax.legend()

autolabel(rects1)
autolabel(rects2)

fig.tight_layout()
sns.despine(left=True)
plt.savefig('./img/time.png', dpi=200, transparent=False, bbox_inches='tight')

#################

batch_size = 128

times_batch_128 = time_model(models, model_kwargs, batch_size, num_clips)
times_batch_128 = np.round(times_batch_128, 4)

#################

batch_size = 256

times_batch_256 = time_model(models, model_kwargs, batch_size, num_clips)
times_batch_256 = np.round(times_batch_256, 4)

fig, ax = plt.subplots()
ax.grid(axis='y', color="0.9", linestyle='-', linewidth=0.5)
ax.set_prop_cycle(mpl.cycler(color=sns.color_palette("coolwarm_r",3)))

rects1 = ax.bar(1.33*x - width, times_batch_256, width, label='batch_size=256')
rects2 = ax.bar(1.33*x, times_batch_128, width, label='batch_size=128')
rects3 = ax.bar(1.33*x + width, times, width, label='batch_size=32')

ax.set_ylabel('Time (s)')
#plt.semilogy()
ax.set_xticks(1.33*x)
ax.set_xticklabels(models)
ax.legend()

autolabel(rects1)
autolabel(rects2)
autolabel(rects3)

fig.tight_layout()
sns.despine(left=True)
plt.savefig('./img/time_batch.png', dpi=200, transparent=False, bbox_inches='tight')

#################

batch_size = 32
num_clips = 24

times_clips = time_model(models, model_kwargs, batch_size, num_clips)
times_clips = np.round(times_clips, 4)

#################

for hcrn_model in models:
    model_kwargs[hcrn_model]['subvids'] = 6
    model_kwargs[hcrn_model]['k_max_clip_level'] //= 2
num_clips = 24

times_levels = time_model(models, model_kwargs, batch_size, num_clips)
times_levels = np.round(times_levels, 4)

fig, ax = plt.subplots()
ax.grid(axis='y', color="0.9", linestyle='-', linewidth=0.5)

rects2 = ax.bar(x - width/2, times_clips, width, label='2 levels')
ax.bar(0, 0, 0)
rects3 = ax.bar(x + width/2, times_levels, width, label='3 levels')

ax.set_ylabel('Time (s)')
#plt.semilogy()
ax.set_xticks(x)
ax.set_xticklabels(models)
ax.legend()

autolabel(rects2)
autolabel(rects3)

fig.tight_layout()
sns.despine(left=True)
plt.savefig('./img/time_24clips.png', dpi=200, transparent=False, bbox_inches='tight')

################
