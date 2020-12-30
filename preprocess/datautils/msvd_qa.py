import json
from datautils import utils

import pickle
import numpy as np


def load_video_paths(args):
    ''' Load a list of (path,image_id tuples).'''
    video_paths = []
    video_ids = []
    modes = ['train', 'val', 'test']
    for mode in modes:
        with open(args.annotation_file.format(mode), 'r') as anno_file:
            instances = json.load(anno_file)
        [video_ids.append(instance['video_id']) for instance in instances]
    video_ids = set(video_ids)
    with open(args.video_name_mapping, 'r') as mapping:
        mapping_pairs = mapping.read().split('\n')
    mapping_dict = {}
    for idx in range(len(mapping_pairs)):
        cur_pair = mapping_pairs[idx].split(' ')
        mapping_dict[cur_pair[1]] = cur_pair[0]
    for video_id in video_ids:
        video_paths.append(
            (args.video_dir + 'YouTubeClips/{}.avi'.format(mapping_dict['vid' + str(video_id)]), video_id))
    return video_paths


def process_questions(args):
    ''' Encode question tokens'''
    print('Loading data')
    with open(args.annotation_file, 'r') as dataset_file:
        instances = json.load(dataset_file)

    questions = [instance['question'] for instance in instances]
    answers = [instance['answer'] for instance in instances]
    video_ids = [instance['video_id'] for instance in instances]

    # Either create the vocab or load it from disk
    if args.mode in ['train']:
        vocab = utils.create_vocab(
            questions,
            answers,
            answer_unk_token={'<UNK0>': 0, '<UNK1>': 1},
            answer_top=args.answer_top,
            mulchoices=False)
        print('Write into %s' % args.vocab_json.format(args.dataset, args.dataset))
        with open(args.vocab_json.format(args.dataset, args.dataset), 'w') as f:
            json.dump(vocab, f, indent=4)
    else:
        print('Loading vocab')
        with open(args.vocab_json.format(args.dataset, args.dataset), 'r') as f:
            vocab = json.load(f)

    obj = utils.encode_data(vocab, questions, answers, video_ids, video_ids, args.mode, "none", args.glove_pt)

    print('Writing', args.output_pt.format(
        args.dataset, args.dataset, args.mode))
    with open(args.output_pt.format(args.dataset, args.dataset, args.mode), 'wb') as f:
        pickle.dump(obj, f)
