import json
from datautils import utils
import os

import pickle
import numpy as np


def load_video_paths(args):
    ''' Load a list of (path,image_id tuples).'''
    video_paths = []
    video_ids = []
    modes = ['train', 'val', 'test']
    for mode in modes:
        with open(args.question_file.format(mode), 'r') as q_file:
            questions = json.load(q_file)
        [video_ids.append(question['video_name']) for question in questions]
    for mode in modes:
        with open(args.answer_file.format(mode), 'r') as a_file:
            answers = json.load(a_file)
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


def create_name2ids(args, force=False):
    if os.path.isfile(args.name2ids_pt.format(args.dataset, args.mode)) and not force:
        return

    modes = ['train', 'val', 'test']
    video_names = []
    for mode in modes:
        with open(args.annotation_file.format(mode, 'q'), 'r') as q_file:
            questions = json.load(q_file)
        [video_names.append(question['video_name']) for question in questions]

    name2ids = {}
    i = 0
    for name in set(video_names):
        name2ids[name] = i
        i += 1

    with open(args.name2ids_pt.format(args.dataset, args.mode), 'wb') as handle:
        pickle.dump(name2ids, handle, protocol=pickle.HIGHEST_PROTOCOL)


def process_questions(args):
    ''' Encode question tokens'''
    print('Loading data')
    with open(args.question_file, 'r') as q_file:
        questions_dict = json.load(q_file)
    with open(args.answer_file, 'r') as a_file:
        answers_dict = json.load(a_file)

    for q, a in zip(questions_dict, answers_dict):
        assert q['question_id'] == a['question_id']

    with open(args.name2ids_pt.format(args.dataset, args.mode), 'rb') as handle:
        name2ids = pickle.load(handle)

    questions = [d['question'] for d in questions_dict]
    answers = [d['answer'] for d in answers_dict]
    video_names = [d['video_name'] for d in questions_dict]
    video_ids = [name2ids[name] for name in video_names]

    
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

    if args.bert != "none":
        outfile = args.output_pt.format(args.dataset, args.dataset, args.mode)
        utils.encode_data_BERT(args.bert, questions, answers, video_ids, video_ids, args.cuda, args.batch_size, outfile, ans_candidates=None)

