import os
import pandas as pd
import json
from datautils import utils

import pickle
import numpy as np


def load_video_paths(args):
    ''' Load a list of (path,image_id tuples).'''
    input_paths = []
    annotation = pd.read_csv(args.annotation_file.format(args.question_type), delimiter='\t')
    gif_names = list(annotation['gif_name'])
    keys = list(annotation['key'])
    print("Number of questions: {}".format(len(gif_names)))
    for idx, gif in enumerate(gif_names):
        gif_abs_path = os.path.join(args.video_dir, ''.join([gif, '.gif']))
        input_paths.append((gif_abs_path, keys[idx]))
    input_paths = list(set(input_paths))
    print("Number of unique videos: {}".format(len(input_paths)))

    return input_paths


def openeded_encoding_data(args, vocab, questions, video_names, video_ids, answers, mode='train'):
    obj = utils.encode_data(vocab, questions, answers, video_names, video_ids, mode, args.question_type, args.glove_pt)

    print('Writing ', args.output_pt.format(args.question_type, args.question_type, mode))
    with open(args.output_pt.format(args.question_type, args.question_type, mode), 'wb') as f:
        pickle.dump(obj, f)

    if args.bert != "none":
        outfile = args.output_pt.format(args.question_type, args.question_type, mode)
        outfile = outfile.replace('.pt', '_feat.h5')
        utils.encode_data_BERT(questions, answers, video_names, video_ids, args.cuda, args.batch_size, outfile, ans_candidates=None)

def multichoice_encoding_data(args, vocab, questions, video_names, video_ids, answers, ans_candidates, mode='train'):
    obj = utils.encode_data(vocab, questions, answers, video_names, video_ids, mode, args.question_type, args.glove_pt, ans_candidates)

    print('Writing ', args.output_pt.format(args.question_type, args.question_type, mode))
    with open(args.output_pt.format(args.question_type, args.question_type, mode), 'wb') as f:
        pickle.dump(obj, f)

def process_questions_openended(args):
    print('Loading data')
    if args.mode in ["train"]:
        csv_data = pd.read_csv(args.annotation_file.format("Train", args.question_type), delimiter='\t')
    else:
        csv_data = pd.read_csv(args.annotation_file.format("Test", args.question_type), delimiter='\t')
    csv_data = csv_data.iloc[np.random.permutation(len(csv_data))]
    questions = list(csv_data['question'])
    answers = list(csv_data['answer'])
    video_names = list(csv_data['gif_name'])
    video_ids = list(csv_data['key'])

    print('number of questions: %s' % len(questions))
    # Either create the vocab or load it from disk
    if args.mode in ['train']:
        vocab = utils.create_vocab(
            questions,
            answers,
            answer_unk_token={'<UNK0>': 0},
            answer_top='all',
            mulchoices=False)
        if args.question_type == 'count':
            vocab['answer_token_to_idx'] = {'<UNK>': 0}

        print('Write into %s' % args.vocab_json.format(args.question_type, args.question_type))
        with open(args.vocab_json.format(args.question_type, args.question_type), 'w') as f:
            json.dump(vocab, f, indent=4)

        # split 10% of questions for evaluation
        split = int(0.9 * len(questions))
        train_questions = questions[:split]
        train_answers = answers[:split]
        train_video_names = video_names[:split]
        train_video_ids = video_ids[:split]

        val_questions = questions[split:]
        val_answers = answers[split:]
        val_video_names = video_names[split:]
        val_video_ids = video_ids[split:]

        openeded_encoding_data(args, vocab, train_questions, train_video_names, train_video_ids, train_answers, mode='train')
        openeded_encoding_data(args, vocab, val_questions, val_video_names, val_video_ids, val_answers, mode='val')
    else:
        print('Loading vocab')
        with open(args.vocab_json.format(args.question_type, args.question_type), 'r') as f:
            vocab = json.load(f)
        openeded_encoding_data(args, vocab, questions, video_names, video_ids, answers, mode='test')




def process_questions_mulchoices(args):
    print('Loading data')
    if args.mode in ["train", "val"]:
        csv_data = pd.read_csv(args.annotation_file.format("Train", args.question_type), delimiter='\t')
    else:
        csv_data = pd.read_csv(args.annotation_file.format("Test", args.question_type), delimiter='\t')
    csv_data = csv_data.iloc[np.random.permutation(len(csv_data))]
    questions = list(csv_data['question'])
    answers = list(csv_data['answer'])
    video_names = list(csv_data['gif_name'])
    video_ids = list(csv_data['key'])
    ans_candidates = np.asarray(
        [csv_data['a1'], csv_data['a2'], csv_data['a3'], csv_data['a4'], csv_data['a5']])
    ans_candidates = ans_candidates.transpose()
    print(ans_candidates.shape)
    # ans_candidates: (num_ques, 5)
    print('number of questions: %s' % len(questions))
    # Either create the vocab or load it from disk
    if args.mode in ['train']:
        vocab = utils.create_vocab(
            questions,
            ans_candidates,
            answer_unk_token={'<UNK0>': 0, '<UNK1>': 1},
            answer_top='all',
            mulchoices=True)

        print('Write into %s' % args.vocab_json.format(args.question_type, args.question_type))
        with open(args.vocab_json.format(args.question_type, args.question_type), 'w') as f:
            json.dump(vocab, f, indent=4)

        # split 10% of questions for evaluation
        split = int(0.9 * len(questions))
        train_questions = questions[:split]
        train_answers = answers[:split]
        train_video_names = video_names[:split]
        train_video_ids = video_ids[:split]
        train_ans_candidates = ans_candidates[:split, :]

        val_questions = questions[split:]
        val_answers = answers[split:]
        val_video_names = video_names[split:]
        val_video_ids = video_ids[split:]
        val_ans_candidates = ans_candidates[split:, :]

        multichoice_encoding_data(args, vocab, train_questions, train_video_names, train_video_ids, train_answers, train_ans_candidates, mode='train')
        multichoice_encoding_data(args, vocab, val_questions, val_video_names, val_video_ids, val_answers,
                                  val_ans_candidates, mode='val')
    else:
        print('Loading vocab')
        with open(args.vocab_json.format(args.question_type, args.question_type), 'r') as f:
            vocab = json.load(f)
        multichoice_encoding_data(args, vocab, questions, video_names, video_ids, answers,
                                  ans_candidates, mode='test')
