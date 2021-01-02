import time
import nltk
from collections import Counter

import pickle
import h5py
import numpy as np
from tqdm import tqdm

import torch
from transformers import RobertaConfig, RobertaTokenizer, RobertaModel


def encode(seq_tokens, token_to_idx, allow_unk=False):
    seq_idx = []
    for token in seq_tokens:
        if token not in token_to_idx:
            if allow_unk:
                token = '<UNK>'
            else:
                raise KeyError('Token "%s" not in vocab' % token)
        seq_idx.append(token_to_idx[token])
    return seq_idx


def decode(seq_idx, idx_to_token, delim=None, stop_at_end=True):
    tokens = []
    for idx in seq_idx:
        tokens.append(idx_to_token[idx])
        if stop_at_end and tokens[-1] == '<END>':
            break
    if delim is None:
        return tokens
    else:
        return delim.join(tokens)


def create_vocab(questions, answers,
                 answer_unk_token={'<UNK0>': 0, '<UNK1>': 1}, answer_top='all',
                 mulchoices=False):
    print('Building vocab')

    answer_token_to_idx = answer_unk_token
    question_answer_token_to_idx = {'<NULL>': 0, '<UNK>': 1}

    if not mulchoices:
        answer_cnt = {}
        for i, answer in enumerate(answers):
            answer_cnt[answer] = answer_cnt.get(answer, 0) + 1
        answer_counter = Counter(answer_cnt)
        if answer_top == 'all':
            answer_top = len(answer_counter)
        frequent_answers = answer_counter.most_common(answer_top)
        total_ans = sum(item[1] for item in answer_counter.items())
        total_freq_ans = sum(item[1] for item in frequent_answers)
        print("Number of unique answers:", len(answer_counter))
        print("Total number of answers:", total_ans)
        print("Top %i answers account for %f%%" %
              (len(frequent_answers), total_freq_ans * 100.0 / total_ans))

        for token, cnt in frequent_answers:
            answer_token_to_idx[token] = len(answer_token_to_idx)
        print('Get answer_token_to_idx, num: %d' % len(answer_token_to_idx))
    else:
        for candidates in ans_candidates:
            for ans in candidates:
                answer = ans.lower()
                for token in nltk.word_tokenize(answer):
                    if token not in answer_token_to_idx:
                        answer_token_to_idx[token] = len(answer_token_to_idx)
                    if token not in question_answer_token_to_idx:
                        question_answer_token_to_idx[token] = len(
                            question_answer_token_to_idx)

    question_token_to_idx = {'<NULL>': 0, '<UNK>': 1}
    for i, q in enumerate(questions):
        question = q.lower()[:-1]
        for token in nltk.word_tokenize(question):
            if token not in question_token_to_idx:
                question_token_to_idx[token] = len(question_token_to_idx)
            if mulchoices and token not in question_answer_token_to_idx:
                question_answer_token_to_idx[token] = len(
                    question_answer_token_to_idx)
    print('Get question_token_to_idx')
    print(len(question_token_to_idx))
    if mulchoices:
        print('Get question_answer_token_to_idx')
        print(len(question_answer_token_to_idx))

    vocab = {
        'question_token_to_idx': question_token_to_idx,
        'answer_token_to_idx': answer_token_to_idx,
        'question_answer_token_to_idx': question_answer_token_to_idx
    }

    return vocab


def encode_data(vocab, questions, answers, video_names, video_ids, mode, question_type, glove_pt, ans_candidates=None):
    # Encode all questions
    print('Encoding data')
    questions_encoded = []
    questions_len = []
    question_ids = []
    video_ids_tbw = []
    video_names_tbw = []
    all_answers = []
    if ans_candidates is not None:
        # multichoice
        all_answer_cands_encoded = []
        all_answer_cands_len = []

    for idx, question in enumerate(questions):
        question = question.lower()[:-1]
        question_tokens = nltk.word_tokenize(question)
        question_encoded = encode(
            question_tokens, vocab['question_token_to_idx'], allow_unk=True)
        questions_encoded.append(question_encoded)
        questions_len.append(len(question_encoded))
        question_ids.append(idx)
        video_ids_tbw.append(video_ids[idx])
        video_names_tbw.append(video_names[idx])

        if ans_candidates is not None:
            answer = int(answers[idx])
        elif question_type == "count":
            answer = max(int(answers[idx]), 1)
        else:
            if answers[idx] in vocab['answer_token_to_idx']:
                answer = vocab['answer_token_to_idx'][answers[idx]]
            elif mode in ['train']:
                answer = 0
            elif mode in ['val', 'test']:
                answer = 1

        all_answers.append(answer)

        if ans_candidates is not None:
            # answer candidates
            candidates = ans_candidates[idx]
            candidates_encoded = []
            candidates_len = []
            for ans in candidates:
                ans = ans.lower()
                ans_tokens = nltk.word_tokenize(ans)
                cand_encoded = utils.encode(
                    ans_tokens, vocab['question_answer_token_to_idx'], allow_unk=True)
                candidates_encoded.append(cand_encoded)
                candidates_len.append(len(cand_encoded))
            all_answer_cands_encoded.append(candidates_encoded)
            all_answer_cands_len.append(candidates_len)

    if ans_candidates is None:
        vocab_dict = 'question_token_to_idx'
    else:
        vocab_dict = 'question_answer_token_to_idx'

    # Pad encoded questions
    max_question_length = max(len(x) for x in questions_encoded)
    for qe in questions_encoded:
        while len(qe) < max_question_length:
            qe.append(vocab[vocab_dict]['<NULL>'])

    questions_encoded = np.asarray(questions_encoded, dtype=np.int32)
    questions_len = np.asarray(questions_len, dtype=np.int32)
    print(questions_encoded.shape)

    if ans_candidates is not None:
        # Pad encoded answer candidates
        max_answer_cand_length = max(
            max(len(x) for x in candidate) for candidate in all_answer_cands_encoded)
        for ans_cands in all_answer_cands_encoded:
            for ans in ans_cands:
                while len(ans) < max_answer_cand_length:
                    ans.append(vocab['question_answer_token_to_idx']['<NULL>'])
        all_answer_cands_encoded = np.asarray(
            all_answer_cands_encoded, dtype=np.int32)
        all_answer_cands_len = np.asarray(all_answer_cands_len, dtype=np.int32)
        print(all_answer_cands_encoded.shape)

    glove_matrix = None
    if mode == 'train':
        token_itow = {i: w for w, i in vocab[vocab_dict].items()}
        print("Load glove from %s" % glove_pt)
        glove = pickle.load(open(glove_pt, 'rb'))
        dim_word = glove['the'].shape[0]
        glove_matrix = []
        for i in range(len(token_itow)):
            vector = glove.get(token_itow[i], np.zeros((dim_word,)))
            glove_matrix.append(vector)
        glove_matrix = np.asarray(glove_matrix, dtype=np.float32)
        print(glove_matrix.shape)

    obj = {
        'questions': questions_encoded,
        'questions_len': questions_len,
        'question_id': question_ids,
        'video_ids': np.asarray(video_ids_tbw),
        'video_names': np.array(video_names_tbw),
        'answers': all_answers,
        'glove': glove_matrix,
    }

    if ans_candidates is not None:
        obj.update({
            'ans_candidates': all_answer_cands_encoded,
            'ans_candidates_len': all_answer_cands_len,
        })

    return obj


def encode_data_BERT(questions, answers, video_names, video_ids, cuda, batch_size, outfile, ans_candidates=None):
    tokenizer = RobertaTokenizer.from_pretrained("roberta-base")
    model = RobertaModel.from_pretrained("roberta-base")
    model.eval()
    if cuda:
        model.cuda()

    questions_tokens = tokenizer(questions, return_tensors="pt", padding=True)
    questions_input_ids = questions_tokens['input_ids']
    questions_attention_mask = questions_tokens['attention_mask']

    questions_len_bert = questions_attention_mask.sum(1).numpy()

    dataset_size = questions_input_ids.shape[0]
    T = questions_input_ids.shape[1]
    F = 768

    with h5py.File(outfile, 'w') as fd:
        feat_dset = fd.create_dataset('question_features', (dataset_size, T, F), dtype=np.float32)
        for batch in tqdm(range(len(questions) // batch_size + int(len(questions) % batch_size != 0))):
            batch_min, batch_max = batch*batch_size, min(len(questions),(batch+1)*batch_size)
            question_input_ids = questions_input_ids[batch_min:batch_max]
            question_attention_mask = questions_input_ids[batch_min:batch_max]
            if cuda:
                question_input_ids = question_input_ids.cuda()
                question_attention_mask = question_attention_mask.cuda()
            with torch.no_grad():
                question_bert = model(input_ids=question_input_ids, attention_mask=question_attention_mask).last_hidden_state.cpu().numpy()
            feat_dset[batch_min:batch_max] = question_bert
        len_dset = fd.create_dataset('question_len', (dataset_size,), dtype=np.int32)
        len_dset[:] = questions_len_bert
        video_ids_dset = fd.create_dataset('ids', shape=(dataset_size,), dtype=np.int)
        video_ids_dset[:] = video_ids


# --------------------------------------------------------
# Fast R-CNN
# Copyright (c) 2015 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Ross Girshick
# --------------------------------------------------------


class Timer(object):
    """A simple timer."""

    def __init__(self):
        self.total_time = 0.
        self.calls = 0
        self.start_time = 0.
        self.diff = 0.
        self.average_time = 0.

    def tic(self):
        # using time.time instead of time.clock because time time.clock
        # does not normalize for multithreading
        self.start_time = time.time()

    def toc(self, average=True):
        self.diff = time.time() - self.start_time
        self.total_time += self.diff
        self.calls += 1
        self.average_time = self.total_time / self.calls
        if average:
            return self.average_time
        else:
            return self.diff
