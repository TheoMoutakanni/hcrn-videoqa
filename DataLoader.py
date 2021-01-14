# DISTRIBUTION STATEMENT A. Approved for public release: distribution unlimited.
#
# This material is based upon work supported by the Assistant Secretary of Defense for Research and
# Engineering under Air Force Contract No. FA8721-05-C-0002 and/or FA8702-15-D-0001. Any opinions,
# findings, conclusions or recommendations expressed in this material are those of the author(s) and
# do not necessarily reflect the views of the Assistant Secretary of Defense for Research and
# Engineering.
#
# © 2017 Massachusetts Institute of Technology.
#
# MIT Proprietary, Subject to FAR52.227-11 Patent Rights - Ownership by the contractor (May 2014)
#
# The software/firmware is provided to you on an As-Is basis
#
# Delivered to the U.S. Government with Unlimited Rights, as defined in DFARS Part 252.227-7013 or
# 7014 (Feb 2014). Notwithstanding any copyright notice, U.S. Government rights in this work are
# defined by DFARS 252.227-7013 or DFARS 252.227-7014 as detailed above. Use of this work other than
# as specifically authorized by the U.S. Government may violate any copyrights that exist in this
# work.

import numpy as np
import json
import pickle
import torch
import math
import h5py
from torch.utils.data import Dataset, DataLoader


def invert_dict(d):
    return {v: k for k, v in d.items()}


def load_vocab(path):
    with open(path, 'r') as f:
        vocab = json.load(f)
        vocab['question_idx_to_token'] = invert_dict(vocab['question_token_to_idx'])
        vocab['answer_idx_to_token'] = invert_dict(vocab['answer_token_to_idx'])
        vocab['question_answer_idx_to_token'] = invert_dict(vocab['question_answer_token_to_idx'])
    return vocab


class VideoQADataset(Dataset):

    def __init__(self, answers, ans_candidates, ans_candidates_len, questions, questions_len, video_ids, q_ids,
                 app_feature_h5, app_feat_id_to_index, motion_feature_h5, motion_feat_id_to_index,
                 question_feature_h5=None, question_feat_id_to_index=None, attention_mask=None,
                 app_feature_torch=None, motion_feature_torch=None):
        # convert data to tensor
        self.all_answers = answers
        if question_feature_h5 is None:
            self.all_questions = torch.LongTensor(np.asarray(questions))
        if attention_mask is not None:
            self.all_attention_mask = torch.LongTensor(np.asarray(attention_mask))
        else:
            self.all_attention_mask = None
        self.question_feature_h5 = question_feature_h5
        self.dataset_question_feature = None
        self.all_questions_len = torch.LongTensor(np.asarray(questions_len))
        self.all_video_ids = torch.LongTensor(np.asarray(video_ids))
        self.all_q_ids = q_ids

        self.app_feature_torch = app_feature_torch
        self.app_feature_h5 = app_feature_h5
        self.dataset_app_feature = None

        self.motion_feature_torch = motion_feature_torch
        self.motion_feature_h5 = motion_feature_h5
        self.dataset_motion_feature = None

        self.app_feat_id_to_index = app_feat_id_to_index
        self.motion_feat_id_to_index = motion_feat_id_to_index
        self.question_feat_id_to_index = question_feat_id_to_index

        if not np.any(ans_candidates):
            self.question_type = 'openended'
        else:
            self.question_type = 'mulchoices'
            self.all_ans_candidates = torch.LongTensor(np.asarray(ans_candidates))
            self.all_ans_candidates_len = torch.LongTensor(np.asarray(ans_candidates_len))

    def __getitem__(self, index):
        answer = self.all_answers[index] if self.all_answers is not None else None
        ans_candidates = torch.zeros(5)
        ans_candidates_len = torch.zeros(5)
        if self.question_type == 'mulchoices':
            ans_candidates = self.all_ans_candidates[index]
            ans_candidates_len = self.all_ans_candidates_len[index]

        video_idx = self.all_video_ids[index].item()
        question_idx = self.all_q_ids[index]
        app_index = self.app_feat_id_to_index[str(video_idx)]
        motion_index = self.motion_feat_id_to_index[str(video_idx)]

        if self.question_feature_h5 is None:
            question = self.all_questions[index]
            question_len = self.all_questions_len[index]
        else:
            # question_index = self.question_feat_id_to_index[str(video_idx)]
            if self.dataset_question_feature is None:
                self.dataset_question_feature = h5py.File(self.question_feature_h5, 'r') # , rdcc_nbytes=10 * (1024 ** 2), rdcc_nslots=5000, rdcc_w0=1)
            question = self.dataset_question_feature['question_features'][question_idx]
            question_len = self.dataset_question_feature['question_len'][question_idx]
            question = torch.from_numpy(question)
        
        if self.app_feature_torch is None:
            if self.dataset_app_feature is None:
                self.dataset_app_feature = h5py.File(self.app_feature_h5, 'r') # , rdcc_nbytes=10 * (1024 ** 2), rdcc_nslots=5000, rdcc_w0=1)
            appearance_feat = self.dataset_app_feature['resnet_features'][app_index]  # (8, 16, 2048)
            appearance_feat = torch.from_numpy(appearance_feat)
        else:
            appearance_feat = self.app_feature_torch[app_index]

        if self.motion_feature_torch is None:
            if self.dataset_motion_feature is None:
                self.dataset_motion_feature = h5py.File(self.motion_feature_h5, 'r') #, rdcc_nbytes=10 * (1024 ** 2), rdcc_nslots=5000, rdcc_w0=1)
            motion_feat = self.dataset_motion_feature['resnext_features'][motion_index]  # (8, 2048)
            motion_feat = torch.from_numpy(motion_feat)
        else:
            motion_feat = self.motion_feature_torch[motion_index]

        if self.all_attention_mask is not None:
            attention_mask = self.all_attention_mask[index]
            return (
                video_idx, question_idx, answer, ans_candidates, ans_candidates_len, appearance_feat, motion_feat, (question, attention_mask),
                question_len)

        return (
            video_idx, question_idx, answer, ans_candidates, ans_candidates_len, appearance_feat, motion_feat, question,
            question_len)

    def __len__(self):
        return len(self.all_video_ids)


class VideoQADataLoader(DataLoader):

    def __init__(self, **kwargs):
        vocab_json_path = str(kwargs.pop('vocab_json'))
        print('loading vocab from %s' % (vocab_json_path))
        vocab = load_vocab(vocab_json_path)

        question_pt_path = str(kwargs.pop('question_pt'))
        print('loading questions from %s' % (question_pt_path))
        question_type = kwargs.pop('question_type')
        questions = None
        attention_mask = None
        with open(question_pt_path, 'rb') as f:
            obj = pickle.load(f)
            if 'question_feat' not in kwargs and 'bert_model' not in kwargs:
                questions = obj['questions']
            if 'bert_model' in kwargs and 'precomputed' not in kwargs['bert_model']:
                questions = obj['questions_bert']
                attention_mask = obj['attention_mask']
            questions_len = obj['questions_len']
            video_ids = obj['video_ids']
            q_ids = obj['question_id']
            answers = obj['answers']
            glove_matrix = obj['glove']
            ans_candidates = np.zeros(5)
            ans_candidates_len = np.zeros(5)
            if question_type in ['action', 'transition']:
                ans_candidates = obj['ans_candidates']
                ans_candidates_len = obj['ans_candidates_len']
        if 'train_num' in kwargs:
            trained_num = kwargs.pop('train_num')
            if trained_num > 0:
                if 'question_feat' not in kwargs:
                    questions = questions[:trained_num]
                attention_mask = attention_mask[:trained_num]
                questions_len = questions_len[:trained_num]
                video_ids = video_ids[:trained_num]
                q_ids = q_ids[:trained_num]
                answers = answers[:trained_num]
                if question_type in ['action', 'transition']:
                    ans_candidates = ans_candidates[:trained_num]
                    ans_candidates_len = ans_candidates_len[:trained_num]
        if 'val_num' in kwargs:
            val_num = kwargs.pop('val_num')
            if val_num > 0:
                if 'question_feat' not in kwargs:
                    questions = questions[:val_num]
                attention_mask = attention_mask[:val_num]
                questions_len = questions_len[:val_num]
                video_ids = video_ids[:val_num]
                q_ids = q_ids[:val_num]
                answers = answers[:val_num]
                if question_type in ['action', 'transition']:
                    ans_candidates = ans_candidates[:val_num]
                    ans_candidates_len = ans_candidates_len[:val_num]
        if 'test_num' in kwargs:
            test_num = kwargs.pop('test_num')
            if test_num > 0:
                if 'question_feat' not in kwargs:
                    questions = questions[:test_num]
                attention_mask = attention_mask[:test_num]
                questions_len = questions_len[:test_num]
                video_ids = video_ids[:test_num]
                q_ids = q_ids[:test_num]
                answers = answers[:test_num]
                if question_type in ['action', 'transition']:
                    ans_candidates = ans_candidates[:test_num]
                    ans_candidates_len = ans_candidates_len[:test_num]

        print('loading appearance feature from %s' % (kwargs['appearance_feat']))
        with h5py.File(kwargs['appearance_feat'], 'r') as app_features_file:
            app_video_ids = app_features_file['ids'][()]
        app_feat_id_to_index = {str(id): i for i, id in enumerate(app_video_ids)}
        print('loading motion feature from %s' % (kwargs['motion_feat']))
        with h5py.File(kwargs['motion_feat'], 'r') as motion_features_file:
            motion_video_ids = motion_features_file['ids'][()]
        motion_feat_id_to_index = {str(id): i for i, id in enumerate(motion_video_ids)}
        self.app_feature_h5 = kwargs.pop('appearance_feat')
        self.motion_feature_h5 = kwargs.pop('motion_feat')

        app_feature_torch, motion_feature_torch = None, None
        if 'name2ids' in kwargs:
            out = load_activitynet_qa(
                kwargs['appearance_feat_torch'], kwargs['motion_feat_torch'], kwargs['name2ids'])
            app_feature_torch, motion_feature_torch, app_feat_id_to_index, motion_feat_id_to_index = out

        self.question_feature_h5 = None
        question_video_id_to_index = None
        if 'bert_model' in kwargs:
            if 'precomputed' in kwargs['bert_model']:
                with h5py.File(kwargs['question_feat'], 'r') as question_features_file:
                    question_video_ids = question_features_file['ids'][()]
                question_video_id_to_index = {str(id): i for i, id in enumerate(question_video_ids)}
                self.question_feature_h5 = kwargs.pop('question_feat')
            kwargs.pop('bert_model')

        self.dataset = VideoQADataset(answers, ans_candidates, ans_candidates_len, questions, questions_len,
                                      video_ids, q_ids,
                                      self.app_feature_h5, app_feat_id_to_index, self.motion_feature_h5,
                                      motion_feat_id_to_index, question_feature_h5=self.question_feature_h5,
                                      question_feat_id_to_index=question_video_id_to_index, attention_mask=attention_mask,
                                      app_feature_torch=app_feature_torch, motion_feature_torch=motion_feature_torch)

        self.vocab = vocab
        self.batch_size = kwargs['batch_size']
        self.glove_matrix = glove_matrix

        super().__init__(self.dataset, **kwargs)

    def __len__(self):
        return math.ceil(len(self.dataset) / self.batch_size)


def load_activitynet_qa(resnet_path, resnext_path, name2ids_path, num_frames=16, num_clips=24):
    with open(name2ids_path, 'rb') as handle:
        name2ids = pickle.load(handle)

    app_feature_torch = torch.load(resnet_path)
    motion_feature_torch = torch.load(resnext_path)

    for name in name2ids.keys():
        frames = app_feature_torch[name].shape[0]

    app_feat_id_to_index = {str(id): i for i, id in enumerate(motion_video_ids)}
    motion_feat_id_to_index = {str(id): i for i, id in enumerate(motion_video_ids)}
    return app_feature_torch, motion_feature_torch, app_feat_id_to_index, motion_feat_id_to_index
