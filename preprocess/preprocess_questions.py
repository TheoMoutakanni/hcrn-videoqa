import argparse
import numpy as np
import os

from datautils import tgif_qa
from datautils import msrvtt_qa
from datautils import msvd_qa
from datautils import activitynet_qa

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', default='tgif-qa', choices=['tgif-qa', 'msrvtt-qa', 'msvd-qa', 'activitynet-qa'], type=str)
    parser.add_argument('--answer_top', default=4000, type=int)
    parser.add_argument('--glove_pt',
                        help='glove pickle file, should be a map whose key are words and value are word vectors represented by numpy arrays. Only needed in train mode')
    parser.add_argument('--output_pt', type=str, default='data/{}/{}_{}_questions.pt')
    parser.add_argument('--vocab_json', type=str, default='data/{}/{}_vocab.json')
    parser.add_argument('--mode', choices=['train', 'val', 'test'])
    parser.add_argument('--question_type', choices=['frameqa', 'action', 'transition', 'count', 'none'], default='none')
    parser.add_argument('--seed', type=int, default=666)
    parser.add_argument('--bert', type=str, default="none")
    parser.add_argument('--cuda', action='store_true')
    parser.add_argument('--batch_size', type=int, default=128)

    args = parser.parse_args()
    np.random.seed(args.seed)

    if args.dataset == 'tgif-qa':
        args.annotation_file = '/mnt/285EDDF95EDDC02C/Users/Public/Documents/VideoDatasets/TGIF-QA/{}_{}_question.csv'
        args.output_pt = 'data/tgif-qa/{}/tgif-qa_{}_{}_questions.pt'
        args.vocab_json = 'data/tgif-qa/{}/tgif-qa_{}_vocab.json'
        # check if data folder exists
        if not os.path.exists('data/tgif-qa/{}'.format(args.question_type)):
            os.makedirs('data/tgif-qa/{}'.format(args.question_type))

        if args.question_type in ['frameqa', 'count']:
            tgif_qa.process_questions_openended(args)
        else:
            tgif_qa.process_questions_mulchoices(args)
    elif args.dataset == 'msrvtt-qa':
        args.annotation_file = '/mnt/285EDDF95EDDC02C/Users/Public/Documents/VideoDatasets/MSRVTT-QA/{}_qa.json'.format(args.mode)
        # check if data folder exists
        if not os.path.exists('data/{}'.format(args.dataset)):
            os.makedirs('data/{}'.format(args.dataset))
        msrvtt_qa.process_questions(args)
    elif args.dataset == 'msvd-qa':
        args.annotation_file = '/mnt/285EDDF95EDDC02C/Users/Public/Documents/VideoDatasets/MSVD-QA/{}_qa.json'.format(args.mode)
        # check if data folder exists
        if not os.path.exists('data/{}'.format(args.dataset)):
            os.makedirs('data/{}'.format(args.dataset))
        msvd_qa.process_questions(args)
    elif args.dataset == 'activitynet-qa':
        args.annotation_file = '/mnt/285EDDF95EDDC02C/Users/Public/Documents/VideoDatasets/ACTIVITYNET-QA/{}_{}.json'
        args.name2ids_pt = 'data/{}/name2ids.pt'
        activitynet_qa.create_name2ids(args)

        args.question_file = args.annotation_file.format(args.mode, 'q')
        args.answer_file = args.annotation_file.format(args.mode, 'a')
        if not os.path.exists('data/{}'.format(args.dataset)):
            os.makedirs('data/{}'.format(args.dataset))
        activitynet_qa.process_questions(args)
