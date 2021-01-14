import numpy as np
from torch.nn import functional as F

from transformers import AutoModel

from .utils import *
from .CRN import CRN, FasterCRN


class FeatureAggregation(nn.Module):
    def __init__(self, module_dim=512):
        super(FeatureAggregation, self).__init__()
        self.module_dim = module_dim

        self.q_proj = nn.Linear(module_dim, module_dim, bias=False)
        self.v_proj = nn.Linear(module_dim, module_dim, bias=False)

        self.cat = nn.Linear(2 * module_dim, module_dim)
        self.attn = nn.Linear(module_dim, 1)

        self.activation = nn.ELU()
        self.dropout = nn.Dropout(0.15)

    def forward(self, question_rep, visual_feat):
        visual_feat = self.dropout(visual_feat)
        q_proj = self.q_proj(question_rep)
        v_proj = self.v_proj(visual_feat)

        v_q_cat = torch.cat((v_proj, q_proj.unsqueeze(1) * v_proj), dim=-1)
        v_q_cat = self.cat(v_q_cat)
        v_q_cat = self.activation(v_q_cat)

        attn = self.attn(v_q_cat)  # (bz, k, 1)
        attn = F.softmax(attn, dim=1)  # (bz, k, 1)

        v_distill = (attn * visual_feat).sum(1)

        return v_distill


class InputUnitLinguistic(nn.Module):
    def __init__(self, vocab_size, wordvec_dim=300, rnn_dim=512, module_dim=512, bidirectional=True):
        super(InputUnitLinguistic, self).__init__()

        self.dim = module_dim

        self.bidirectional = bidirectional
        if bidirectional:
            rnn_dim = rnn_dim // 2

        self.encoder_embed = nn.Embedding(vocab_size, wordvec_dim)
        self.tanh = nn.Tanh()
        self.encoder = nn.LSTM(wordvec_dim, rnn_dim, batch_first=True, bidirectional=bidirectional)
        self.embedding_dropout = nn.Dropout(p=0.15)
        self.question_dropout = nn.Dropout(p=0.18)

        self.module_dim = module_dim

    def forward(self, questions, question_len):
        """
        Args:
            question: [Tensor] (batch_size, max_question_length)
            question_len: [Tensor] (batch_size)
        return:
            question representation [Tensor] (batch_size, module_dim)
        """
        questions_embedding = self.encoder_embed(questions)  # (batch_size, seq_len, dim_word)
        embed = self.tanh(self.embedding_dropout(questions_embedding))
        embed = nn.utils.rnn.pack_padded_sequence(embed, question_len, batch_first=True,
                                                  enforce_sorted=False)

        self.encoder.flatten_parameters()
        _, (question_embedding, _) = self.encoder(embed)
        if self.bidirectional:
            question_embedding = torch.cat([question_embedding[0], question_embedding[1]], -1)
        question_embedding = self.question_dropout(question_embedding)

        return question_embedding


class InputUnitLinguisticPrecomputed(nn.Module):
    def __init__(self, wordvec_dim=768, rnn_dim=512, module_dim=512, bidirectional=True):
        super(InputUnitLinguisticPrecomputed, self).__init__()

        self.dim = module_dim

        self.bidirectional = bidirectional
        if bidirectional:
            rnn_dim = rnn_dim // 2

        self.encoder_fc = nn.Linear(wordvec_dim, wordvec_dim)
        self.tanh = nn.Tanh()
        self.encoder = nn.LSTM(wordvec_dim, rnn_dim, batch_first=True, bidirectional=bidirectional)
        self.embedding_dropout = nn.Dropout(p=0.15)
        self.question_dropout = nn.Dropout(p=0.18)

        self.module_dim = module_dim

    def forward(self, questions_embedding, question_len):
        """
        Args:
            questions_embedding: [Tensor] (batch_size, max_question_length, wordvec_dim)
            question_len: [Tensor] (batch_size)
        return:
            question representation [Tensor] (batch_size, module_dim)
        """
        # questions_embedding = self.encoder_fc(questions_embedding)
        embed = self.tanh(self.embedding_dropout(questions_embedding))
        embed = nn.utils.rnn.pack_padded_sequence(embed, question_len, batch_first=True,
                                                  enforce_sorted=False)

        self.encoder.flatten_parameters()
        _, (question_embedding, _) = self.encoder(embed)
        if self.bidirectional:
            question_embedding = torch.cat([question_embedding[0], question_embedding[1]], -1)
        question_embedding = self.question_dropout(question_embedding)

        return question_embedding


class InputUnitLinguisticTransformers(nn.Module):
    def __init__(self, transformer_module, rnn_dim=512, module_dim=512, bidirectional=True):
        super(InputUnitLinguisticTransformers, self).__init__()

        if isinstance(transformer_module, str):
            self.transformer = AutoModel.from_pretrained(transformer_module)
        else:
            # Dict means fine tuned model
            model = transformer_module['model']
            path = transformer_module['path']
            self.transformer = AutoModels.from_pretrained(model)
            ckpt = torch.load(path)
            self.transformer = self.transformer.load_state_dict(ckpt['state_dict'])

        self.dim = module_dim

        self.bidirectional = bidirectional
        if bidirectional:
            rnn_dim = rnn_dim // 2

        self.tanh = nn.Tanh()
        self.encoder = nn.LSTM(768, rnn_dim, batch_first=True, bidirectional=bidirectional)
        self.embedding_dropout = nn.Dropout(p=0.15)
        self.question_dropout = nn.Dropout(p=0.18)

        self.module_dim = module_dim

    def forward(self, questions_tokens, question_len):
        """
        Args:
            questions_embedding: [Tensor] (batch_size, max_question_length, wordvec_dim)
            question_len: [Tensor] (batch_size)
        return:
            question representation [Tensor] (batch_size, module_dim)
        """
        questions_inputs_ids, questions_attention_mask = questions_tokens
        output_transformer = self.transformer(input_ids=questions_inputs_ids, attention_mask=questions_attention_mask)
        questions_embedding = output_transformer[0]
        embed = self.tanh(self.embedding_dropout(questions_embedding))
        embed = nn.utils.rnn.pack_padded_sequence(embed, question_len, batch_first=True,
                                                  enforce_sorted=False)

        self.encoder.flatten_parameters()
        _, (question_embedding, _) = self.encoder(embed)
        if self.bidirectional:
            question_embedding = torch.cat([question_embedding[0], question_embedding[1]], -1)
        question_embedding = self.question_dropout(question_embedding)

        return question_embedding


class InputUnitVisual(nn.Module):
    def __init__(self, k_max_frame_level, k_max_clip_level, spl_resolution, vision_dim, module_dim=512, subvids=0):
        super(InputUnitVisual, self).__init__()

        self.clip_level_motion_cond = CRN(module_dim, k_max_frame_level, k_max_frame_level, gating=False, spl_resolution=spl_resolution)
        self.clip_level_question_cond = CRN(module_dim, k_max_frame_level-2, k_max_frame_level-2, gating=True, spl_resolution=spl_resolution)
        self.video_level_motion_cond = CRN(module_dim, k_max_clip_level, k_max_clip_level, gating=False, spl_resolution=spl_resolution)
        self.video_level_question_cond = CRN(module_dim, k_max_clip_level-2, k_max_clip_level-2, gating=True, spl_resolution=spl_resolution)

        self.subvids = subvids
        if subvids > 0:
            self.subvid_level_motion_cond = CRN(module_dim, 6, 6, gating=False, spl_resolution=spl_resolution)
            self.subvid_level_question_cond = CRN(module_dim, 6-2, 6-2, gating=True, spl_resolution=spl_resolution)
            self.subvid_level_motion_proj = nn.Linear(module_dim, module_dim)

        self.sequence_encoder = nn.LSTM(vision_dim, module_dim, batch_first=True, bidirectional=False)
        self.clip_level_motion_proj = nn.Linear(vision_dim, module_dim)
        self.video_level_motion_proj = nn.Linear(module_dim, module_dim)
        self.appearance_feat_proj = nn.Linear(vision_dim, module_dim)

        self.question_embedding_proj = nn.Linear(module_dim, module_dim)

        self.module_dim = module_dim
        self.activation = nn.ELU()

    def forward(self, appearance_video_feat, motion_video_feat, question_embedding):
        """
        Args:
            appearance_video_feat: [Tensor] (batch_size, num_clips, num_frames, visual_inp_dim)
            motion_video_feat: [Tensor] (batch_size, num_clips, visual_inp_dim)
            question_embedding: [Tensor] (batch_size, module_dim)
        return:
            encoded video feature: [Tensor] (batch_size, N, module_dim)
        """
        batch_size = appearance_video_feat.size(0)
        clip_level_crn_outputs = []
        question_embedding_proj = self.question_embedding_proj(question_embedding)
        for i in range(appearance_video_feat.size(1)):
            clip_level_motion = motion_video_feat[:, i, :]  # (bz, 2048)
            clip_level_motion_proj = self.clip_level_motion_proj(clip_level_motion)

            clip_level_appearance = appearance_video_feat[:, i, :, :]  # (bz, 16, 2048)
            clip_level_appearance_proj = self.appearance_feat_proj(clip_level_appearance)  # (bz, 16, 512)
            # clip level CRNs
            clip_level_crn_motion = self.clip_level_motion_cond(torch.unbind(clip_level_appearance_proj, dim=1),
                                                                clip_level_motion_proj)
            clip_level_crn_question = self.clip_level_question_cond(clip_level_crn_motion, question_embedding_proj)

            clip_level_crn_output = torch.stack(clip_level_crn_question, dim=1)
            clip_level_crn_output = clip_level_crn_output.view(batch_size, -1, self.module_dim)
            clip_level_crn_outputs.append(clip_level_crn_output)

        if self.subvids > 0:
            # assert len(clip_level_crn_outputs) % self.subvids == 0
            nb_subvids = len(clip_level_crn_outputs) // self.subvids
            subvid_level_crn_outputs = []
            for i in range(nb_subvids):
                _, (subvid_level_motion, _) = self.sequence_encoder(motion_video_feat[:, i*self.subvids:(i+1)*self.subvids, :])
                subvid_level_motion = subvid_level_motion.transpose(0, 1)
                subvid_level_motion_feat_proj = self.subvid_level_motion_proj(subvid_level_motion)

                subvid_level_crn_motion = self.subvid_level_motion_cond(clip_level_crn_outputs[i*self.subvids:(i+1)*self.subvids],
                                                                        subvid_level_motion_feat_proj)
                subvid_level_crn_question = self.subvid_level_question_cond(subvid_level_crn_motion, question_embedding_proj)
                subvid_level_crn_output = torch.stack(subvid_level_crn_question, dim=1)
                subvid_level_crn_output = subvid_level_crn_output.view(batch_size, -1, self.module_dim)
                subvid_level_crn_outputs.append(subvid_level_crn_output)

            next_level_inputs = subvid_level_crn_outputs
        else:
            next_level_inputs = clip_level_crn_outputs

        # Encode video level motion
        _, (video_level_motion, _) = self.sequence_encoder(motion_video_feat)
        video_level_motion = video_level_motion.transpose(0, 1)
        video_level_motion_feat_proj = self.video_level_motion_proj(video_level_motion)
        # video level CRNs
        video_level_crn_motion = self.video_level_motion_cond(next_level_inputs, video_level_motion_feat_proj)
        video_level_crn_question = self.video_level_question_cond(video_level_crn_motion, question_embedding_proj.unsqueeze(1))

        video_level_crn_output = torch.stack(video_level_crn_question, dim=1)
        video_level_crn_output = video_level_crn_output.view(batch_size, -1, self.module_dim)

        return video_level_crn_output


class FastInputUnitVisual(nn.Module):
    def __init__(self, k_max_frame_level, k_max_clip_level, spl_resolution, vision_dim, module_dim=512, subvids=0):
        super(FastInputUnitVisual, self).__init__()

        self.clip_level_motion_cond = CRN(module_dim, k_max_frame_level, k_max_frame_level, gating=False, spl_resolution=spl_resolution)
        self.clip_level_question_cond = CRN(module_dim, k_max_frame_level-2, k_max_frame_level-2, gating=True, spl_resolution=spl_resolution)
        self.video_level_motion_cond = CRN(module_dim, k_max_clip_level, k_max_clip_level, gating=False, spl_resolution=spl_resolution)
        self.video_level_question_cond = CRN(module_dim, k_max_clip_level-2, k_max_clip_level-2, gating=True, spl_resolution=spl_resolution)

        self.subvids = subvids
        if subvids > 0:
            self.subvid_level_motion_cond = CRN(module_dim, subvids, subvids, gating=False, spl_resolution=spl_resolution)
            self.subvid_level_question_cond = CRN(module_dim, subvids-2, subvids-2, gating=True, spl_resolution=spl_resolution)
            self.subvid_level_motion_proj = nn.Linear(module_dim, module_dim)

        self.sequence_encoder = nn.LSTM(vision_dim, module_dim, batch_first=True, bidirectional=False)
        self.clip_level_motion_proj = nn.Linear(vision_dim, module_dim)
        self.video_level_motion_proj = nn.Linear(module_dim, module_dim)
        self.appearance_feat_proj = nn.Linear(vision_dim, module_dim)

        self.question_embedding_proj = nn.Linear(module_dim, module_dim)

        self.module_dim = module_dim
        self.vision_dim = vision_dim
        self.activation = nn.ELU()

    def forward(self, appearance_video_feat, motion_video_feat, question_embedding):
        """
        Args:
            appearance_video_feat: [Tensor] (batch_size, num_clips, num_frames, visual_inp_dim)
            motion_video_feat: [Tensor] (batch_size, num_clips, visual_inp_dim)
            question_embedding: [Tensor] (batch_size, module_dim)
        return:
            encoded video feature: [Tensor] (batch_size, N, module_dim)
        """
        batch_size = appearance_video_feat.size(0)

        question_embedding_proj = self.question_embedding_proj(question_embedding)
        motion_proj = self.clip_level_motion_proj(motion_video_feat)
        appearance_proj = torch.unbind(self.appearance_feat_proj(appearance_video_feat), dim=2)
        
        # clip level CRNs
        crn_motion = self.clip_level_motion_cond(appearance_proj, motion_proj)
        crn_question = self.clip_level_question_cond(crn_motion, question_embedding_proj)
        crn_output = torch.stack(crn_question, dim=2)

        if self.subvids > 0:
            # assert motion_video_feat.shape[1] % self.subvids == 0
            nb_subvids = motion_video_feat.shape[1] // self.subvids
            _, (subvid_level_motion, _) = self.sequence_encoder(motion_video_feat.view(batch_size * nb_subvids, self.subvids, self.vision_dim))
            subvid_level_motion = subvid_level_motion.transpose(0, 1).view(batch_size, nb_subvids, self.module_dim)
            subvid_level_motion_feat_proj = self.subvid_level_motion_proj(subvid_level_motion)

            clip_level_crn_outputs = torch.unbind(crn_output.view(batch_size, nb_subvids, self.subvids, crn_output.shape[2], self.module_dim), dim=2)

            subvid_level_crn_motion = self.subvid_level_motion_cond(clip_level_crn_outputs, subvid_level_motion_feat_proj)
            subvid_level_crn_question = self.subvid_level_question_cond(subvid_level_crn_motion, question_embedding_proj)
            subvid_level_crn_output = torch.stack(subvid_level_crn_question, dim=1)

            subvid_level_crn_output = subvid_level_crn_output.view(batch_size, nb_subvids, -1, self.module_dim)
            next_level_inputs = torch.unbind(subvid_level_crn_output, dim=1)
        else:
            next_level_inputs = torch.unbind(crn_output, dim=1)

        # Encode video level motion
        _, (video_level_motion, _) = self.sequence_encoder(motion_video_feat)
        video_level_motion = video_level_motion.transpose(0, 1)
        video_level_motion_feat_proj = self.video_level_motion_proj(video_level_motion)
        # video level CRNs
        video_level_crn_motion = self.video_level_motion_cond(next_level_inputs, video_level_motion_feat_proj)
        video_level_crn_question = self.video_level_question_cond(video_level_crn_motion, question_embedding_proj.unsqueeze(1))

        video_level_crn_output = torch.stack(video_level_crn_question, dim=1)
        video_level_crn_output = video_level_crn_output.view(batch_size, -1, self.module_dim)

        return video_level_crn_output


class FasterInputUnitVisual(nn.Module):
    def __init__(self, k_max_frame_level, k_max_clip_level, spl_resolution, vision_dim, module_dim=512, subvids=0):
        super(FasterInputUnitVisual, self).__init__()

        self.clip_level_motion_cond = FasterCRN(module_dim, k_max_frame_level, k_max_frame_level, gating=False, spl_resolution=spl_resolution)
        self.clip_level_question_cond = FasterCRN(module_dim, k_max_frame_level, k_max_frame_level, gating=True, spl_resolution=spl_resolution) #-2
        self.video_level_motion_cond = FasterCRN(module_dim, k_max_clip_level, k_max_clip_level, gating=False, spl_resolution=spl_resolution)
        self.video_level_question_cond = FasterCRN(module_dim, k_max_clip_level, k_max_clip_level, gating=True, spl_resolution=spl_resolution) #-2

        self.subvids = subvids
        if subvids > 0:
            self.subvid_level_motion_cond = FasterCRN(module_dim, subvids, subvids, gating=False, spl_resolution=spl_resolution)
            self.subvid_level_question_cond = FasterCRN(module_dim, subvids, subvids, gating=True, spl_resolution=spl_resolution)
            self.subvid_level_motion_proj = nn.Linear(module_dim, module_dim)

        self.sequence_encoder = nn.LSTM(vision_dim, module_dim, batch_first=True, bidirectional=False)
        self.clip_level_motion_proj = nn.Linear(vision_dim, module_dim)
        self.video_level_motion_proj = nn.Linear(module_dim, module_dim)
        self.appearance_feat_proj = nn.Linear(vision_dim, module_dim)

        self.question_embedding_proj = nn.Linear(module_dim, module_dim)

        self.module_dim = module_dim
        self.vision_dim = vision_dim
        self.activation = nn.ELU()

    def forward(self, appearance_video_feat, motion_video_feat, question_embedding):
        """
        Args:
            appearance_video_feat: [Tensor] (batch_size, num_clips, num_frames, visual_inp_dim)
            motion_video_feat: [Tensor] (batch_size, num_clips, visual_inp_dim)
            question_embedding: [Tensor] (batch_size, module_dim)
        return:
            encoded video feature: [Tensor] (batch_size, N, module_dim)
        """
        batch_size = appearance_video_feat.size(0)

        question_embedding_proj = self.question_embedding_proj(question_embedding)
        motion_proj = self.clip_level_motion_proj(motion_video_feat)
        appearance_proj = self.appearance_feat_proj(appearance_video_feat)

        # clip level CRNs
        crn_motion = self.clip_level_motion_cond(appearance_proj, motion_proj)
        crn_question = self.clip_level_question_cond(crn_motion, question_embedding_proj)
        clip_level_crn_outputs = crn_question.transpose(1, 2)

        if self.subvids > 0:
            # assert motion_video_feat.shape[1] % self.subvids == 0
            nb_subvids = motion_video_feat.shape[1] // self.subvids
            _, (subvid_level_motion, _) = self.sequence_encoder(motion_video_feat.view(batch_size * nb_subvids, self.subvids, self.vision_dim))
            subvid_level_motion = subvid_level_motion.squeeze(0).view(batch_size, nb_subvids, self.module_dim)
            subvid_level_motion_feat_proj = self.subvid_level_motion_proj(subvid_level_motion)
            subvid_level_motion_feat_proj = subvid_level_motion_feat_proj.unsqueeze(1).repeat(1, clip_level_crn_outputs.shape[1], 1, 1)

            clip_level_crn_outputs = clip_level_crn_outputs.view(batch_size, clip_level_crn_outputs.shape[1], nb_subvids, self.subvids, self.module_dim)

            subvid_level_crn_motion = self.subvid_level_motion_cond(clip_level_crn_outputs, subvid_level_motion_feat_proj)
            subvid_level_crn_question = self.subvid_level_question_cond(subvid_level_crn_motion, question_embedding_proj)

            subvid_level_crn_output = subvid_level_crn_question.view(batch_size, -1, nb_subvids, self.module_dim)
            next_level_inputs = subvid_level_crn_output
        else:
            next_level_inputs = clip_level_crn_outputs

        # Encode video level motion
        _, (video_level_motion, _) = self.sequence_encoder(motion_video_feat)
        video_level_motion = video_level_motion.transpose(0, 1)
        video_level_motion_feat_proj = self.video_level_motion_proj(video_level_motion)

        # video level CRNs
        video_level_crn_motion = self.video_level_motion_cond(clip_level_crn_outputs, video_level_motion_feat_proj)
        video_level_crn_question = self.video_level_question_cond(video_level_crn_motion, question_embedding_proj.unsqueeze(1))

        video_level_crn_output = video_level_crn_question.view(batch_size, -1, self.module_dim)

        return video_level_crn_output


class OutputUnitOpenEnded(nn.Module):
    def __init__(self, module_dim=512, num_answers=1000):
        super(OutputUnitOpenEnded, self).__init__()

        self.question_proj = nn.Linear(module_dim, module_dim)

        self.classifier = nn.Sequential(nn.Dropout(0.15),
                                        nn.Linear(module_dim * 2, module_dim),
                                        nn.ELU(),
                                        nn.BatchNorm1d(module_dim),
                                        nn.Dropout(0.15),
                                        nn.Linear(module_dim, num_answers))

    def forward(self, question_embedding, visual_embedding):
        question_embedding = self.question_proj(question_embedding)
        out = torch.cat([visual_embedding, question_embedding], 1)
        out = self.classifier(out)

        return out


class OutputUnitMultiChoices(nn.Module):
    def __init__(self, module_dim=512):
        super(OutputUnitMultiChoices, self).__init__()

        self.question_proj = nn.Linear(module_dim, module_dim)

        self.ans_candidates_proj = nn.Linear(module_dim, module_dim)

        self.classifier = nn.Sequential(nn.Dropout(0.15),
                                        nn.Linear(module_dim * 4, module_dim),
                                        nn.ELU(),
                                        nn.BatchNorm1d(module_dim),
                                        nn.Dropout(0.15),
                                        nn.Linear(module_dim, 1))

    def forward(self, question_embedding, q_visual_embedding, ans_candidates_embedding,
                a_visual_embedding):
        question_embedding = self.question_proj(question_embedding)
        ans_candidates_embedding = self.ans_candidates_proj(ans_candidates_embedding)
        out = torch.cat([q_visual_embedding, question_embedding, a_visual_embedding,
                         ans_candidates_embedding], 1)
        out = self.classifier(out)

        return out


class OutputUnitCount(nn.Module):
    def __init__(self, module_dim=512):
        super(OutputUnitCount, self).__init__()

        self.question_proj = nn.Linear(module_dim, module_dim)

        self.regression = nn.Sequential(nn.Dropout(0.15),
                                        nn.Linear(module_dim * 2, module_dim),
                                        nn.ELU(),
                                        nn.BatchNorm1d(module_dim),
                                        nn.Dropout(0.15),
                                        nn.Linear(module_dim, 1))

    def forward(self, question_embedding, visual_embedding):
        question_embedding = self.question_proj(question_embedding)
        out = torch.cat([visual_embedding, question_embedding], 1)
        out = self.regression(out)

        return out


class HCRNNetwork(nn.Module):
    def __init__(self, vision_dim, module_dim, word_dim, k_max_frame_level, k_max_clip_level, spl_resolution, vocab, question_type, hcrn_model="basic", bert_model="none", subvids=0):
        super(HCRNNetwork, self).__init__()

        self.question_type = question_type
        self.feature_aggregation = FeatureAggregation(module_dim)

        if self.question_type in ['action', 'transition']:
            encoder_vocab_size = len(vocab['question_answer_token_to_idx'])
            self.output_unit = OutputUnitMultiChoices(module_dim=module_dim)

        elif self.question_type == 'count':
            encoder_vocab_size = len(vocab['question_token_to_idx'])
            self.output_unit = OutputUnitCount(module_dim=module_dim)
        else:
            encoder_vocab_size = len(vocab['question_token_to_idx'])
            self.num_classes = len(vocab['answer_token_to_idx'])
            self.output_unit = OutputUnitOpenEnded(module_dim=module_dim, num_answers=self.num_classes)

        if bert_model == "none":
            self.linguistic_input_unit = InputUnitLinguistic(vocab_size=encoder_vocab_size, wordvec_dim=word_dim,
                                                             module_dim=module_dim, rnn_dim=module_dim)

        elif bert_model == "precomputed":
            self.linguistic_input_unit = InputUnitLinguisticPrecomputed(wordvec_dim=word_dim, module_dim=module_dim, rnn_dim=module_dim)
        else:
            self.linguistic_input_unit = InputUnitLinguisticTransformers(transformer_module=bert_model, module_dim=module_dim, rnn_dim=module_dim)

        if hcrn_model == "faster":
            self.visual_input_unit = FasterInputUnitVisual(k_max_frame_level=k_max_frame_level, k_max_clip_level=k_max_clip_level, spl_resolution=spl_resolution, vision_dim=vision_dim, module_dim=module_dim, subvids=subvids)
        elif hcrn_model == "fast":
            self.visual_input_unit = FastInputUnitVisual(k_max_frame_level=k_max_frame_level, k_max_clip_level=k_max_clip_level, spl_resolution=spl_resolution, vision_dim=vision_dim, module_dim=module_dim, subvids=subvids)
        else:
            # Normal/basic model
            self.visual_input_unit = InputUnitVisual(k_max_frame_level=k_max_frame_level, k_max_clip_level=k_max_clip_level, spl_resolution=spl_resolution, vision_dim=vision_dim, module_dim=module_dim, subvids=subvids)

        init_modules(self.modules(), w_init="xavier_uniform")
        if bert_model == "none":
            nn.init.uniform_(self.linguistic_input_unit.encoder_embed.weight, -1.0, 1.0)

    def forward(self, ans_candidates, ans_candidates_len, video_appearance_feat, video_motion_feat, question,
                question_len):
        """
        Args:
            ans_candidates: [Tensor] (batch_size, 5, max_ans_candidates_length)
            ans_candidates_len: [Tensor] (batch_size, 5)
            video_appearance_feat: [Tensor] (batch_size, num_clips, num_frames, visual_inp_dim)
            video_motion_feat: [Tensor] (batch_size, num_clips, visual_inp_dim)
            question: [Tensor] (batch_size, max_question_length)
            question_len: [Tensor] (batch_size)
        return:
            logits.
        """
        if torch.__version__ >= "1.7":
            # Pack padded from rnn doesn't automaticaly cast length on cpu after 1.7
            question_len = question_len.cpu()
        batch_size = question_len.size(0)
        if self.question_type in ['frameqa', 'count', 'none']:
            # get image, word, and sentence embeddings
            question_embedding = self.linguistic_input_unit(question, question_len)
            visual_embedding = self.visual_input_unit(video_appearance_feat, video_motion_feat, question_embedding)

            visual_embedding = self.feature_aggregation(question_embedding, visual_embedding)

            out = self.output_unit(question_embedding, visual_embedding)
        else:
            question_embedding = self.linguistic_input_unit(question, question_len)
            visual_embedding = self.visual_input_unit(video_appearance_feat, video_motion_feat, question_embedding)

            q_visual_embedding = self.feature_aggregation(question_embedding, visual_embedding)

            # ans_candidates: (batch_size, num_choices, max_len)
            ans_candidates_agg = ans_candidates.view(-1, ans_candidates.size(2))
            ans_candidates_len_agg = ans_candidates_len.view(-1)

            batch_agg = np.reshape(
                np.tile(np.expand_dims(np.arange(batch_size), axis=1), [1, 5]), [-1])

            ans_candidates_embedding = self.linguistic_input_unit(ans_candidates_agg, ans_candidates_len_agg)

            a_visual_embedding = self.feature_aggregation(ans_candidates_embedding, visual_embedding[batch_agg])
            out = self.output_unit(question_embedding[batch_agg], q_visual_embedding[batch_agg],
                                   ans_candidates_embedding,
                                   a_visual_embedding)
        return out
