from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import logging
import math

import torch

from .module_encoder import TfModel, TextConfig, VisualConfig, AudioConfig
from .until_module import PreTrainedModel, LayerNorm
from .until_module import getBinaryTensor, CTCModule, MLLinear, MLAttention, TL_SelfAttention, \
    TL_CrossAttention, Emotion_CrossAttention, KG2E
from .MatchingAttention import MatchingAttention
import warnings
from .losses import *
import numpy as np
from torch_geometric.nn import GraphConv, GATConv
import torch.nn.functional as F
from .transformers_encoder.transformer import TransformerEncoder
import random


warnings.filterwarnings("ignore")
logger = logging.getLogger(__name__)


class PreTrainedModel(PreTrainedModel, nn.Module):
    def __init__(self, text_config, visual_config, audio_config, *inputs, **kwargs):
        # utilize bert config as base config
        super(PreTrainedModel, self).__init__(visual_config)
        self.text_config = text_config
        self.visual_config = visual_config
        self.audio_config = audio_config
        self.visual = None
        self.audio = None
        self.text = None

    @classmethod
    def from_pretrained(cls, text_model_name, visual_model_name, audio_model_name,
                        state_dict=None, cache_dir=None, type_vocab_size=2, *inputs, **kwargs):
        task_config = None
        if "task_config" in kwargs.keys():
            task_config = kwargs["task_config"]
            if not hasattr(task_config, "local_rank"):
                task_config.__dict__["local_rank"] = 0
            elif task_config.local_rank == -1:
                task_config.local_rank = 0
        text_config, _ = TextConfig.get_config(text_model_name, cache_dir, type_vocab_size, state_dict=None,
                                               task_config=task_config)
        visual_config, _ = VisualConfig.get_config(visual_model_name, cache_dir, type_vocab_size, state_dict=None,
                                                   task_config=task_config)
        audio_config, _ = AudioConfig.get_config(audio_model_name, cache_dir, type_vocab_size, state_dict=None,
                                                 task_config=task_config)
        model = cls(text_config, visual_config, audio_config, *inputs, **kwargs)
        if state_dict is not None:
            model = cls.init_preweight(model, state_dict, task_config=task_config)
        return model


class Normalize(nn.Module):
    def __init__(self, dim):
        super(Normalize, self).__init__()
        self.norm2d = LayerNorm(dim)

    def forward(self, inputs):
        inputs = torch.as_tensor(inputs).float()
        inputs = inputs.view(-1, inputs.shape[-2], inputs.shape[-1])
        output = self.norm2d(inputs)
        return output

def improved_cos_sim(labels):
    sim = F.cosine_similarity(labels.unsqueeze(0),labels.unsqueeze(1),dim=-1)
    norms = labels.norm(dim=-1)
    zero_indices = torch.nonzero(norms == 0, as_tuple=False).flatten()
    if zero_indices.numel() > 0:
        for i in zero_indices:
            for j in zero_indices:
                sim[i, j] = 1.0
    return sim

def squared_frobenius_norm(A, B):
    A = F.normalize(A,p=2,dim=-1)
    B = F.normalize(B,p=2,dim=-1)
    out = torch.sum(A*B,dim=-1)
    return torch.sum(out ** 2)



def show_log(task_config, info):
    if task_config is None or task_config.local_rank == 0:
        logger.warning(info)


def kaiming_init(m):
    if isinstance(m, nn.Linear):
        nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
        if m.bias is not None:
            m.bias.data.fill_(0.0)


def xavier_init(m):
    if isinstance(m, nn.Linear):
        nn.init.xavier_uniform_(m.weight)
        if m.bias is not None:
            m.bias.data.fill_(0.0)


def update_attr(target_name, target_config, target_attr_name, source_config, source_attr_name, default_value=None):
    if hasattr(source_config, source_attr_name):
        if default_value is None or getattr(source_config, source_attr_name) != default_value:
            setattr(target_config, target_attr_name, getattr(source_config, source_attr_name))
            show_log(source_config, "Set {}.{}: {}.".format(target_name,
                                                            target_attr_name, getattr(target_config, target_attr_name)))
    return target_config


class HGE(PreTrainedModel):
    def __init__(self, text_config, visual_config, audio_config, task_config):
        super(HGE, self).__init__(text_config, visual_config, audio_config)
        self.task_config = task_config
        if self.task_config.m3ed:
            self.num_classes = 7
        else:
            self.num_classes = 6

        self.aligned = task_config.aligned
        self.proto_m = task_config.proto_m
        self.task_config = task_config

        text_config = update_attr("text_config", text_config, "num_hidden_layers",
                                  self.task_config, "text_num_hidden_layers")
        self.text = TfModel(text_config)
        visual_config = update_attr("visual_config", visual_config, "num_hidden_layers",
                                    self.task_config, "visual_num_hidden_layers")
        self.visual = TfModel(visual_config)
        audio_config = update_attr("audio_config", audio_config, "num_hidden_layers",
                                   self.task_config, "audio_num_hidden_layers")
        self.audio = TfModel(audio_config)

        self.text_norm = Normalize(task_config.text_dim)
        self.visual_norm = Normalize(task_config.video_dim)
        self.audio_norm = Normalize(task_config.audio_dim)
        self.mse_loss = nn.MSELoss()

        self.apply(self.init_weights)

        self.text_attention = MLAttention(self.num_classes, task_config.hidden_size)
        self.visual_attention = MLAttention(self.num_classes, task_config.hidden_size)
        self.audio_attention = MLAttention(self.num_classes, task_config.hidden_size)

        self.d_l = self.d_a = self.d_v = self.task_config.hidden_size
        self.attn_dropout = 0.0
        self.attn_dropout_v = 0.1
        self.attn_dropout_a = 0.2
        self.margin = task_config.margin
        self.num_heads = 4
        self.layers = 3
        self.relu_dropout = 0.0
        self.res_dropout = 0.0
        self.embed_dropout = 0.2
        self.attn_mask = None
        self.threshold = task_config.neg_threshold
        self.pos_threshold = task_config.pos_threshold

        self.emotion_quires = nn.Embedding(self.num_classes, task_config.hidden_size)
        self.lstm = nn.LSTM(3 * task_config.hidden_size, task_config.hidden_size // 2, bidirectional=True, num_layers=2,
                            batch_first=True)

        self.tl_sa = TL_SelfAttention(task_config.hidden_size)
        self.common_decoder = nn.Linear(task_config.hidden_size * 3, task_config.hidden_size)

        self.emotion_ca_1 = Emotion_CrossAttention(task_config.hidden_size)
        self.emotion_ca_2 = Emotion_CrossAttention(task_config.hidden_size)

        self.tl_ca = TL_CrossAttention(task_config.hidden_size)
        self.time_clf = MLLinear([task_config.hidden_size * self.num_classes], self.num_classes)

        self.common_clf = MLLinear([task_config.hidden_size * self.num_classes], self.num_classes)
        self.bce_loss = nn.BCEWithLogitsLoss()


        self.align_c_l = MLLinear([task_config.hidden_size * self.num_classes], task_config.hidden_size)
        self.align_c_v = MLLinear([task_config.hidden_size * self.num_classes], task_config.hidden_size)
        self.align_c_a = MLLinear([task_config.hidden_size * self.num_classes], task_config.hidden_size)

        self.align_s_l = MLLinear([task_config.hidden_size * self.num_classes], task_config.hidden_size)
        self.align_s_v = MLLinear([task_config.hidden_size * self.num_classes], task_config.hidden_size)
        self.align_s_a = MLLinear([task_config.hidden_size * self.num_classes], task_config.hidden_size)

        self.common_encoder = MLLinear([task_config.hidden_size, task_config.hidden_size // 2], task_config.hidden_size)
        self.s_t_encoder = MLLinear([task_config.hidden_size, task_config.hidden_size // 2], task_config.hidden_size)
        self.s_v_encoder = MLLinear([task_config.hidden_size, task_config.hidden_size // 2], task_config.hidden_size)
        self.s_a_encoder = MLLinear([task_config.hidden_size, task_config.hidden_size // 2], task_config.hidden_size)

        self.decoder_t = MLLinear([self.task_config.hidden_size * 2], self.task_config.hidden_size)
        self.decoder_v = MLLinear([self.task_config.hidden_size * 2], self.task_config.hidden_size)
        self.decoder_a = MLLinear([self.task_config.hidden_size * 2], self.task_config.hidden_size)

        self.recon_loss = nn.MSELoss()
        self.sim_loss = nn.MSELoss()

        self.weight_t = MLLinear(
            [self.num_classes * self.task_config.hidden_size * 2],
            self.num_classes * self.task_config.hidden_size)
        self.weight_a = MLLinear(
            [self.num_classes * self.task_config.hidden_size * 2],
            self.num_classes * self.task_config.hidden_size)
        self.weight_v = MLLinear(
            [self.num_classes * self.task_config.hidden_size * 2],
            self.num_classes * self.task_config.hidden_size)
        self.weight_c = MLLinear(
            [self.num_classes * self.task_config.hidden_size * 3],
            self.num_classes * self.task_config.hidden_size)

        self.th_weight = MLLinear([3 * self.task_config.hidden_size, task_config.hidden_size],
                                  self.task_config.hidden_size)
        self.vh_weight = MLLinear([3 * self.task_config.hidden_size, task_config.hidden_size],
                                  self.task_config.hidden_size)
        self.ah_weight = MLLinear([3 * self.task_config.hidden_size, task_config.hidden_size],
                                  self.task_config.hidden_size)

        self.window_future = task_config.window_future
        self.window_past = task_config.window_past

        self.conv_t = GraphConv(task_config.hidden_size, task_config.hidden_size)
        self.conv_a = GraphConv(task_config.hidden_size, task_config.hidden_size)
        self.conv_v = GraphConv(task_config.hidden_size, task_config.hidden_size)
        self.graph_linear = nn.Linear(task_config.hidden_size, task_config.hidden_size)
        self.time_conv = GraphConv(task_config.hidden_size, task_config.hidden_size)

        self.matchatt = MatchingAttention(2 * task_config.hidden_size, 2 * task_config.hidden_size)
        self.time_linear = nn.Linear(task_config.hidden_size, task_config.hidden_size)

        self.trans_l_with_a = self.get_network(self_type='la')
        self.trans_l_with_v = self.get_network(self_type='lv')
        self.trans_a_with_l = self.get_network(self_type='al')
        self.trans_a_with_v = self.get_network(self_type='av')
        self.trans_v_with_l = self.get_network(self_type='vl')
        self.trans_v_with_a = self.get_network(self_type='va')
        self.trans_l_with_cl = self.get_network(self_type='lv')
        self.trans_v_with_cv = self.get_network(self_type='vl')
        self.trans_a_with_ca = self.get_network(self_type='va')
        self.trans_l_mem = self.get_network(self_type='l_mem', layers=3)
        self.trans_a_mem = self.get_network(self_type='a_mem', layers=3)
        self.trans_v_mem = self.get_network(self_type='v_mem', layers=3)
        self.self_attention = self.get_network(self_type='l')
        self.self_attention_l = self.get_network(self_type='l')
        self.self_attention_a = self.get_network(self_type='a')
        self.self_attention_v = self.get_network(self_type='v')

        self.transR = KG2E(task_config.hidden_size, task_config.hidden_size, task_config.hidden_size,
                           vmin=task_config.vmin, vmax=task_config.vmax)
        self.transR_common = KG2E(task_config.hidden_size, self.task_config.hidden_size,
                                  task_config.hidden_size, vmin=task_config.vmin, vmax=task_config.vmax)
        self.transR_common_2 = KG2E(task_config.hidden_size, self.task_config.hidden_size,
                                    task_config.hidden_size, vmin=task_config.vmin, vmax=task_config.vmax)

        self.transR_emotion = KG2E(task_config.hidden_size, self.task_config.hidden_size,
                                    task_config.hidden_size, vmin=task_config.vmin, vmax=task_config.vmax)

        self.clf = MLLinear([task_config.hidden_size * self.num_classes], self.num_classes)

        self.sigmoid = nn.Sigmoid()

        if not self.aligned:
            self.a2t_ctc = CTCModule(task_config.audio_dim, 50 if task_config.unaligned_mask_same_length else 500)
            self.v2t_ctc = CTCModule(task_config.video_dim, 50 if task_config.unaligned_mask_same_length else 500)

    def get_text_visual_audio_output(self, text, text_mask, visual, visual_mask, audio, audio_mask):
        text_layers, text_pooled_output = self.text(text, text_mask, output_all_encoded_layers=True)
        text_output = text_layers[-1]
        visual_layers, visual_pooled_output = self.visual(visual, visual_mask, output_all_encoded_layers=True)
        visual_output = visual_layers[-1]
        audio_layers, audio_pooled_output = self.audio(audio, audio_mask, output_all_encoded_layers=True)
        audio_output = audio_layers[-1]
        return text_output, visual_output, audio_output


    def get_network(self, self_type='l', layers=-1):
        if self_type in ['l', 'al', 'vl']:
            embed_dim, attn_dropout = self.d_l, self.attn_dropout
        elif self_type in ['a', 'la', 'va']:
            embed_dim, attn_dropout = self.d_a, self.attn_dropout_a
        elif self_type in ['v', 'lv', 'av']:
            embed_dim, attn_dropout = self.d_v, self.attn_dropout_v
        elif self_type == 'l_mem':
            embed_dim, attn_dropout = 2 * self.d_l, self.attn_dropout
        elif self_type == 'a_mem':
            embed_dim, attn_dropout = 2 * self.d_a, self.attn_dropout
        elif self_type == 'v_mem':
            embed_dim, attn_dropout = 2 * self.d_v, self.attn_dropout
        else:
            raise ValueError("Unknown network type")

        return TransformerEncoder(embed_dim=embed_dim,
                                  num_heads=self.num_heads,
                                  layers=max(self.layers, layers),
                                  attn_dropout=attn_dropout,
                                  relu_dropout=self.relu_dropout,
                                  res_dropout=self.res_dropout,
                                  embed_dropout=self.embed_dropout,
                                  attn_mask=self.attn_mask)

    def edge_perms(self, l, window_past, window_future):
        all_perms = set()
        array = np.arange(l)
        for j in range(l):  # j: start index
            perms = set()

            if window_past == -1 and window_future == -1:
                eff_array = array
            elif window_past == -1:
                eff_array = array[:min(l, j + window_future + 1)]
            elif window_future == -1:
                eff_array = array[max(0, j - window_past):]
            else:
                eff_array = array[max(0, j - window_past):min(l, j + window_future + 1)]

            for item in eff_array:
                perms.add((j, item))
            all_perms = all_perms.union(perms)
        return list(all_perms)

    def cal_sim(self, features, edges):
        features = F.normalize(features,dim=-1,p=2)
        sim = torch.einsum('bld,nld->bn', features, features)
        weights = torch.zeros(len(edges)).to(features.device)
        edge_index = []
        for index, edge in enumerate(edges):
            weights[index] += sim[edge[0], edge[1]]
            edge_index.append([edge[0], edge[1]])
        return weights, torch.tensor(edge_index)

    def cal_time_sim(self, features, edges):
        features = F.normalize(features,dim=-1,p=2)
        sim = torch.einsum('btld,ntld->bn', features, features)
        weights = torch.zeros(len(edges)).to(features.device)
        edge_index = []
        for index, edge in enumerate(edges):
            weights[index] += sim[edge[0], edge[1]]
            edge_index.append([edge[0], edge[1]])
        return weights, torch.tensor(edge_index)

    def cal_label_sim(self, labels, edges):
        if type(labels) != torch.float:
            labels = labels.to(torch.float)
        similarity_matrix = improved_cos_sim(labels)
        weights = torch.zeros((labels.size(0), labels.size(0))).to(labels.device)
        for index, edge in enumerate(edges):
            weights[edge[0], edge[1]] += similarity_matrix[edge[0], edge[1]]
        return weights

    def cal_feature_sim(self, features, edges):
        features = F.normalize(features,dim=-1,p=2)
        sim = torch.einsum('bld,nld->bn', features, features)
        weights = torch.zeros((sim.size(0), sim.size(0))).to(features.device)
        for index, edge in enumerate(edges):
            weights[edge[0], edge[1]] += sim[edge[0], edge[1]]
        return weights

    def cal_time_feature_sim(self, features, edges):
        features = F.normalize(features,dim=-1,p=2)
        sim = torch.einsum('btld,ntld->bn', features, features)
        weights = torch.zeros((sim.size(0), sim.size(0))).to(features.device)
        for index, edge in enumerate(edges):
            weights[edge[0], edge[1]] += sim[edge[0], edge[1]]
        return weights

    def generate_neg_triple(self, textual, visual, audio, labels):
        if type(labels) != torch.float:
            labels = labels.to(torch.float)

        similarity_matrix = improved_cos_sim(labels)

        neg_text = textual.clone()
        neg_visual = visual.clone()
        neg_audio = audio.clone()

        feature_index = [0, 1, 2]

        for i in range(textual.size(0)):
            diff_index = (similarity_matrix[i] < self.threshold).nonzero(as_tuple=True)
            replace_num = random.randint(1, 3)
            index = random.choice(diff_index[0])
            index_2 = random.choice(diff_index[0])

            if replace_num == 1:
                replace_index = random.sample(feature_index, replace_num)
                if replace_index[0] == 0:
                    neg_text[i, :] = textual[index, :]
                elif replace_index[0] == 1:
                    neg_visual[i, :] = visual[index, :]
                else:
                    neg_audio[i, :] = audio[index, :]
            else:
                preserve_index = random.sample(feature_index, 1)
                if preserve_index[0] == 0:
                    neg_visual[i, :] = visual[index, :]
                    neg_audio[i, :] = audio[index_2, :]
                elif preserve_index[0] == 1:
                    neg_text[i, :] = textual[index_2, :]
                    neg_audio[i, :] = audio[index, :]
                else:
                    neg_text[i, :] = textual[index, :]
                    neg_visual[i, :] = visual[index_2, :]

        return neg_text, neg_visual, neg_audio

    def generate_neg_cfeature(self, features, labels):
        if type(labels) != torch.float:
            labels = labels.to(torch.float)
        similarity_matrix = improved_cos_sim(labels)
        neg_features = features.clone()
        for i in range(features.size(0)):
            diff_index = (similarity_matrix[i] < self.threshold).nonzero(as_tuple=True)
            index = random.choice(diff_index[0])
            neg_features[i] = features[index]
        return neg_features

    def generate_pos_feature(self, features, labels):
        if type(labels) != torch.float:
            labels = labels.to(torch.float)
        similarity_matrix = improved_cos_sim(labels)
        pos_features = features.clone()
        similar = torch.zeros(features.size(0)).to(features.device)
        for i in range(features.size(0)):
            condition = (similarity_matrix[i] >= self.pos_threshold) & (similarity_matrix[i] <= 1.0)
            diff_index = condition.nonzero(as_tuple=True)
            all_zeros = diff_index[0].numel() == 0
            if all_zeros:
                index = i
            else:
                index = random.choice(diff_index[0])
            pos_features[i] = features[index]
            similar[i] = similarity_matrix[i,index]
        return pos_features,similar

    def forward(self, text, text_mask, visual, visual_mask, audio, audio_mask,
                label_input, label_mask, groundTruth_labels=None, training=True):
        text = self.text_norm(text)
        visual = self.visual_norm(visual)
        audio = self.audio_norm(audio)
        if self.aligned == False and self.task_config.m3ed == False:
            visual, v2t_position = self.v2t_ctc(visual)
            audio, a2t_position = self.a2t_ctc(audio)
        text_output, visual_output, audio_output = self.get_text_visual_audio_output(text, text_mask, visual,
                                                                                     visual_mask, audio,
                                                                                     audio_mask)  # [B, L, D]
        text_lsr, text_attention = self.text_attention(text_output, (1 - text_mask).type(torch.bool))
        visual_lsr, visual_attention = self.visual_attention(visual_output, (1 - visual_mask).type(torch.bool))
        audio_lsr, audio_attention = self.audio_attention(audio_output, (1 - audio_mask).type(
            torch.bool))
        bsz = text_lsr.size(0)
        common_text = self.common_encoder(text_lsr)
        common_visual = self.common_encoder(visual_lsr)
        common_audio = self.common_encoder(audio_lsr)

        edges = self.edge_perms(bsz, self.window_past, self.window_future)
        t_edge_weight, t_edge_index = self.cal_sim(common_text, edges)
        v_edge_weight, v_edge_index = self.cal_sim(common_visual, edges)
        a_edge_weight, a_edge_index = self.cal_sim(common_audio, edges)
        t_edge_index = t_edge_index.transpose(0, 1).to(common_text.device)
        v_edge_index = v_edge_index.transpose(0, 1).to(common_text.device)
        a_edge_index = a_edge_index.transpose(0, 1).to(common_text.device)
        common_t = common_text.view(self.num_classes, bsz, -1)
        common_v = common_visual.view(self.num_classes, bsz, -1)
        common_a = common_audio.view(self.num_classes, bsz, -1)

        common_t = F.relu(self.graph_linear(self.conv_t(common_t, t_edge_index, t_edge_weight)))
        common_v = F.relu(self.graph_linear(self.conv_v(common_v, v_edge_index, v_edge_weight)))
        common_a = F.relu(self.graph_linear(self.conv_a(common_a, a_edge_index, a_edge_weight)))

        t_similarity = self.cal_feature_sim(common_t.permute(1, 0, 2), t_edge_index)
        v_similarity = self.cal_feature_sim(common_v.permute(1, 0, 2), v_edge_index)
        a_similarity = self.cal_feature_sim(common_a.permute(1, 0, 2), a_edge_index)
        label_similarity = self.cal_label_sim(groundTruth_labels, edges)

        loss_sim = self.sim_loss(t_similarity, label_similarity) + self.sim_loss(v_similarity,
                                                                                 label_similarity) + self.sim_loss(
            a_similarity, label_similarity)
        loss_sim = loss_sim / 3

        common_t = self.self_attention_l(common_t)
        common_v = self.self_attention_v(common_v)
        common_a = self.self_attention_a(common_a)

        input_ids = torch.tensor(np.arange(self.num_classes), device=text.device).long()
        input_ids = self.emotion_quires(input_ids)
        input_ids = input_ids.unsqueeze(1).repeat(1, text.size(0), 1)
        time_feature = torch.cat([text_output, visual_output, audio_output], dim=-1)
        if self.task_config.m3ed:
            combined_mask = text_mask & visual_mask & audio_mask
            lengths = combined_mask.sum(dim=1)
            time_feature = torch.nn.utils.rnn.pack_padded_sequence(time_feature, lengths.cpu(), batch_first=True,
                                                                   enforce_sorted=False)
            time_feature, _ = self.lstm(time_feature)
            input_ids = input_ids.permute(1, 0, 2)
            time_feature, _ = torch.nn.utils.rnn.pad_packed_sequence(time_feature, batch_first=True)
            time_feature = torch.einsum('bik,bjk->bijk', time_feature, input_ids)
        else:
            time_feature, _ = self.lstm(time_feature)
            input_ids = input_ids.permute(1, 0, 2)
            time_feature = torch.einsum('bik,bjk->bijk', time_feature, input_ids)

        # 这里加一个时间的Graph
        time_edge_weight, time_edge_index = self.cal_time_sim(time_feature, edges)
        time_edge_index = time_edge_index.transpose(0, 1).to(time_feature.device)

        time_feature = F.relu(self.time_linear(self.time_conv(time_feature.permute(1, 2, 0, 3), time_edge_index, time_edge_weight)))

        time_feature = time_feature.permute(2, 0, 1, 3)
        time_similarity = self.cal_time_feature_sim(time_feature, time_edge_index)
        loss_sim += self.sim_loss(time_similarity, label_similarity)
        time_feature = self.tl_sa(time_feature,mask=None)

        emotion_tv = self.emotion_ca_1(common_t, common_v,
                                       common_v)
        emotion_feature = self.emotion_ca_2(emotion_tv, common_a, common_a)
        emotion_feature = emotion_feature.permute(1, 0, 2)
        time_feature = self.tl_ca(time_feature, emotion_feature, emotion_feature,mask=None)

        s_t = self.s_t_encoder(text_lsr)
        s_v = self.s_v_encoder(visual_lsr)
        s_a = self.s_a_encoder(audio_lsr)
        recon_t = self.decoder_t(torch.cat([s_t, common_text], dim=-1))
        recon_v = self.decoder_v(torch.cat([s_v, common_visual], dim=-1))
        recon_a = self.decoder_a(torch.cat([s_a, common_audio], dim=-1))
        recon_loss_t = self.recon_loss(recon_t, text_lsr)
        recon_loss_v = self.recon_loss(recon_v, visual_lsr)
        recon_loss_a = self.recon_loss(recon_a, audio_lsr)

        c_sim_t = self.align_c_l(common_t.permute(1, 0, 2).contiguous().view(s_t.size(0), -1))
        c_sim_v = self.align_c_v(common_v.permute(1, 0, 2).contiguous().view(s_t.size(0), -1))
        c_sim_a = self.align_c_a(common_a.permute(1, 0, 2).contiguous().view(s_t.size(0), -1))

        s_sim_t = self.align_s_l(s_t.view(s_t.size(0), -1))
        s_sim_v = self.align_s_v(s_v.view(s_t.size(0), -1))
        s_sim_a = self.align_s_a(s_a.view(s_t.size(0), -1))

        diff_l = squared_frobenius_norm(c_sim_t, s_sim_t)
        diff_v = squared_frobenius_norm(c_sim_v, s_sim_v)
        diff_a = squared_frobenius_norm(c_sim_a, s_sim_a)
        loss_ort = diff_l + diff_v + diff_a

        neg_t, neg_v, neg_a = self.generate_neg_triple(s_t, s_v, s_a, groundTruth_labels)

        # s-s-s
        t_h1, v_t1, a_r1, score_tav = self.transR(s_t, s_v, s_a, neg_t, neg_v, s_a,
                                                  mode='tav')  # (bsz,hidden * label)
        t_h2, a_t1, v_r1, score_tva = self.transR(s_t, s_v, s_a, neg_t, s_v, neg_a, mode='tva')
        v_h1, t_t1, a_r2, score_vat = self.transR(s_t, s_v, s_a, neg_t, neg_v, s_a, mode='vat')
        v_h2, a_t2, t_r1, score_vta = self.transR(s_t, s_v, s_a, s_t, neg_v, neg_a, mode='vta')
        a_h1, v_t2, t_r2, score_avt = self.transR(s_t, s_v, s_a, s_t, neg_v, neg_a, mode='atv')
        a_h2, t_t2, v_r2, score_atv = self.transR(s_t, s_v, s_a, neg_t, s_v, neg_a, mode='avt')

        # 实验
        common_t = common_t.permute(1, 0, 2)
        common_a = common_a.permute(1, 0, 2)
        common_v = common_v.permute(1, 0, 2)

        # s-c-s
        ct_h1, cv_t1, ca_r1, score_tcv = self.transR_common(s_t, s_v, common_a, neg_t, neg_v, common_a,
                                                            mode='tav')
        ct_h2, ca_t1, cv_r1, score_tca = self.transR_common(s_t, common_v, s_a, neg_t, common_v, neg_a,
                                                            mode='tva')
        cv_h1, ca_t2, ct_r1, score_vca = self.transR_common(common_t, s_v, s_a, common_t, neg_v, neg_a,
                                                            mode='vta')
        cv_h2, ct_t1, ca_r2, score_vct = self.transR_common(s_t, s_v, common_a, neg_t, neg_v, common_a,
                                                            mode='vat')
        ca_h1, cv_t2, ct_r2, score_acv = self.transR_common(common_t, s_v, s_a, common_t, neg_v, neg_a,
                                                            mode='atv')
        ca_h2, ct_t2, cv_r2, score_act = self.transR_common(s_t, common_v, s_a, neg_t, common_v, neg_a,
                                                            mode='avt')
        # s-e-s
        cte_h1, cve_t1, cae_r1, score_tcv_1 = self.transR_emotion(s_t, s_v, emotion_feature, neg_t, neg_v,
                                                                 emotion_feature,
                                                                 mode='tav')
        cte_h2, cae_t1, cve_r1, score_tca_1 = self.transR_emotion(s_t, emotion_feature, s_a, neg_t, emotion_feature,
                                                                 neg_a,
                                                                 mode='tva')
        cve_h1, cae_t2, cte_r1, score_vca_1 = self.transR_emotion(emotion_feature, s_v, s_a, emotion_feature, neg_v,
                                                                 neg_a,
                                                                 mode='vta')
        cve_h2, cte_t1, cae_r2, score_vct_1 = self.transR_emotion(s_t, s_v, emotion_feature, neg_t, neg_v,
                                                                 emotion_feature,
                                                                 mode='vat')
        cae_h1, cve_t2, cte_r2, score_acv_1 = self.transR_emotion(emotion_feature, s_v, s_a, emotion_feature, neg_v,
                                                                 neg_a,
                                                                 mode='atv')
        cae_h2, cte_t2, cve_r2, score_act_1 = self.transR_emotion(s_t, emotion_feature, s_a, neg_t, emotion_feature,
                                                                 neg_a,
                                                                 mode='avt')

        score_emotion = torch.mean(score_tcv_1) + torch.mean(score_tca_1) + torch.mean(score_vca_1) + torch.mean(
            score_vct_1) + torch.mean(score_acv_1) + torch.mean(score_act_1)
        score_emotion = score_emotion / 6

        pos_t,t_similar = self.generate_pos_feature(s_t, groundTruth_labels)
        pos_v,v_similar = self.generate_pos_feature(s_v, groundTruth_labels)
        pos_a,a_similar = self.generate_pos_feature(s_a, groundTruth_labels)

        # st-ct-st
        st_h, st_t, ct_r, score_tct = self.transR_common(s_t, pos_t, common_t, s_t, neg_t, common_t,pos_similar=t_similar,
                                                         mode='t')
        sv_h, sv_t, cv_r, score_vcv = self.transR_common(s_v, pos_v, common_v, s_v, neg_v, common_v,pos_similar=v_similar,
                                                         mode='v')
        sa_h, sa_t, ca_r, score_aca = self.transR_common(s_a, pos_a, common_a, s_a, neg_a, common_v,pos_similar=a_similar,
                                                         mode='a')

        sim_score_loss = torch.mean(score_tct) + torch.mean(score_vcv) + torch.mean(score_aca)
        sim_score_loss = sim_score_loss / 3

        #st_e_st
        et_h, et_t, et_r, score_tet = self.transR_emotion(s_t,pos_t,emotion_feature,s_t,neg_t,emotion_feature,pos_similar=t_similar,mode='t')
        ev_h, ev_t, ev_r, score_vev = self.transR_emotion(s_v,pos_v,emotion_feature,s_v,neg_v,emotion_feature,pos_similar=v_similar,mode='v')
        ea_h, ea_t, ea_r, score_aea = self.transR_emotion(s_a,pos_a,emotion_feature,s_a,neg_a,emotion_feature,pos_similar=a_similar,mode='a')

        emotion_score_loss = torch.mean(score_tet) + torch.mean(score_vev) + torch.mean(score_aea)
        emotion_score_loss = emotion_score_loss / 3

        neg_comm_t, neg_comm_v, neg_comm_a = self.generate_neg_triple(common_t, common_v, common_a, groundTruth_labels)

        # c-c-c
        comm_t_h1, comm_v_t1, comm_a_r1, score_comm_tav = self.transR_common_2(common_t, common_v, common_a,
                                                                               neg_comm_t, neg_comm_v, common_a,
                                                                               mode='tav')
        comm_t_h2, comm_a_t1, comm_v_r1, score_comm_tva = self.transR_common_2(common_t, common_v, common_a,
                                                                               neg_comm_t, common_v, neg_comm_a,
                                                                               mode='tva')
        comm_v_h1, comm_a_t2, comm_t_r1, score_comm_vta = self.transR_common_2(common_t, common_v, common_a,
                                                                               common_t, neg_comm_v, neg_comm_a,
                                                                               mode='vta')
        comm_v_h2, comm_t_t1, comm_a_r2, score_comm_vat = self.transR_common_2(common_t, common_v, common_a,
                                                                               neg_comm_t, neg_comm_v, common_a,
                                                                               mode='vat')
        comm_a_h1, comm_v_t2, comm_t_r2, score_comm_atv = self.transR_common_2(common_t, common_v, common_a,
                                                                               common_t, neg_comm_v, neg_comm_a,
                                                                               mode='atv')
        comm_a_h2, comm_t_t2, comm_v_r2, score_comm_avt = self.transR_common_2(common_t, common_v, common_a,
                                                                               neg_comm_t, common_v, neg_comm_a,
                                                                               mode='avt')

        # s-s-s
        s_t_h = (t_h1 + t_h2) * 0.5
        s_v_h = (v_h1 + v_h2) * 0.5
        s_a_h = (a_h1 + a_h2) * 0.5

        # s-e-s
        e_t_h = (cte_h1 + cte_h2) * 0.5
        e_v_h = (cve_h1 + cve_h2) * 0.5
        e_a_h = (cae_h1 + cae_h2) * 0.5

        #s-c-s
        c_t_h = (ct_h1 + ct_h2) * 0.5
        c_v_h = (cv_h1 + cv_h2) * 0.5
        c_a_h = (ca_h1 + ca_h2) * 0.5

        cr_score_loss = torch.mean(score_tcv) + torch.mean(score_tca) + torch.mean(score_vca) + torch.mean(
            score_vct) + torch.mean(score_acv) + torch.mean(score_act)

        score_loss = torch.mean(score_tav) + torch.mean(score_tva) + torch.mean(score_vat) + torch.mean(
            score_vta) + torch.mean(score_atv) + torch.mean(score_avt)
        score_loss = score_loss / 6

        score_common_tri = torch.mean(score_comm_tav) + torch.mean(score_comm_tva) + torch.mean(
            score_comm_vat) + torch.mean(score_comm_vta) + torch.mean(score_comm_avt) + torch.mean(score_comm_atv)
        score_common_tri = score_common_tri / 6

        text_lsr = self.th_weight(torch.cat([s_t_h,c_t_h,e_t_h], dim=-1))
        visual_lsr = self.vh_weight(torch.cat([s_v_h,c_v_h,e_v_h], dim=-1))
        audio_lsr = self.ah_weight(torch.cat([s_a_h,c_a_h,e_a_h], dim=-1))
        c_feature = torch.cat([common_t, common_v, common_a], dim=-1)  # (b,l,3d)

        text_lsr = text_lsr.view(-1, bsz, self.task_config.hidden_size)
        visual_lsr = visual_lsr.view(-1, bsz, self.task_config.hidden_size)
        audio_lsr = audio_lsr.view(-1, bsz, self.task_config.hidden_size)

        h_l_with_as = self.trans_l_with_a(text_lsr, audio_lsr, audio_lsr)
        h_l_with_vs = self.trans_l_with_v(text_lsr, visual_lsr, visual_lsr)
        h_ls = torch.cat([h_l_with_as, h_l_with_vs], dim=2)
        h_ls = self.trans_l_mem(h_ls)  # (L,B,2D)


        h_v_with_as = self.trans_v_with_a(visual_lsr, audio_lsr, audio_lsr)
        h_v_with_ls = self.trans_v_with_l(visual_lsr, text_lsr, text_lsr)
        h_vs = torch.cat([h_v_with_as, h_v_with_ls], dim=2)
        h_vs = self.trans_v_mem(h_vs)  # (L,B,2D)

        h_a_with_ls = self.trans_a_with_l(audio_lsr, text_lsr, text_lsr)
        h_a_with_vs = self.trans_a_with_v(audio_lsr, visual_lsr, visual_lsr)
        h_as = torch.cat([h_a_with_ls, h_a_with_vs], dim=2)
        h_as = self.trans_a_mem(h_as)  # (L,B,2D)

        last_h_l = self.weight_t(h_ls.view(bsz, -1))
        last_h_v = self.weight_v(h_vs.view(bsz, -1))
        last_h_a = self.weight_a(h_as.view(bsz, -1))
        last_h_c = self.weight_c(c_feature.view(bsz, -1))

        time_feature = time_feature[:,-1]

        last_hs = torch.stack([last_h_l, last_h_v, last_h_a, last_h_c], dim=1)
        logits = self.clf(last_hs)
        time_logits = self.time_clf(time_feature.view(bsz,-1))
        time_scores = self.sigmoid(time_logits)

        pred_scores = self.sigmoid(logits)
        scores = torch.mean(pred_scores, dim=1)

        scores = (time_scores + scores) / 2
        final_label = getBinaryTensor(scores, boundary=self.task_config.binary_threshold)

        res = {
            'recon_loss_t': recon_loss_t,
            'recon_loss_v': recon_loss_v,
            'recon_loss_a': recon_loss_a,
            'score_loss': score_loss,
            'label_loss': loss_sim,
            'pred_label': final_label,
            'true_label': groundTruth_labels,
            'predict_scores': logits,
            'cr_score_loss': cr_score_loss,
            'common_score_loss': score_common_tri,
            'sim_score_loss': sim_score_loss,
            'loss_ort': loss_ort,
            'time_logits': time_logits,
            'score_emotion': score_emotion,
            'c_sim_t': c_sim_t,
            'c_sim_v': c_sim_v,
            'c_sim_a': c_sim_a,
            's_sim_t': s_sim_t,
            's_sim_v': s_sim_v,
            's_sim_a': s_sim_a,
            'emotion_score_loss':emotion_score_loss,
        }

        return res


