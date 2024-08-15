

import logging
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
import math
from random import sample

logger = logging.getLogger(__name__)

def gelu(x):
    """Implementation of the gelu activation function.
        For information: OpenAI GPT's gelu is slightly different (and gives slightly different results):
        0.5 * x * (1 + torch.tanh(math.sqrt(2 / math.pi) * (x + 0.044715 * torch.pow(x, 3))))
    """
    return x * 0.5 * (1.0 + torch.erf(x / math.sqrt(2.0)))

def swish(x):
    return x * torch.sigmoid(x)

ACT2FN = {"gelu": gelu, "relu": torch.nn.functional.relu, "swish": swish}

class LayerNorm(nn.Module):
    def __init__(self, hidden_size, eps=1e-12):
        """Construct a layernorm module in the TF style (epsilon inside the square root).
        """
        super(LayerNorm, self).__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.bias = nn.Parameter(torch.zeros(hidden_size))
        self.variance_epsilon = eps

    def forward(self, x):
        u = x.mean(-1, keepdim=True)
        s = (x - u).pow(2).mean(-1, keepdim=True)
        x = (x - u) / torch.sqrt(s + self.variance_epsilon)
        return self.weight * x + self.bias

class PreTrainedModel(nn.Module):
    """ An abstract class to handle weights initialization and
        a simple interface for dowloading and loading pretrained models.
    """
    def __init__(self, config, *inputs, **kwargs):
        super(PreTrainedModel, self).__init__()
        self.config = config

    def init_weights(self, module):
        """ Initialize the weights.
        """
        if isinstance(module, (nn.Linear, nn.Embedding)):
            # Slightly different from the TF version which uses truncated_normal for initialization
            # cf https://github.com/pytorch/pytorch/pull/5617
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
        elif isinstance(module, LayerNorm):
            if 'beta' in dir(module) and 'gamma' in dir(module):
                module.beta.data.zero_()
                module.gamma.data.fill_(1.0)
            else:
                module.bias.data.zero_()
                module.weight.data.fill_(1.0)
        if isinstance(module, nn.Linear) and module.bias is not None:
            module.bias.data.zero_()

    def resize_token_embeddings(self, new_num_tokens=None):
        raise NotImplementedError

    @classmethod
    def init_preweight(cls, model, state_dict, prefix=None, task_config=None):
        old_keys = []
        new_keys = []
        for key in state_dict.keys():
            new_key = None
            if 'gamma' in key:
                new_key = key.replace('gamma', 'weight')
            if 'beta' in key:
                new_key = key.replace('beta', 'bias')
            if new_key:
                old_keys.append(key)
                new_keys.append(new_key)
        for old_key, new_key in zip(old_keys, new_keys):
            state_dict[new_key] = state_dict.pop(old_key)

        if prefix is not None:
            old_keys = []
            new_keys = []
            for key in state_dict.keys():
                old_keys.append(key)
                new_keys.append(prefix + key)
            for old_key, new_key in zip(old_keys, new_keys):
                state_dict[new_key] = state_dict.pop(old_key)

        missing_keys = []
        unexpected_keys = []
        error_msgs = []
        # copy state_dict so _load_from_state_dict can modify it
        metadata = getattr(state_dict, '_metadata', None)
        state_dict = state_dict.copy()
        if metadata is not None:
            state_dict._metadata = metadata

        def load(module, prefix=''):
            local_metadata = {} if metadata is None else metadata.get(prefix[:-1], {})
            module._load_from_state_dict(
                state_dict, prefix, local_metadata, True, missing_keys, unexpected_keys, error_msgs)
            for name, child in module._modules.items():
                if child is not None:
                    load(child, prefix + name + '.')

        load(model, prefix='')

        if prefix is None and (task_config is None or task_config.local_rank == 0):
            logger.info("-" * 20)
            if len(missing_keys) > 0:
                logger.info("Weights of {} not initialized from pretrained model: {}"
                            .format(model.__class__.__name__, "\n   " + "\n   ".join(missing_keys)))
            if len(unexpected_keys) > 0:
                logger.info("Weights from pretrained model not used in {}: {}"
                            .format(model.__class__.__name__, "\n   " + "\n   ".join(unexpected_keys)))
            if len(error_msgs) > 0:
                logger.error("Weights from pretrained model cause errors in {}: {}"
                             .format(model.__class__.__name__, "\n   " + "\n   ".join(error_msgs)))

        return model

    @property
    def dtype(self):
        """
        :obj:`torch.dtype`: The dtype of the module (assuming that all the module parameters have the same dtype).
        """
        try:
            return next(self.parameters()).dtype
        except StopIteration:
            # For nn.DataParallel compatibility in PyTorch 1.5
            def find_tensor_attributes(module: nn.Module):
                tuples = [(k, v) for k, v in module.__dict__.items() if torch.is_tensor(v)]
                return tuples

            gen = self._named_members(get_members_fn=find_tensor_attributes)
            first_tuple = next(gen)
            return first_tuple[1].dtype

    @classmethod
    def from_pretrained(cls, config, state_dict=None,  *inputs, **kwargs):
        """
        Instantiate a PreTrainedModel from a pre-trained model file or a pytorch state dict.
        Download and cache the pre-trained model file if needed.
        """
        # Instantiate model.
        model = cls(config, *inputs, **kwargs)
        if state_dict is None:
            return model
        model = cls.init_preweight(model, state_dict)

        return model


def getBinaryTensor(imgTensor, boundary = 0.35):
    one = torch.ones_like(imgTensor).fill_(1)
    zero = torch.zeros_like(imgTensor).fill_(0)
    return torch.where(imgTensor > boundary, one, zero)


class CTCModule(nn.Module): #
    def __init__(self, in_dim, out_seq_len):
        '''
        This module is performing alignment from A (e.g., audio) to B (e.g., text).
        :param in_dim: Dimension for input modality A
        :param out_seq_len: Sequence length for output modality B
        '''
        super(CTCModule, self).__init__()
        # Use LSTM for predicting the position from A to B
        self.pred_output_position_inclu_blank = nn.LSTM(in_dim, out_seq_len+1, num_layers=2, batch_first=True) # 1 denoting blank
        
        self.out_seq_len = out_seq_len
        
        self.softmax = nn.Softmax(dim=2)
    def forward(self, x):
        '''
        :input x: Input with shape [batch_size x in_seq_len x in_dim]
        '''
        # NOTE that the index 0 refers to blank. 
        pred_output_position_inclu_blank, _ = self.pred_output_position_inclu_blank(x)
        # print(x.shape, pred_output_position_inclu_blank.shape)
        prob_pred_output_position_inclu_blank = self.softmax(pred_output_position_inclu_blank) # batch_size x in_seq_len x out_seq_len+1
        prob_pred_output_position = prob_pred_output_position_inclu_blank[:, :, 1:] # batch_size x in_seq_len x out_seq_len
        prob_pred_output_position = prob_pred_output_position.transpose(1,2) # batch_size x out_seq_len x in_seq_len
        # print(prob_pred_output_position.shape)
        pseudo_aligned_out = torch.bmm(prob_pred_output_position, x) # batch_size x out_seq_len x in_dim
        
        # pseudo_aligned_out is regarded as the aligned A (w.r.t B)
        return pseudo_aligned_out, (pred_output_position_inclu_blank)

class MLAttention(nn.Module):
    def __init__(self, label_num, hidden_size):
        super(MLAttention, self).__init__()
        self.attention = nn.Linear(hidden_size, label_num, bias=False)
        nn.init.xavier_uniform_(self.attention.weight)

    def forward(self, inputs, masks):
        masks = torch.unsqueeze(masks, 1)
        attention = self.attention(inputs).transpose(1,2).masked_fill(masks, -np.inf)
        attention = F.softmax(attention, -1)
        return attention @ inputs, attention

def init_weights(m):
    if isinstance(m, nn.Linear):
        torch.nn.init.xavier_uniform_(m.weight)

class MLLinear(nn.Module):
    def __init__(self, state_list, output_size):
        super(MLLinear, self).__init__()
        # print('hello', state_list)
        self.linear = nn.ModuleList(nn.Linear(in_s, out_s)
                                    for in_s, out_s in zip(state_list[:-1], state_list[1:]))
        for linear in self.linear:
            nn.init.xavier_uniform_(linear.weight)
        self.output = nn.Linear(state_list[-1], output_size)
        nn.init.xavier_uniform_(self.output.weight)

    def forward(self, inputs):
        linear_out = inputs
        for linear in self.linear:
            linear_out = F.relu(linear(linear_out))
        return torch.squeeze(self.output(linear_out), -1)


class TL_SelfAttention(nn.Module):
    def __init__(self, D):
        super(TL_SelfAttention, self).__init__()
        self.query = nn.Linear(D,D)
        self.key = nn.Linear(D,D)
        self.value = nn.Linear(D,D)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x,mask=None):
        B, T, L, D = x.shape

        # Compute query, key, value
        q = self.query(x)  # Shape: (B, T, L, D)
        k = self.key(x)  # Shape: (B, T, L, D)
        v = self.value(x)  # Shape: (B, T, L, D)

        # Permute to get dimensions ready for attention calculation
        q = q.permute(0, 2, 1, 3)  # Shape: (B, L, T, D)
        k = k.permute(0, 2, 1, 3)  # Shape: (B, L, T, D)
        v = v.permute(0, 2, 1, 3)  # Shape: (B, L, T, D)

        # Compute attention scores
        scores = torch.matmul(q, k.transpose(-2, -1)) / (D ** 0.5)  # Shape: (B, L, T, T)

        if mask is not None:
            mask = mask.unsqueeze(1).unsqueeze(2)  # Shape: (B, 1, 1, T)
            scores = scores.masked_fill(mask == 0, float('-inf'))
        # Apply softmax to get attention weights
        attn_weights = self.softmax(scores)  # Shape: (B, L, T, T)

        # Apply attention weights to value
        attended = torch.matmul(attn_weights, v)  # Shape: (B, L, T, D)

        # Permute back to original shape
        attended = attended.permute(0, 2, 1, 3)  # Shape: (B, T, L, D)

        return attended
class MultiLayer_TL_SelfAttention(nn.Module):
    def __init__(self, D, num_layers):
        super(MultiLayer_TL_SelfAttention, self).__init__()
        self.num_layers = num_layers
        self.layers = nn.ModuleList([TL_SelfAttention(D) for _ in range(num_layers)])

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x
class TL_CrossAttention(nn.Module):
    def __init__(self, D):
        super(TL_CrossAttention, self).__init__()
        self.query = nn.Linear(D,D)
        self.key = nn.Linear(D,D)
        self.value = nn.Linear(D,D)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, query,key,value,mask=None):
        """
        :param query:(bsz,seqlen,label,hidden_dim)
        :param key:(bsz,label,hidden_dim)
        :param value:(bsz,label,hidden_dim)
        :return:
        """
        B, T, L, D = query.shape

        # Compute query, key, value
        q = self.query(query)  # Shape: (B, T, L, D)
        k = self.key(key)  # Shape: (B, L, D)
        v = self.value(value)  # Shape: (B, L, D)

        # Compute attention scores
        scores = torch.einsum('btld,bld->btl',q, k) / (D ** 0.5)  # Shape: (B, T, L)
        if mask is not None:
            mask = mask.unsqueeze(-1)  # Shape: (B, T, 1)
            scores = scores.masked_fill(mask == 0, float('-inf'))
        # Apply softmax to get attention weights
        attn_weights = self.softmax(scores)  # Shape: (B, T, L)

        # Apply attention weights to value
        attended = torch.einsum('btl,bld->btld',attn_weights,v)  # Shape: (B, T, L, D)
        return attended

class MultiLayer_TL_CrossAttention(nn.Module):
    def __init__(self, D, num_layers):
        super(MultiLayer_TL_CrossAttention, self).__init__()
        self.num_layers = num_layers
        self.layers = nn.ModuleList([TL_CrossAttention(D) for _ in range(num_layers)])

    def forward(self, query, key, value):
        for layer in self.layers:
            query = layer(query, key, value)
        return query

class Emotion_CrossAttention(nn.Module):
    def __init__(self,D):
        super().__init__()
        self.hidden_dim = D
        self.query = nn.Linear(D,D)
        self.key = nn.Linear(D,D)
        self.value = nn.Linear(D,D)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self,query,key,value):

        q = self.query(query)  # Shape: (B, L, D)
        k = self.key(key)  # Shape: (B, L, D)
        v = self.value(value)  # Shape: (B, L, D)

        scores = torch.matmul(q, k.transpose(-2, -1)) / (self.hidden_dim ** 0.5) # Shape: (B, L, L)
        attn_weights = self.softmax(scores)#(B,L,L)
        attended = torch.matmul(attn_weights, v)#(B,L,D)
        return attended

class MultiLayerEmotion_CrossAttention(nn.Module):
    def __init__(self, D, num_layers):
        super().__init__()
        self.num_layers = num_layers
        self.layers = nn.ModuleList([Emotion_CrossAttention(D) for _ in range(num_layers)])

    def forward(self, query, key, value):
        for layer in self.layers:
            query = layer(query, key, value)
        return query

class MultimodalTransH(nn.Module):
    def __init__(self,hidden_dim,out_dim,regularization):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.text_entity = MLLinear([hidden_dim],out_dim)
        self.video_entity = MLLinear([hidden_dim],out_dim)
        self.audio_entity = MLLinear([hidden_dim],out_dim)
        self.text_relation = MLLinear([hidden_dim],out_dim)
        self.audio_relation = MLLinear([hidden_dim],out_dim)
        self.video_relation = MLLinear([hidden_dim],out_dim)
        #是不是可以这样写
        self.text_norms = MLLinear([hidden_dim],out_dim)
        self.video_norms = MLLinear([hidden_dim],out_dim)
        self.audio_norms = MLLinear([hidden_dim],out_dim)
        self.regularization = regularization

    def project(self, entities, normals):
        normals = F.normalize(normals, p=2, dim=-1)
        return entities - torch.sum(entities * normals, dim=-1, keepdim=True) * normals

    def forward(self,text,video,audio,mode='tva'):
        if mode in ['tva']:
            head = self.text_entity(text)
            relation = self.video_relation(video)
            tail = self.audio_entity(audio)
            norms = self.video_norms(video)

            head = head / torch.norm(head, p=2, dim=-1, keepdim=True)
            tail = tail / torch.norm(tail, p=2, dim=-1, keepdim=True)

            h_e = self.project(head,norms)
            t_e = self.project(tail,norms)

        elif mode in ['tav']:
            head = self.text_entity(text)
            relation = self.audio_relation(audio)
            tail = self.video_relation(video)
            norms = self.audio_norms(audio)

            head = head / torch.norm(head, p=2, dim=-1, keepdim=True)
            tail = tail / torch.norm(tail, p=2, dim=-1, keepdim=True)

            h_e = self.project(head,norms)
            t_e = self.project(tail,norms)

        elif mode in ['vat']:
            head = self.video_entity(video)
            relation = self.audio_relation(audio)
            tail = self.text_entity(text)
            norms = self.audio_norms(audio)

            head = head / torch.norm(head, p=2, dim=-1, keepdim=True)
            tail = tail / torch.norm(tail, p=2, dim=-1, keepdim=True)

            h_e = self.project(head,norms)
            t_e = self.project(tail,norms)

        elif mode in ['vta']:
            head = self.video_entity(video)
            relation = self.text_relation(text)
            tail = self.audio_entity(audio)
            norms = self.text_norms(text)

            head = head / torch.norm(head, p=2, dim=-1, keepdim=True)
            tail = tail / torch.norm(tail, p=2, dim=-1, keepdim=True)

            h_e = self.project(head, norms)
            t_e = self.project(tail, norms)

        elif mode in ['atv']:
            head = self.audio_entity(audio)
            relation = self.text_relation(text)
            tail = self.video_entity(video)
            norms = self.text_norms(text)

            head = head / torch.norm(head, p=2, dim=-1, keepdim=True)
            tail = tail / torch.norm(tail, p=2, dim=-1, keepdim=True)

            h_e = self.project(head, norms)
            t_e = self.project(tail, norms)

        elif mode in ['avt']:
            head = self.audio_entity(audio)
            relation = self.video_relation(video)
            tail = self.text_entity(text)
            norms = self.video_norms(video)

            head = head / torch.norm(head, p=2, dim=-1, keepdim=True)
            tail = tail / torch.norm(tail, p=2, dim=-1, keepdim=True)

            h_e = self.project(head, norms)
            t_e = self.project(tail, norms)

        elif mode in ['t']:
            head = self.text_entity(text)
            relation = self.text_relation(audio)
            tail = self.text_entity(video)
            norms = self.text_norms(audio)

            head = head / torch.norm(head, p=2, dim=-1, keepdim=True)
            tail = tail / torch.norm(tail, p=2, dim=-1, keepdim=True)

            h_e = self.project(head, norms)
            t_e = self.project(tail, norms)

        elif mode in ['a']:
            head = self.audio_entity(text)
            relation = self.audio_relation(audio)
            tail = self.audio_entity(video)
            norms = self.audio_norms(audio)

            head = head / torch.norm(head, p=2, dim=-1, keepdim=True)
            tail = tail / torch.norm(tail, p=2, dim=-1, keepdim=True)

            h_e = self.project(head, norms)
            t_e = self.project(tail, norms)

        elif mode in ['v']:
            head = self.video_entity(text)
            relation = self.video_relation(audio)
            tail = self.video_entity(video)
            norms = self.video_norms(audio)

            head = head / torch.norm(head, p=2, dim=-1, keepdim=True)
            tail = tail / torch.norm(tail, p=2, dim=-1, keepdim=True)

            h_e = self.project(head, norms)
            t_e = self.project(tail, norms)
        return h_e, t_e, relation

    def score(self,h,t,r,neg_h,neg_t,neg_r,margin,mode='tva'):
        h_e,t_e,r_e = self.forward(h,t,r,mode)
        neg_h_e,neg_t_e,neg_r_e = self.forward(neg_h,neg_t,neg_r,mode)
        pos_score = torch.norm(h_e+r_e-t_e,p=2,dim=-1)
        neg_score = torch.norm(neg_h_e+neg_r_e-neg_t_e,p=2,dim=-1)
        reg_term = self.regularization * (
                torch.mean(h_e ** 2) + torch.mean(t_e ** 2) +
                torch.mean(neg_h_e ** 2) + torch.mean(neg_t_e ** 2)
        )
        loss = torch.mean(torch.relu(margin + pos_score - neg_score))
        return h_e,t_e,r_e,loss + reg_term

class TransR(nn.Module):
    def __init__(self, hidden_dim,ent_dim, rel_dim):
        super(TransR, self).__init__()
        self.ent_dim = ent_dim
        self.rel_dim = rel_dim
        self.norm = 2

        self.text_entity = MLLinear([hidden_dim], ent_dim)
        self.video_entity = MLLinear([hidden_dim], ent_dim)
        self.audio_entity = MLLinear([hidden_dim], ent_dim)
        self.text_relation = MLLinear([hidden_dim], rel_dim)
        self.audio_relation = MLLinear([hidden_dim], rel_dim)
        self.video_relation = MLLinear([hidden_dim], rel_dim)
        self.transfer_matrix_text = MLLinear([hidden_dim], ent_dim * rel_dim)
        self.transfer_matrix_video = MLLinear([hidden_dim], ent_dim * rel_dim)
        self.transfer_matrix_audio = MLLinear([hidden_dim], ent_dim * rel_dim)

    def _transfer(self, e, r, mode='tva'):
        #e:[bsz,label,ent_dim]
        #r:[bsz,label,rel_dim,ent_dim]
        if mode in ['tva','avt','v']:
            transfer_matrix = self.transfer_matrix_video
        elif mode in ['tav','vat','a']:
            transfer_matrix = self.transfer_matrix_audio
        else:
            transfer_matrix = self.transfer_matrix_text
        r_m = transfer_matrix(r).view(r.size(0),-1, self.rel_dim, self.ent_dim)
        # e = e.view(-1, 1, self.ent_dim)
        e = torch.matmul(e.unsqueeze(2), r_m).squeeze(2) #(bsz,label,rel_dim)
        return e

    def forward(self,text,video,audio,mode='tva'):
        if mode in ['tva']:
            head = self.text_entity(text)
            relation = self.video_relation(video)
            tail = self.audio_entity(audio)
            head = F.normalize(head,p=2,dim=-1)
            relation = F.normalize(relation,p=2,dim=-1)
            tail = F.normalize(tail,p=2,dim=-1)

            h_e = self._transfer(head,video,mode)
            t_e = self._transfer(tail,video,mode)

            h_e = F.normalize(h_e,p=2,dim=-1)
            t_e = F.normalize(t_e,p=2,dim=-1)

        elif mode in ['tav']:
            head = self.text_entity(text)
            relation = self.audio_relation(audio)
            tail = self.video_relation(video)

            head = F.normalize(head, p=2, dim=-1)
            relation = F.normalize(relation, p=2, dim=-1)
            tail = F.normalize(tail, p=2, dim=-1)

            h_e = self._transfer(head, audio, mode)
            t_e = self._transfer(tail, audio, mode)

            h_e = F.normalize(h_e, p=2, dim=-1)
            t_e = F.normalize(t_e, p=2, dim=-1)

        elif mode in ['vat']:
            head = self.video_entity(video)
            relation = self.audio_relation(audio)
            tail = self.text_entity(text)

            head = F.normalize(head, p=2, dim=-1)
            relation = F.normalize(relation, p=2, dim=-1)
            tail = F.normalize(tail, p=2, dim=-1)

            h_e = self._transfer(head, audio, mode)
            t_e = self._transfer(tail, audio, mode)

            h_e = F.normalize(h_e, p=2, dim=-1)
            t_e = F.normalize(t_e, p=2, dim=-1)

        elif mode in ['vta']:
            head = self.video_entity(video)
            relation = self.text_relation(text)
            tail = self.audio_entity(audio)

            head = F.normalize(head, p=2, dim=-1)
            relation = F.normalize(relation, p=2, dim=-1)
            tail = F.normalize(tail, p=2, dim=-1)

            h_e = self._transfer(head, text, mode)
            t_e = self._transfer(tail, text, mode)

            h_e = F.normalize(h_e, p=2, dim=-1)
            t_e = F.normalize(t_e, p=2, dim=-1)

        elif mode in ['atv']:
            head = self.audio_entity(audio)
            relation = self.text_relation(text)
            tail = self.video_entity(video)

            head = F.normalize(head, p=2, dim=-1)
            relation = F.normalize(relation, p=2, dim=-1)
            tail = F.normalize(tail, p=2, dim=-1)

            h_e = self._transfer(head, text, mode)
            t_e = self._transfer(tail, text, mode)

            h_e = F.normalize(h_e, p=2, dim=-1)
            t_e = F.normalize(t_e, p=2, dim=-1)

        elif mode in ['avt']:
            head = self.audio_entity(audio)
            relation = self.video_relation(video)
            tail = self.text_entity(text)

            head = F.normalize(head, p=2, dim=-1)
            relation = F.normalize(relation, p=2, dim=-1)
            tail = F.normalize(tail, p=2, dim=-1)

            h_e = self._transfer(head, video, mode)
            t_e = self._transfer(tail, video, mode)

            h_e = F.normalize(h_e, p=2, dim=-1)
            t_e = F.normalize(t_e, p=2, dim=-1)

        elif mode in ['t']:
            head = self.text_entity(text)
            relation = self.text_relation(audio)
            tail = self.text_entity(video)

            head = F.normalize(head, p=2, dim=-1)
            relation = F.normalize(relation, p=2, dim=-1)
            tail = F.normalize(tail, p=2, dim=-1)

            h_e = self._transfer(head, audio, mode)
            t_e = self._transfer(tail, audio, mode)

            h_e = F.normalize(h_e, p=2, dim=-1)
            t_e = F.normalize(t_e, p=2, dim=-1)

        elif mode in ['a']:
            head = self.audio_entity(text)
            relation = self.audio_relation(audio)
            tail = self.audio_entity(video)

            head = F.normalize(head, p=2, dim=-1)
            relation = F.normalize(relation, p=2, dim=-1)
            tail = F.normalize(tail, p=2, dim=-1)

            h_e = self._transfer(head, audio, mode)
            t_e = self._transfer(tail, audio, mode)

            h_e = F.normalize(h_e, p=2, dim=-1)
            t_e = F.normalize(t_e, p=2, dim=-1)

        elif mode in ['v']:
            head = self.video_entity(text)
            relation = self.video_relation(audio)
            tail = self.video_entity(video)

            head = F.normalize(head, p=2, dim=-1)
            relation = F.normalize(relation, p=2, dim=-1)
            tail = F.normalize(tail, p=2, dim=-1)

            h_e = self._transfer(head, audio, mode)
            t_e = self._transfer(tail, audio, mode)

            h_e = F.normalize(h_e, p=2, dim=-1)
            t_e = F.normalize(t_e, p=2, dim=-1)
        return h_e, t_e, relation

    def score(self, pos_h, pos_t, pos_r, neg_h, neg_t, neg_r,margin,mode='tva'):
        pos_h_e, pos_t_e, pos_r_e = self.forward(pos_h, pos_t, pos_r,mode)
        neg_h_e, neg_t_e, neg_r_e = self.forward(neg_h, neg_t, neg_r,mode)

        # pos_score = F.normalize(pos_h_e + pos_r_e - pos_t_e, self.norm, dim=-1)
        # neg_score = F.normalize(neg_h_e + neg_r_e - neg_t_e, self.norm, dim=-1)
        pos_score = pos_h_e + pos_r_e - pos_t_e
        neg_score = neg_h_e + neg_r_e - neg_t_e

        loss = torch.mean(torch.relu(pos_score - neg_score + margin))
        return pos_h_e,pos_t_e,pos_r_e,loss



class KG2E(nn.Module):
    def __init__(self, hidden_dim, ent_dim, rel_dim, margin=1.0, sim="KL", vmin=0.03, vmax=3.0):
        super(KG2E, self).__init__()
        assert (sim in ["KL", "EL"])
        self.model = "KG2E"
        self.margin = margin
        self.ke = ent_dim
        self.sim = sim
        self.vmin = vmin
        self.vmax = vmax

        self.text_entity = nn.Linear(hidden_dim,ent_dim)
        self.video_entity = nn.Linear(hidden_dim,ent_dim)
        self.audio_entity = nn.Linear(hidden_dim,ent_dim)

        # self.text_entity = MLLinear([hidden_dim],ent_dim)
        # self.video_entity = MLLinear([hidden_dim],ent_dim)
        # self.audio_entity = MLLinear([hidden_dim],ent_dim)

        self.text_entity_co = nn.Linear(hidden_dim,ent_dim)
        self.video_entity_co = nn.Linear(hidden_dim,ent_dim)
        self.audio_entity_co = nn.Linear(hidden_dim,ent_dim)
        # self.text_entity_co = MLLinear([hidden_dim],ent_dim)
        # self.video_entity_co = MLLinear([hidden_dim],ent_dim)
        # self.audio_entity_co = MLLinear([hidden_dim],ent_dim)

        self.text_relation = nn.Linear(hidden_dim,rel_dim)
        self.audio_relation = nn.Linear(hidden_dim,rel_dim)
        self.video_relation = nn.Linear(hidden_dim,rel_dim)
        # self.text_relation = MLLinear([hidden_dim],rel_dim)
        # self.audio_relation = MLLinear([hidden_dim],rel_dim)
        # self.video_relation = MLLinear([hidden_dim],rel_dim)

        self.text_relation_co = nn.Linear(hidden_dim,rel_dim)
        self.audio_relation_co = nn.Linear(hidden_dim,rel_dim)
        self.video_relation_co = nn.Linear(hidden_dim,rel_dim)
        # self.text_relation_co = MLLinear([hidden_dim],rel_dim)
        # self.audio_relation_co = MLLinear([hidden_dim],rel_dim)
        # self.video_relation_co = MLLinear([hidden_dim],rel_dim)

        self._init_weights()

    '''
    Calculate the KL loss between T-H distribution and R distribution.
    There are four parts in loss function.
    '''
    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)

    def KLScore(self, **kwargs):
        # Calculate KL(e, r)
        losep1 = torch.sum(kwargs["errorv"]/kwargs["relationv"], dim=1)
        losep2 = torch.sum((kwargs["relationm"]-kwargs["errorm"])**2 / kwargs["relationv"], dim=1)
        KLer = (losep1 + losep2 - self.ke) / 2

        # Calculate KL(r, e)
        losep1 = torch.sum(kwargs["relationv"]/kwargs["errorv"], dim=1)
        losep2 = torch.sum((kwargs["errorm"] - kwargs["relationm"]) ** 2 / kwargs["errorv"], dim=1)
        KLre = (losep1 + losep2 - self.ke) / 2
        return (KLer + KLre) / 2

    '''
    Calculate the EL loss between T-H distribution and R distribution.
    There are three parts in loss function.
    '''
    def ELScore(self, **kwargs):
        losep1 = torch.sum((kwargs["errorm"] - kwargs["relationm"]) ** 2 / (kwargs["errorv"] + kwargs["relationv"]), dim=1)
        losep2 = torch.sum(torch.log(kwargs["errorv"]+kwargs["relationv"]), dim=1)
        return (losep1 + losep2) / 2

    '''
    Calculate the score of triples
    Step1: Split input as head, relation and tail index
    Step2: Transform index tensor to embedding
    Step3: Calculate the score with "KL" or "EL"
    Step4: Return the score 
    '''
    def scoreOp(self, text, video, audio,similar=None,mode='tva'):
        if mode in ['tva']:
            headm = self.text_entity(text)
            headv= self.text_entity_co(text)

            tailm= self.audio_entity(audio)
            tailv= self.audio_entity_co(audio)

            relationm= self.video_relation(video)
            relationv = self.video_relation_co(video)

        elif mode in ['tav']:
            headm = self.text_entity(text)
            headv = self.text_entity_co(text)

            tailm= self.video_entity(video)
            tailv = self.video_entity_co(video)

            relationm = self.audio_relation(audio)
            relationv= self.audio_relation_co(audio)

        elif mode in ['vat']:
            headm= self.video_entity(video)
            headv = self.video_entity_co(video)

            tailm = self.text_entity(text)
            tailv = self.text_entity_co(text)

            relationm = self.audio_relation(audio)
            relationv = self.audio_relation_co(audio)

        elif mode in ['vta']:
            headm = self.video_entity(video)
            headv = self.video_entity_co(video)

            tailm = self.audio_entity(audio)
            tailv = self.audio_entity_co(audio)

            relationm= self.text_relation(text)
            relationv= self.text_relation_co(text)

        elif mode in ['avt']:
            headm= self.audio_entity(audio)
            headv = self.audio_entity_co(audio)

            tailm= self.text_entity(text)
            tailv = self.text_entity_co(text)

            relationm= self.video_relation(video)
            relationv= self.video_relation_co(video)

        elif mode in ['atv']:
            headm= self.audio_entity(audio)
            headv= self.audio_entity_co(audio)

            tailm= self.video_entity(video)
            tailv= self.video_entity_co(video)

            relationm = self.text_relation(text)
            relationv = self.text_relation_co(text)

        elif mode in ['t']:
            headm = self.text_entity(text)
            headv = self.text_entity_co(text)

            tailm = self.text_entity(video)
            tailv = self.text_entity_co(video)

            relationm = self.text_relation(audio)
            relationv = self.text_relation_co(audio)

        elif mode in ['v']:
            headm = self.video_entity(text)
            headv= self.video_entity_co(text)

            tailm = self.video_entity(video)
            tailv = self.video_entity_co(video)

            relationm = self.video_relation(audio)
            relationv= self.video_relation_co(audio)

        else:
            headm = self.audio_entity(text)
            headv = self.audio_entity_co(text)

            tailm = self.audio_entity(video)
            tailv = self.audio_entity_co(video)

            relationm= self.audio_relation(audio)
            relationv = self.audio_relation_co(audio)

        headm = F.normalize(headm,dim=-1,p=2)
        tailm = F.normalize(tailm,dim=-1,p=2)
        relationm = F.normalize(relationm,dim=-1,p=2)

        headv = torch.clamp(headv,self.vmin,self.vmax)
        tailv = torch.clamp(tailv,self.vmin,self.vmax)
        relationv = torch.clamp(relationv,self.vmin,self.vmax)

        if similar is not None:
            relationm = torch.einsum('b,bld->bld',similar,relationm)
            relationv = torch.einsum('b,bld->bld',similar,relationv)
        # total_loss = loss_hm + loss_hv + loss_rm + loss_rv + loss_tv + loss_tm

        errorm = tailm - headm
        errorv = tailv + headv
        pos_score = self.KLScore(relationm=relationm, relationv=relationv, errorm=errorm, errorv=errorv)
        # if similar is not None:
        #     pos_score = torch.einsum('b,bl->bl',similar,pos_score)
        if self.sim == "KL":
            return headm , tailm, relationm, pos_score
        elif self.sim == "EL":
            return headm , tailm, relationm,self.ELScore(relationm=relationm, relationv=relationv, errorm=errorm, errorv=errorv)
        else:
            print("ERROR : Sim %s is not supported!" % self.sim)
            exit(1)
    def generate_neg(self,text,video,audio,neg_text,neg_video,neg_audio,mode='tva'):
        #0代表head, 1代表tail
        index = [0,1]
        ava_index = sample(index,1)#选择的代表保留
        if mode in ['tva']:
            if ava_index == 0:
                neg_text = text
            else:
                neg_audio = audio
        elif mode in ['tav']:
            if ava_index == 0:
                neg_text = text
            else:
                neg_video = video
        elif mode in ['avt']:
            if ava_index == 0:
                neg_audio = audio
            else:
                neg_text = text
        elif mode in ['atv']:
            if ava_index == 0:
                neg_audio = audio
            else:
                neg_video = video
        elif mode in ['vat']:
            if ava_index == 0:
                neg_video = video
            else:
                neg_text = text
        elif mode in ['vta']:
            if ava_index == 0:
                neg_video = video
            else:
                neg_audio = audio
        else:
            return text,video,audio,neg_text,neg_video,neg_audio
        return text,video,audio,neg_text,neg_video,neg_audio

    def forward(self, text,video, audio, neg_text,neg_video,neg_audio,pos_similar=None,mode='tva'):
        # Calculate score
        pos_head, pos_tail, pos_relation, posScore = self.scoreOp(text,video,audio,similar=None,mode=mode)
        # text,video,audio,neg_text,neg_video,neg_audio = self.generate_neg(text,video,audio,neg_text,neg_video,neg_audio,mode)
        neg_head, neg_tail, neg_relation, negScore = self.scoreOp(neg_text,neg_video,neg_audio,mode=mode)
        posScore, negScore = F.normalize(posScore), F.normalize(negScore)
        score = torch.mean((F.relu(input=posScore-negScore+self.margin)))
        return pos_head, pos_tail, pos_relation, score

