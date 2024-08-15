import torch
import torch.nn as nn
import torch.nn.functional as F

class MatchingAttention(nn.Module):
    def __init__(self, input_dim, attn_dim):
        super(MatchingAttention, self).__init__()
        self.attn = nn.Linear(input_dim, attn_dim)
        self.context = nn.Linear(input_dim, attn_dim)
        self.query = nn.Linear(input_dim, attn_dim)
        self.v = nn.Parameter(torch.rand(attn_dim))

    def forward(self, context, query, mask=None):
        # context: [seqlen,label, batch, input_dim]
        # query: [label,batch, input_dim]

        # Linear projections
        context_proj = self.context(context)  # [seqlen,label, batch, attn_dim]
        query_proj = self.query(query).unsqueeze(0)  # [1, label, batch, attn_dim]

        # Compute attention scores
        scores = torch.tanh(context_proj + query_proj)  # [seqlen,label, batch, attn_dim]
        scores = torch.einsum('slbd,d->slb', scores, self.v)  # [seqlen,label, batch]

        # Apply mask (if any)
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)

        # Normalize scores to get attention weights
        alpha = F.softmax(scores, dim=0)  # [seqlen, label, batch]

        # Compute the attention-weighted context
        attn_output = torch.einsum('slb,slbd->lbd', alpha, context)  # [label,batch, input_dim]

        return attn_output, alpha