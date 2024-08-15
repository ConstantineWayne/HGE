import torch
import torch.nn as nn


class HingeLoss(nn.Module):
    def __init__(self):
        super(HingeLoss, self).__init__()

    def compute_cosine(self, x, y):
        # 计算 x 和 y 的余弦相似度
        x_norm = torch.sqrt(torch.sum(torch.pow(x, 2), 1) + 1e-8)
        x_norm = torch.max(x_norm, 1e-8 * torch.ones_like(x_norm))
        y_norm = torch.sqrt(torch.sum(torch.pow(y, 2), 1) + 1e-8)
        y_norm = torch.max(y_norm, 1e-8 * torch.ones_like(y_norm))
        cosine = torch.sum(x * y, 1) / (x_norm * y_norm)
        return cosine

    def forward(self, ids, feats, margin=0.1):
        B, F = feats.shape  # 获取批量大小 B 和特征维度 F
        num_classes = ids.shape[1]

        # 通过重复和重塑操作构建余弦相似度计算的输入
        s = feats.repeat(1, B).view(-1, F)  # B**2 x F
        s_ids = ids.repeat(B, 1)  # B**2 x num_classes

        t = feats.repeat(B, 1)  # B**2 x F
        t_ids = ids.repeat_interleave(B, dim=0)  # B**2 x num_classes

        cosine = self.compute_cosine(s, t)  # B**2
        equal_mask = torch.eye(B, dtype=torch.bool).repeat_interleave(num_classes, dim=0).repeat_interleave(num_classes,
                                                                                                            dim=1)  # B*num_classes x B*num_classes

        cosine = cosine.view(B, B)  # B x B
        cosine = cosine[~torch.eye(B, dtype=torch.bool)].view(B, B - 1)  # B x (B-1)

        s_ids = s_ids.view(B, B, -1)[~torch.eye(B, dtype=torch.bool)].view(B, B - 1, -1)  # B x (B-1) x num_classes
        t_ids = t_ids.view(B, B, -1)[~torch.eye(B, dtype=torch.bool)].view(B, B - 1, -1)  # B x (B-1) x num_classes

        sim_mask = (s_ids == t_ids).all(dim=-1)  # B x (B-1)
        margin = 0.15 * (s_ids - t_ids).abs().sum(dim=-1).float()  # B x (B-1)

        loss = 0
        loss_num = 0

        for i in range(B):
            sim_num = sim_mask[i].sum()  # 计算与第 i 个样本相似的样本数量
            dif_num = B - 1 - sim_num  # 计算与第 i 个样本不同的样本数量
            if not sim_num or not dif_num:
                continue
            sim_cos = cosine[i, sim_mask[i]].reshape(-1, 1).repeat(1, dif_num)
            dif_cos = cosine[i, ~sim_mask[i]].reshape(-1, 1).repeat(1, sim_num).transpose(0, 1)
            t_margin = margin[i, ~sim_mask[i]].reshape(-1, 1).repeat(1, sim_num).transpose(0, 1)

            loss_i = torch.max(torch.zeros_like(sim_cos), t_margin - sim_cos + dif_cos).mean()
            loss += loss_i
            loss_num += 1

        if loss_num == 0:
            loss_num = 1

        loss = loss / loss_num
        return loss
