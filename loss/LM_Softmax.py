import torch
import torch.nn.functional as F
from torch import nn


class LM_Softmax(nn.Module):
    def __init__(self, num_classes):
        super(LM_Softmax, self).__init__()
        self.num_classes = num_classes
        self.m = torch.tensor([0.1], device='cuda')
        self.s = torch.tensor([1.05], device='cuda')

    def class_angle(self, output, label):
        if len(label) == 0:
            return output

        index = label[0]

        logits_c = output[:, index]
        part1 = output[:, :index]
        part2 = output[:, index + 1:]

        val = torch.where(
            logits_c > 0,
            (logits_c + logits_c * self.m) / self.s,
            (logits_c - logits_c.abs() * self.m) * self.s
        )

        new_tensor = torch.cat((part1, val.unsqueeze(1), part2), dim=1)
        return new_tensor

    def combine(self, source_output, source_label):
        data_set = []
        label_set = []

        for j in range(self.num_classes):
            mask = (source_label == j)
            if mask.any():
                output_j = source_output[mask]
                label_j = source_label[mask]
                new_output = self.class_angle(output_j, label_j)
                data_set.append(new_output)
                label_set.append(label_j.unsqueeze(1))

        if data_set and label_set:
            data = torch.vstack(data_set).cuda()
            label = torch.vstack(label_set).squeeze().cuda()
            return data, label
        else:
            return torch.empty(0, device='cuda'), torch.empty(0, dtype=torch.long, device='cuda')

    def forward(self, source_output, source_label):
        data, label = self.combine(source_output, source_label)
        if data.numel() == 0 or label.numel() == 0:
            return torch.tensor(0.0, device='cuda')
        loss = F.nll_loss(F.log_softmax(data, dim=-1), label)
        return loss


