import torch
import torch.nn as nn

class myLoss(nn.Module):
    def __init__(self,parameters):
        super(myLoss, self).__init__()

    def forward(self, out, targets):
        n, c, w, h = targets.size()
        m = (out+targets)/2
        tmp = 1/m
        part_p = out.mul(tmp)
        part_q = targets.mul(tmp)
        part_a = out.mul(torch.log(part_p)).view(n, c, w * h)
        part_b = targets.mul(torch.log(part_q)).view(n, c, w * h)
        part_a = torch.sum(part_a, 2)  # 10*81
        part_b = torch.sum(part_b, 2)  # 10*81
        part_a_p = part_a[:, 0:68] # 10*68
        part_a_q = part_a[:, 68:81] # 10*13
        part_b_p = part_b[:, 0:68] # 10*68
        part_b_q = part_b[:, 68:81] # 10*13
        part_a = part_a_p+part_b_p # 10*68
        part_b = part_a_q + part_b_q # 10*13
        part_a = torch.sum(part_a, 1)  #10*1
        part_b = torch.sum(part_b, 1)  #10*1
        part_a = torch.mean(part_a, 0)
        part_b = torch.mean(part_b, 0)
        loss = (part_a + part_b)  # 10*68
        return loss
