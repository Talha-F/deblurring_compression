import torch
import torch.nn as nn

class L1Loss(nn.Module):

    def __init__(self):
        super(L1Loss, self).__init__()
        self.criterion = torch.nn.L1Loss()

    def forward(self, x, y):
        return self.criterion(x,y)

class FFTLoss(nn.Module):
    def __init__(self):
        super(FFTLoss, self).__init__()
        self.L1 = L1Loss()

    def forward(self, x, y):

        pred_fft = torch.view_as_real(torch.fft.rfftn(x, dim=(2,3),norm="backward"))
        gt_fft = torch.view_as_real(torch.fft.rfftn(y, dim=(2,3),norm="backward"))

        loss = self.L1(pred_fft, gt_fft)
        return loss
