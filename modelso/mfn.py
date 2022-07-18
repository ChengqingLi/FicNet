import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.modules.utils import _quadruple
# from modelso.others.rcca import CrissCrossAttention
# from modelso.others.fse import MultiSpectralAttentionLayer
class SCR(nn.Module):
    def __init__(self, planes=[640, 64, 64, 64, 640], stride=(1, 1, 1), ksize=3, do_padding=False, bias=False):
        super(SCR, self).__init__()
        self.ksize = _quadruple(ksize) if isinstance(ksize, int) else ksize
        padding1 = (0, self.ksize[2] // 2, self.ksize[3] // 2) if do_padding else (0, 0, 0)

        self.conv1x1_in = nn.Sequential(nn.Conv2d(planes[0], planes[1], kernel_size=1, bias=False, padding=0),
                                        nn.BatchNorm2d(planes[1]),
                                        nn.ReLU(inplace=True))
        self.conv1 = nn.Sequential(nn.Conv3d(planes[1], planes[2], (1, self.ksize[2], self.ksize[3]),
                                             stride=stride, bias=bias, padding=padding1),
                                   nn.BatchNorm3d(planes[2]),
                                   nn.ReLU(inplace=True))
        self.conv2 = nn.Sequential(nn.Conv3d(planes[2], planes[3], (1, self.ksize[2], self.ksize[3]),
                                             stride=stride, bias=bias, padding=padding1),
                                   nn.BatchNorm3d(planes[3]),
                                   nn.ReLU(inplace=True))
        self.conv1x1_out = nn.Sequential(
            nn.Conv2d(planes[3], planes[4], kernel_size=1, bias=False, padding=0),
            nn.BatchNorm2d(planes[4]))
        # self.fc = nn.Sequential(
        #     nn.Linear(planes[4], planes[4] // 16, bias=False),
        #     nn.ReLU(inplace=True),
        #     nn.Linear(planes[4] // 16, planes[4], bias=False),
        #     nn.Sigmoid()
        # )
    def forward(self, x):
        #b, h, w, u, v = x.shape[0],5,5,5,5
        b, c, h, w, u, v = x.shape
        x = x.view(b, c, h * w, u * v)
        x = self.conv1x1_in(x)   # [80, 640, hw, 25] -> [80, 64, HW, 25]

        c = x.shape[1]
        x = x.view(b, c, h * w, u, v)
        x = self.conv1(x)  # [80, 64, hw, 5, 5] --> [80, 64, hw, 3, 3]
        x = self.conv2(x)  # [80, 64, hw, 3, 3] --> [80, 64, hw, 1, 1]

        c = x.shape[1]
        x = x.view(b, c, h, w)
        x = self.conv1x1_out(x)  # [80, 64, h, w] --> [80, 640, h, w]
        # x = torch.sum(x, dim=[2,3])
        # x = self.fc(x).view(b, 640, 1, 1)
        return x


class mfn(nn.Module):
    def __init__(self, kernel_size=(5, 5), padding=2):
        super(mfn, self).__init__()
        self.kernel_size = kernel_size
        self.unfold = nn.Unfold(kernel_size=kernel_size, padding=padding)
        self.relu = nn.ReLU(inplace=True)
        #self.head=CrissCrossAttention(640)
        self.encoder_dim=640
        c2wh = dict([(84,42), (160,21), (512,6) ,(640,25)])
        mapper_x, mapper_y = get_freq_indices('top1')
        mapper_x = [temp_x * (c2wh[self.encoder_dim] // 7) for temp_x in mapper_x]
        mapper_y = [temp_y * (c2wh[self.encoder_dim] // 7) for temp_y in mapper_y]
        self.dct_layer = MultiSpectralDCTLayer(c2wh[self.encoder_dim],c2wh[self.encoder_dim], mapper_x, mapper_y, self.encoder_dim)
        # self.fc = nn.Sequential(
        #     nn.Linear(self.encoder_dim, self.encoder_dim // 16, bias=False),
        #     nn.ReLU(inplace=True),
        #     nn.Linear(self.encoder_dim // 16, self.encoder_dim, bias=False),
        #     nn.Sigmoid()
        # )
    def forward(self, x):
        b, c, h, w = x.shape
        #x = self.dct_layer(x)
        x = self.relu(x)

        x = F.normalize(x, dim=1, p=2)
        # x=self.head(x)
        # x=self.head(x)
        #x = self.dct_layer(x)
        x = F.normalize(x, dim=1, p=2)
        identity = x
        x = self.unfold(x)  # b, cuv, hw
        x = x.view(b, c, self.kernel_size[0], self.kernel_size[1], h, w)
        # 位置有权重
        q=torch.ones([5,5],dtype=torch.float32).cuda()
        for i in range(5):
            for j in range(5):
                q[i][j]=1/(abs(i-2)+abs(j-2)+1)
        q=q.unsqueeze(2).unsqueeze(3).unsqueeze(0).unsqueeze(0)
        identity=identity.unsqueeze(2).unsqueeze(2)*q
        #  位置有权重
        x = x * identity   # b, c, u, v, h, w * b, c, 1, 1, h, w
        x = x.permute(0, 1, 4, 5, 2, 3).contiguous()  # b, c, h, w, u, v

        x = self.dct_layer(x.view(b, c,h*w,-1))
        x=x.view(b, c,h,w,h,-1)
        # x = torch.sum(x, dim=[2,3])
        # x = self.fc(x).view(b, 640, 1, 1)
        # x=y*x.expand_as(y)
        return x

def get_freq_indices(method):
    assert method in ['top1','top2','top4','top8','top16','top32',
                      'bot1','bot2','bot4','bot8','bot16','bot32',
                      'low1','low2','low4','low8','low16','low32']
    num_freq = int(method[3:])
    if 'top' in method:
        all_top_indices_x = [25,1,6,0,0,1,1,4,5,1,3,0,0,0,3,2,4,6,3,5,5,2,6,5,5,3,3,4,2,2,6,1]
        all_top_indices_y = [15,0,0,5,2,0,2,0,0,6,0,4,6,3,5,2,6,3,3,3,5,1,1,2,4,2,1,1,3,0,5,3]
        mapper_x = all_top_indices_x[:num_freq]
        mapper_y = all_top_indices_y[:num_freq]
    elif 'low' in method:
        all_low_indices_x = [0,0,1,1,0,2,2,1,2,0,3,4,0,1,3,0,1,2,3,4,5,0,1,2,3,4,5,6,1,2,3,4]
        all_low_indices_y = [0,1,0,1,2,0,1,2,2,3,0,0,4,3,1,5,4,3,2,1,0,6,5,4,3,2,1,0,6,5,4,3]
        mapper_x = all_low_indices_x[:num_freq]
        mapper_y = all_low_indices_y[:num_freq]
    elif 'bot' in method:
        all_bot_indices_x = [6,1,3,3,2,4,1,2,4,4,5,1,4,6,2,5,6,1,6,2,2,4,3,3,5,5,6,2,5,5,3,6]
        all_bot_indices_y = [6,4,4,6,6,3,1,4,4,5,6,5,2,2,5,1,4,3,5,0,3,1,1,2,4,2,1,1,5,3,3,3]
        mapper_x = all_bot_indices_x[:num_freq]
        mapper_y = all_bot_indices_y[:num_freq]
    else:
        raise NotImplementedError
    return mapper_x, mapper_y
class MultiSpectralDCTLayer(nn.Module):
    """
    Generate dct filters
    """
    def __init__(self, height, width, mapper_x, mapper_y, channel):
        super(MultiSpectralDCTLayer, self).__init__()

        assert len(mapper_x) == len(mapper_y)
        assert channel % len(mapper_x) == 0

        self.num_freq = len(mapper_x)

        # fixed DCT init
        self.register_buffer('weight', self.get_dct_filter(height, width, mapper_x, mapper_y, channel))

        # fixed random init
        # self.register_buffer('weight', torch.rand(channel, height, width))

        # learnable DCT init
        # self.register_parameter('weight', self.get_dct_filter(height, width, mapper_x, mapper_y, channel))

        # learnable random init
        # self.register_parameter('weight', torch.rand(channel, height, width))

        # num_freq, h, w

    def forward(self, x):
        assert len(x.shape) == 4, 'x must been 4 dimensions, but got ' + str(len(x.shape))
        # n, c, h, w = x.shape

        x = x * self.weight
        result=x
        #result = torch.sum(x, dim=[2,3])
        return result

    def build_filter(self, pos, freq, POS):
        result = math.cos(math.pi * freq * (pos + 0.5) / POS) / math.sqrt(POS)
        if freq == 0:
            return result
        else:
            return result * math.sqrt(2)

    def get_dct_filter(self, tile_size_x, tile_size_y, mapper_x, mapper_y, channel):
        dct_filter = torch.zeros(channel, tile_size_x, tile_size_y)

        c_part = channel // len(mapper_x)

        for i, (u_x, v_y) in enumerate(zip(mapper_x, mapper_y)):
            for t_x in range(tile_size_x):
                for t_y in range(tile_size_y):
                    dct_filter[i * c_part: (i+1)*c_part, t_x, t_y] = self.build_filter(t_x, u_x, tile_size_x) * self.build_filter(t_y, v_y, tile_size_y)

        return dct_filter