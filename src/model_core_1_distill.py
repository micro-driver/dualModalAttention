import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Parameter
import torchvision.transforms as transforms

from components.attention import ChannelAttention, SpatialAttention, DualCrossModalAttention
from components.srm_conv import SRMConv2d_simple, SRMConv2d_Separate
from networks.xception import TransferModel


class AngleSimpleLinear(nn.Module):
    """Computes cos of angles between input vectors and weights vectors"""

    def __init__(self, in_features, out_features):
        super(AngleSimpleLinear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = Parameter(torch.Tensor(in_features, out_features))
        self.weight.data.uniform_(-1, 1).renorm_(2, 1, 1e-5).mul_(1e5)

    def forward(self, x):
        cos_theta = F.normalize(x, dim=1).mm(F.normalize(self.weight, dim=0))
        return cos_theta.clamp(-1, 1)


class SRMPixelAttention(nn.Module):
    def __init__(self, in_channels):
        super(SRMPixelAttention, self).__init__()
        # self.srm = SRMConv2d_simple()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, 32, 3, 2, 0, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, 3, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
        )

        self.pa = SpatialAttention()

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, a=1)
                if not m.bias is None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, x_srm):
        # x_srm = self.srm(x)
        fea = self.conv(x_srm)
        att_map = self.pa(fea)

        return att_map


class FeatureFusionModule(nn.Module):
    def __init__(self, in_chan=2048 * 2, out_chan=2048, *args, **kwargs):
        super(FeatureFusionModule, self).__init__()
        self.convblk = nn.Sequential(
            nn.Conv2d(in_chan, out_chan, 1, 1, 0, bias=False),
            nn.BatchNorm2d(out_chan),
            nn.ReLU()
        )
        self.ca = ChannelAttention(out_chan, ratio=16)
        self.init_weight()

    def forward(self, x, y):
        fuse_fea = self.convblk(torch.cat((x, y), dim=1))
        fuse_fea = fuse_fea + fuse_fea * self.ca(fuse_fea)
        return fuse_fea

    def init_weight(self):
        for ly in self.children():
            if isinstance(ly, nn.Conv2d):
                nn.init.kaiming_normal_(ly.weight, a=1)
                if not ly.bias is None:
                    nn.init.constant_(ly.bias, 0)


class Two_Stream_Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.xception_rgb = TransferModel(
            'xception', dropout=0.5, inc=3, return_fea=True)
        self.xception_srm = TransferModel(
            'xception', dropout=0.5, inc=3, return_fea=True)

        self.srm_conv0 = SRMConv2d_simple(inc=3)
        self.srm_conv1 = SRMConv2d_Separate(32, 32)
        self.srm_conv2 = SRMConv2d_Separate(64, 64)
        self.relu = nn.ReLU(inplace=True)

        self.att_map = None
        self.srm_sa = SRMPixelAttention(3)
        self.srm_sa_post = nn.Sequential(
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        )

        self.dual_cma0 = DualCrossModalAttention(in_dim=728, ret_att=False)
        self.dual_cma1 = DualCrossModalAttention(in_dim=728, ret_att=False)

        self.fusion = FeatureFusionModule()
        self.anglelinear = AngleSimpleLinear(2048, 2)

        self.att_dic = {}

        #for mutual distillation
        self.K = 2048
        self.T = 0.07
        self.n_dim = self.xception_rgb.model.block8.in_filters

        self.index_g = 0
        self.register_buffer('memory_g', torch.randn(self.K, self.n_dim))
        self.memory_g = F.normalize(self.memory_g)

        self.index_p = 0
        self.register_buffer('memory_p', torch.randn(self.K, self.n_dim))
        self.memory_p = F.normalize(self.memory_p)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

    def update_pointer_g(self, bsz):
        self.index_g = (self.index_g + bsz) % self.K

    def update_memory_g(self, k, queue):
        with torch.no_grad():
            num_neg = k.shape[0]
            out_ids = torch.arange(num_neg).cuda()
            out_ids = torch.fmod(out_ids + self.index_g, self.K).long()
            queue.index_copy_(0, out_ids, k)

    def update_pointer_p(self, bsz):
        self.index_p = (self.index_p + bsz) % self.K

    def update_memory_p(self, k, queue):
        with torch.no_grad():
            num_neg = k.shape[0]
            out_ids = torch.arange(num_neg).cuda()
            out_ids = torch.fmod(out_ids + self.index_p, self.K).long()
            queue.index_copy_(0, out_ids, k)

    def compute_logits(self, q, k, queue):
        bsz = q.shape[0]
        pos = torch.bmm(q.view(bsz, 1, -1), k.view(bsz, -1, 1))
        pos = pos.view(bsz, 1)

        neg = torch.mm(queue, q.transpose(1, 0))
        neg = neg.transpose(0, 1)

        out = torch.cat((pos, neg), dim=1)
        out = torch.div(out, self.T)
        out = out.squeeze().contiguous()

        return out

    def features(self, x):
        srm = self.srm_conv0(x)

        x = self.xception_rgb.model.fea_part1_0(x)
        y = self.xception_srm.model.fea_part1_0(srm) \
            + self.srm_conv1(x)
        y = self.relu(y)

        x = self.xception_rgb.model.fea_part1_1(x)
        y = self.xception_srm.model.fea_part1_1(y) \
            + self.srm_conv2(x)
        y = self.relu(y)

        # srm guided spatial attention
        self.att_map = self.srm_sa(srm)
        x = x * self.att_map + x
        x = self.srm_sa_post(x)

        x = self.xception_rgb.model.fea_part2(x)
        y = self.xception_srm.model.fea_part2(y)

        # x, y = self.dual_cma0(x, y)

        x = self.xception_rgb.model.fea_part3(x)
        y = self.xception_srm.model.fea_part3(y)

        # x, y = self.dual_cma1(x, y)

        f_g = self.avgpool(x)
        f_g = torch.flatten(f_g, 1)

        f_p = self.avgpool(y)
        f_p = torch.flatten(f_p, 1)

        x = self.xception_rgb.model.fea_part4(x)
        y = self.xception_srm.model.fea_part4(y)

        x = self.xception_rgb.model.fea_part5(x)
        y = self.xception_srm.model.fea_part5(y)

        fea = self.fusion(x, y)

        return fea, f_g, f_p

    def classifier(self, fea):
        out, fea = self.xception_rgb.classifier(fea)
        return out, fea

    def forward(self, x):
        '''
        x: original rgb

        Return:
        out: (B, 2) the output for loss computing
        fea: (B, 1024) the flattened features before the last FC
        att_map: srm spatial attention map
        '''
        feature, f_g, f_p = self.features(x)
        _, fea = self.classifier(feature)
        out = self.anglelinear(fea)

        return out, f_g, f_p


if __name__ == '__main__':
    # t_list = [transforms.ToTensor()]
    # composed_transform = transforms.Compose(t_list)

    # img = cv2.imread('out.jpg')
    # img = cv2.resize(img, (256, 256))
    # image = composed_transform(img)
    # image = image.unsqueeze(0)

    model = Two_Stream_Net()
    dummy = torch.rand((1, 3, 256, 256))
    out = model(dummy)
    print(out)
