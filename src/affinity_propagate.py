import torch.nn as nn
import torch
import torch.nn.functional as F


class Affinity_Propagate(nn.Module):

    def __init__(self,
                 prop_time,
                 prop_kernel,
                 norm_type='8sum'):
        """
        Inputs:
            prop_time: how many steps for CSPN to perform
            prop_kernel: the size of kernel (current only support 3x3)
            way to normalize affinity
                '8sum': normalize using 8 surrounding neighborhood
                '8sum_abs': normalization enforcing affinity to be positive
                            This will lead the center affinity to be 0
        """
        super(Affinity_Propagate, self).__init__()
        self.prop_time = prop_time
        self.prop_kernel = prop_kernel
        assert prop_kernel == 3, 'this version only support 8 (3x3 - 1) neighborhood'

        self.norm_type = norm_type
        assert norm_type in ['8sum', '8sum_abs']

        self.in_feature = 1
        self.out_feature = 1

        self.sum_conv = nn.Conv3d(in_channels=8,
                                  out_channels=1,
                                  kernel_size=(1, 1, 1),
                                  stride=1,
                                  padding=0,
                                  bias=False)

    def forward(self, guidance, blur_depth, sparse_depth=None):

        # self.sum_conv = nn.Conv3d(in_channels=8,
        #                           out_channels=1,
        #                           kernel_size=(1, 1, 1),
        #                           stride=1,
        #                           padding=0,
        #                           bias=False)
        weight = torch.ones(1, 8, 1, 1, 1).to(guidance.device)
        self.sum_conv.weight = nn.Parameter(weight)  # requires_grad = True
        for param in self.sum_conv.parameters():
            param.requires_grad = False

        gate_wb, gate_sum = self.affinity_normalization(guidance)

        # pad input and convert to 8 channel 3D features
        raw_depth_input = blur_depth

        # blur_depht_pad = nn.ZeroPad2d((1,1,1,1))
        result_depth = blur_depth

        if sparse_depth is not None:
            sparse_mask = sparse_depth.sign()

        for i in range(self.prop_time):
            # one propagation
            spn_kernel = self.prop_kernel
            result_depth = self.pad_blur_depth(result_depth)
            neigbor_weighted_sum = self.sum_conv(gate_wb * result_depth)
            neigbor_weighted_sum = neigbor_weighted_sum.squeeze(1)
            neigbor_weighted_sum = neigbor_weighted_sum[:, :, 1:-1, 1:-1]
            result_depth = neigbor_weighted_sum

            if '8sum' in self.norm_type:
                result_depth = (1.0 - gate_sum) * raw_depth_input + result_depth
            else:
                raise ValueError('unknown norm %s' % self.norm_type)

            if sparse_depth is not None:
                result_depth = (1 - sparse_mask) * result_depth + sparse_mask * raw_depth_input

        return result_depth

    def affinity_normalization(self, guidance):

        # normalize features
        if 'abs' in self.norm_type:
            guidance = torch.abs(guidance)

        gate1_wb_cmb = guidance.narrow(1, 0, self.out_feature)
        gate2_wb_cmb = guidance.narrow(1, 1 * self.out_feature, self.out_feature)
        gate3_wb_cmb = guidance.narrow(1, 2 * self.out_feature, self.out_feature)
        gate4_wb_cmb = guidance.narrow(1, 3 * self.out_feature, self.out_feature)
        gate5_wb_cmb = guidance.narrow(1, 4 * self.out_feature, self.out_feature)
        gate6_wb_cmb = guidance.narrow(1, 5 * self.out_feature, self.out_feature)
        gate7_wb_cmb = guidance.narrow(1, 6 * self.out_feature, self.out_feature)
        gate8_wb_cmb = guidance.narrow(1, 7 * self.out_feature, self.out_feature)

        # gate1:left_top, gate2:center_top, gate3:right_top
        # gate4:left_center,              , gate5: right_center
        # gate6:left_bottom, gate7: center_bottom, gate8: right_bottm

        # ZeroPad2d: left, right, top, bottom
        # top pad
        left_top_pad = nn.ZeroPad2d((0, 2, 0, 2))
        gate1_wb_cmb = left_top_pad(gate1_wb_cmb).unsqueeze(1)  # batch, 1, 1, h+2, w+2

        center_top_pad = nn.ZeroPad2d((1, 1, 0, 2))
        gate2_wb_cmb = center_top_pad(gate2_wb_cmb).unsqueeze(1)

        right_top_pad = nn.ZeroPad2d((2, 0, 0, 2))
        gate3_wb_cmb = right_top_pad(gate3_wb_cmb).unsqueeze(1)

        # center pad
        left_center_pad = nn.ZeroPad2d((0, 2, 1, 1))
        gate4_wb_cmb = left_center_pad(gate4_wb_cmb).unsqueeze(1)

        right_center_pad = nn.ZeroPad2d((2, 0, 1, 1))
        gate5_wb_cmb = right_center_pad(gate5_wb_cmb).unsqueeze(1)

        # bottom pad
        left_bottom_pad = nn.ZeroPad2d((0, 2, 2, 0))
        gate6_wb_cmb = left_bottom_pad(gate6_wb_cmb).unsqueeze(1)

        center_bottom_pad = nn.ZeroPad2d((1, 1, 2, 0))
        gate7_wb_cmb = center_bottom_pad(gate7_wb_cmb).unsqueeze(1)

        right_bottm_pad = nn.ZeroPad2d((2, 0, 2, 0))
        gate8_wb_cmb = right_bottm_pad(gate8_wb_cmb).unsqueeze(1)

        gate_wb = torch.cat((gate1_wb_cmb, gate2_wb_cmb, gate3_wb_cmb, gate4_wb_cmb,
                             gate5_wb_cmb, gate6_wb_cmb, gate7_wb_cmb, gate8_wb_cmb), 1)

        # normalize affinity using their abs sum
        gate_wb_abs = torch.abs(gate_wb)

        abs_weight = self.sum_conv(gate_wb_abs)

        gate_wb = torch.div(gate_wb, abs_weight)

        gate_sum = self.sum_conv(gate_wb)

        gate_sum = gate_sum.squeeze(1)
        gate_sum = gate_sum[:, :, 1:-1, 1:-1]

        return gate_wb, gate_sum

    def pad_blur_depth(self, blur_depth):
        # top pad
        left_top_pad = nn.ZeroPad2d((0, 2, 0, 2))
        blur_depth_1 = left_top_pad(blur_depth).unsqueeze(1)
        center_top_pad = nn.ZeroPad2d((1, 1, 0, 2))
        blur_depth_2 = center_top_pad(blur_depth).unsqueeze(1)
        right_top_pad = nn.ZeroPad2d((2, 0, 0, 2))
        blur_depth_3 = right_top_pad(blur_depth).unsqueeze(1)

        # center pad
        left_center_pad = nn.ZeroPad2d((0, 2, 1, 1))
        blur_depth_4 = left_center_pad(blur_depth).unsqueeze(1)
        right_center_pad = nn.ZeroPad2d((2, 0, 1, 1))
        blur_depth_5 = right_center_pad(blur_depth).unsqueeze(1)

        # bottom pad
        left_bottom_pad = nn.ZeroPad2d((0, 2, 2, 0))
        blur_depth_6 = left_bottom_pad(blur_depth).unsqueeze(1)
        center_bottom_pad = nn.ZeroPad2d((1, 1, 2, 0))
        blur_depth_7 = center_bottom_pad(blur_depth).unsqueeze(1)
        right_bottm_pad = nn.ZeroPad2d((2, 0, 2, 0))
        blur_depth_8 = right_bottm_pad(blur_depth).unsqueeze(1)

        result_depth = torch.cat((blur_depth_1, blur_depth_2, blur_depth_3, blur_depth_4,
                                  blur_depth_5, blur_depth_6, blur_depth_7, blur_depth_8), 1)
        return result_depth

    def normalize_gate(self, guidance):
        gate1_x1_g1 = guidance.narrow(1, 0, 1)
        gate1_x1_g2 = guidance.narrow(1, 1, 1)
        gate1_x1_g1_abs = torch.abs(gate1_x1_g1)
        gate1_x1_g2_abs = torch.abs(gate1_x1_g2)
        elesum_gate1_x1 = torch.add(gate1_x1_g1_abs, gate1_x1_g2_abs)
        gate1_x1_g1_cmb = torch.div(gate1_x1_g1, elesum_gate1_x1)
        gate1_x1_g2_cmb = torch.div(gate1_x1_g2, elesum_gate1_x1)
        return gate1_x1_g1_cmb, gate1_x1_g2_cmb

    def max_of_4_tensor(self, element1, element2, element3, element4):
        max_element1_2 = torch.max(element1, element2)
        max_element3_4 = torch.max(element3, element4)
        return torch.max(max_element1_2, max_element3_4)

    def max_of_8_tensor(self, element1, element2, element3, element4, element5, element6, element7, element8):
        max_element1_2 = self.max_of_4_tensor(element1, element2, element3, element4)
        max_element3_4 = self.max_of_4_tensor(element5, element6, element7, element8)
        return torch.max(max_element1_2, max_element3_4)


class Unpool(nn.Module):
    def __init__(self, num_channels, stride=2):
        super(Unpool, self).__init__()

        self.num_channels = num_channels
        self.stride = stride

        self.weights = torch.autograd.Variable(torch.zeros((num_channels, 1, stride, stride)).cuda())
        self.weights[:, :, 0, 0] = 1

    def forward(self, x):
        # transpose2d for each channels, i.e. groups is output channels
        return F.conv_transpose2d(x, self.weights, stride=self.stride, groups=self.num_channels)


class Simple_Gudi_UpConv_Block_Last_Layer(nn.Module):
    def __init__(self, in_channels, out_channels, oheight=0, owidth=0):
        super(Simple_Gudi_UpConv_Block_Last_Layer, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.oheight = oheight
        self.owidth = owidth
        self._up_pool = Unpool(in_channels)

    def _up_pooling(self, x, scale):
        # Given x is [batch, channels, height, width],
        # _up_pool will be [batch, channels, height * 2, width * 2]
        x = self._up_pool(x)
        print(x.shape)
        if self.oheight != 0 and self.owidth != 0:
            x = x.narrow(2, 0, self.oheight)
            x = x.narrow(3, 0, self.owidth)
        return x

    def forward(self, x):
        x = self._up_pooling(x, 2)
        out = self.conv1(x)
        return out


if __name__ == '__main__':
    m = Affinity_Propagate(3, 3)
    # def forward(self, guidance, blur_depth, sparse_depth=None):
    g = torch.Tensor(4, 8, 30, 40)
    b = torch.Tensor(4, 16, 30, 40)  # blur_depth can be multi-channel
    print(g.shape, b.shape)
    r = m(g.to('cuda:0'), b.to('cuda:0'))
    print(r.shape)

    # x = torch.Tensor(4, 64, 30, 30)
    # t = Simple_Gudi_UpConv_Block_Last_Layer(64, 8, 30, 30).to('cuda:0')
    # y = t(x.to('cuda:0'))
    # print(y.shape)
