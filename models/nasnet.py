import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from collections import OrderedDict


class SeparableConv2d(nn.Sequential):

    def __init__(self, in_channels, out_channels, dw_kernel, dw_stride, dw_padding, bias=False):
        super(SeparableConv2d, self).__init__()
        self.depthwise_conv2d = nn.Conv2d(in_channels, in_channels, dw_kernel,
                                          stride=dw_stride,
                                          padding=dw_padding,
                                          bias=bias,
                                          groups=in_channels)
        self.pointwise_conv2d = nn.Conv2d(in_channels, out_channels, 1, stride=1, bias=bias)


class BranchSeparables(nn.Sequential):

    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, bias=False):
        super(BranchSeparables, self).__init__()
        self.relu = nn.ReLU()
        self.separable_1 = SeparableConv2d(in_channels, in_channels, kernel_size, stride, padding, bias=bias)
        self.bn_sep_1 = nn.BatchNorm2d(in_channels, eps=0.001, momentum=0.1, affine=True)
        self.relu1 = nn.ReLU()
        self.separable_2 = SeparableConv2d(in_channels, out_channels, kernel_size, 1, padding, bias=bias)
        self.bn_sep_2 = nn.BatchNorm2d(out_channels, eps=0.001, momentum=0.1, affine=True)


class BranchSeparablesStem(nn.Sequential):

    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, bias=False):
        super(BranchSeparablesStem, self).__init__()
        self.relu = nn.ReLU()
        self.separable_1 = SeparableConv2d(in_channels, out_channels, kernel_size, stride, padding, bias=bias)
        self.bn_sep_1 = nn.BatchNorm2d(out_channels, eps=0.001, momentum=0.1, affine=True)
        self.relu1 = nn.ReLU()
        self.separable_2 = SeparableConv2d(out_channels, out_channels, kernel_size, 1, padding, bias=bias)
        self.bn_sep_2 = nn.BatchNorm2d(out_channels, eps=0.001, momentum=0.1, affine=True)


def ReductionCellBranchCombine(cell, x_left, x_right):
    x_comb_iter_0_left = cell.comb_iter_0_left(x_left)
    x_comb_iter_0_right = cell.comb_iter_0_right(x_right)
    x_comb_iter_0 = x_comb_iter_0_left + x_comb_iter_0_right

    x_comb_iter_1_left = cell.comb_iter_1_left(x_left)
    x_comb_iter_1_right = cell.comb_iter_1_right(x_right)
    x_comb_iter_1 = x_comb_iter_1_left + x_comb_iter_1_right

    x_comb_iter_2_left = cell.comb_iter_2_left(x_left)
    x_comb_iter_2_right = cell.comb_iter_2_right(x_right)
    x_comb_iter_2 = x_comb_iter_2_left + x_comb_iter_2_right

    x_comb_iter_3_right = cell.comb_iter_3_right(x_comb_iter_0)
    x_comb_iter_3 = x_comb_iter_3_right + x_comb_iter_1

    x_comb_iter_4_left = cell.comb_iter_4_left(x_comb_iter_0)
    x_comb_iter_4_right = cell.comb_iter_4_right(x_left)
    x_comb_iter_4 = x_comb_iter_4_left + x_comb_iter_4_right

    x_out = torch.cat([x_comb_iter_1, x_comb_iter_2, x_comb_iter_3, x_comb_iter_4], 1)
    return x_out


class CellStem0(nn.Module):

    def __init__(self, in_channels, out_channels):
        super(CellStem0, self).__init__()
        self.conv_1x1 = nn.Sequential()
        self.conv_1x1.add_module('relu', nn.ReLU())
        self.conv_1x1.add_module('conv', nn.Conv2d(in_channels, out_channels, 1, stride=1, bias=False))
        self.conv_1x1.add_module('bn', nn.BatchNorm2d(out_channels, eps=0.001, momentum=0.1, affine=True))

        self.comb_iter_0_left = BranchSeparables(out_channels, out_channels, 5, 2, 2, bias=False)
        self.comb_iter_0_right = BranchSeparablesStem(in_channels, out_channels, 7, 2, 3, bias=False)

        self.comb_iter_1_left = nn.MaxPool2d(3, stride=2, padding=1)
        self.comb_iter_1_right = BranchSeparablesStem(in_channels, out_channels, 7, 2, 3, bias=False)

        self.comb_iter_2_left = nn.AvgPool2d(3, stride=2, padding=1, count_include_pad=False)
        self.comb_iter_2_right = BranchSeparablesStem(in_channels, out_channels, 5, 2, 2, bias=False)

        self.comb_iter_3_right = nn.AvgPool2d(3, stride=1, padding=1, count_include_pad=False)

        self.comb_iter_4_left = BranchSeparables(out_channels, out_channels, 3, 1, 1, bias=False)
        self.comb_iter_4_right = nn.MaxPool2d(3, stride=2, padding=1)

    def forward(self, x):
        x1 = self.conv_1x1(x)

        return ReductionCellBranchCombine(self, x1, x)


class CellStem1(nn.Module):

    def __init__(self, in_channels_x, in_channels_h, out_channels):
        super(CellStem1, self).__init__()
        self.conv_1x1 = nn.Sequential()
        self.conv_1x1.add_module('relu', nn.ReLU())
        self.conv_1x1.add_module('conv', nn.Conv2d(in_channels_x, out_channels, 1, stride=1, bias=False))
        self.conv_1x1.add_module('bn', nn.BatchNorm2d(out_channels, eps=0.001, momentum=0.1, affine=True))

        self.relu = nn.ReLU()
        self.path_1 = nn.Sequential()
        self.path_1.add_module('avgpool', nn.AvgPool2d(1, stride=2, count_include_pad=False))
        self.path_1.add_module('conv', nn.Conv2d(in_channels_h, out_channels // 2, 1, stride=1, bias=False))
        self.path_2 = nn.Sequential()
        self.path_2.add_module('avgpool', nn.AvgPool2d(1, stride=2, ceil_mode=True,
                                                       count_include_pad=False))  # ceil mode for padding
        self.path_2.add_module('conv', nn.Conv2d(in_channels_h, out_channels // 2, 1, stride=1, bias=False))

        self.final_path_bn = nn.BatchNorm2d(out_channels, eps=0.001, momentum=0.1, affine=True)

        self.comb_iter_0_left = BranchSeparables(out_channels, out_channels, 5, 2, 2, bias=False)
        self.comb_iter_0_right = BranchSeparables(out_channels, out_channels, 7, 2, 3, bias=False)

        self.comb_iter_1_left = nn.MaxPool2d(3, stride=2, padding=1)
        self.comb_iter_1_right = BranchSeparables(out_channels, out_channels, 7, 2, 3, bias=False)

        self.comb_iter_2_left = nn.AvgPool2d(3, stride=2, padding=1, count_include_pad=False)
        self.comb_iter_2_right = BranchSeparables(out_channels, out_channels, 5, 2, 2, bias=False)

        self.comb_iter_3_right = nn.AvgPool2d(3, stride=1, padding=1, count_include_pad=False)

        self.comb_iter_4_left = BranchSeparables(out_channels, out_channels, 3, 1, 1, bias=False)
        self.comb_iter_4_right = nn.MaxPool2d(3, stride=2, padding=1)

    def forward(self, x_conv0, x_stem_0):
        x_left = self.conv_1x1(x_stem_0)

        x_relu = self.relu(x_conv0)
        # path 1
        x_path1 = self.path_1(x_relu)
        # path 2
        x_path2 = self.path_2(x_relu[:, :, 1:, 1:])
        # final path
        x_right = self.final_path_bn(torch.cat([x_path1, x_path2], 1))

        return ReductionCellBranchCombine(self, x_left, x_right)


class ReductionCell(nn.Module):

    def __init__(self, in_channels_left, out_channels_left, in_channels_right, out_channels_right):
        super(ReductionCell, self).__init__()
        self.conv_prev_1x1 = nn.Sequential()
        self.conv_prev_1x1.add_module('relu', nn.ReLU())
        self.conv_prev_1x1.add_module('conv', nn.Conv2d(in_channels_left, out_channels_left, 1, stride=1, bias=False))
        self.conv_prev_1x1.add_module('bn', nn.BatchNorm2d(out_channels_left, eps=0.001, momentum=0.1, affine=True))

        self.conv_1x1 = nn.Sequential()
        self.conv_1x1.add_module('relu', nn.ReLU())
        self.conv_1x1.add_module('conv', nn.Conv2d(in_channels_right, out_channels_right, 1, stride=1, bias=False))
        self.conv_1x1.add_module('bn', nn.BatchNorm2d(out_channels_right, eps=0.001, momentum=0.1, affine=True))

        self.comb_iter_0_left = BranchSeparables(out_channels_right, out_channels_right, 5, 2, 2, bias=False)
        self.comb_iter_0_right = BranchSeparables(out_channels_right, out_channels_right, 7, 2, 3, bias=False)

        self.comb_iter_1_left = nn.MaxPool2d(3, stride=2, padding=1)
        self.comb_iter_1_right = BranchSeparables(out_channels_right, out_channels_right, 7, 2, 3, bias=False)

        self.comb_iter_2_left = nn.AvgPool2d(3, stride=2, padding=1, count_include_pad=False)
        self.comb_iter_2_right = BranchSeparables(out_channels_right, out_channels_right, 5, 2, 2, bias=False)

        self.comb_iter_3_right = nn.AvgPool2d(3, stride=1, padding=1, count_include_pad=False)

        self.comb_iter_4_left = BranchSeparables(out_channels_right, out_channels_right, 3, 1, 1, bias=False)
        self.comb_iter_4_right = nn.MaxPool2d(3, stride=2, padding=1)

    def forward(self, x, x_prev):
        x_left = self.conv_1x1(x)
        x_right = self.conv_prev_1x1(x_prev)
        return ReductionCellBranchCombine(self, x_left, x_right)


def NormalCellBranchCombine(cell, x_left, x_right):
    x_comb_iter_0_left = cell.comb_iter_0_left(x_right)
    x_comb_iter_0_right = cell.comb_iter_0_right(x_left)
    x_comb_iter_0 = x_comb_iter_0_left + x_comb_iter_0_right

    x_comb_iter_1_left = cell.comb_iter_1_left(x_left)
    x_comb_iter_1_right = cell.comb_iter_1_right(x_left)
    x_comb_iter_1 = x_comb_iter_1_left + x_comb_iter_1_right

    x_comb_iter_2_left = cell.comb_iter_2_left(x_right)
    x_comb_iter_2 = x_comb_iter_2_left + x_left

    x_comb_iter_3_left = cell.comb_iter_3_left(x_left)
    x_comb_iter_3_right = cell.comb_iter_3_right(x_left)
    x_comb_iter_3 = x_comb_iter_3_left + x_comb_iter_3_right

    x_comb_iter_4_left = cell.comb_iter_4_left(x_right)
    x_comb_iter_4 = x_comb_iter_4_left + x_right

    x_out = torch.cat([x_left, x_comb_iter_0, x_comb_iter_1, x_comb_iter_2, x_comb_iter_3, x_comb_iter_4], 1)
    return x_out


class FirstCell(nn.Module):

    def __init__(self, in_channels_left, out_channels_left, in_channels_right, out_channels_right):
        super(FirstCell, self).__init__()
        self.conv_1x1 = nn.Sequential()
        self.conv_1x1.add_module('relu', nn.ReLU())
        self.conv_1x1.add_module('conv', nn.Conv2d(in_channels_right, out_channels_right, 1, stride=1, bias=False))
        self.conv_1x1.add_module('bn', nn.BatchNorm2d(out_channels_right, eps=0.001, momentum=0.1, affine=True))

        self.relu = nn.ReLU()
        self.path_1 = nn.Sequential()
        self.path_1.add_module('avgpool', nn.AvgPool2d(1, stride=2, count_include_pad=False))
        self.path_1.add_module('conv', nn.Conv2d(in_channels_left, out_channels_left, 1, stride=1, bias=False))
        self.path_2 = nn.Sequential()
        self.path_2.add_module('avgpool', nn.AvgPool2d(1, stride=2, ceil_mode=True, count_include_pad=False))
        self.path_2.add_module('conv', nn.Conv2d(in_channels_left, out_channels_left, 1, stride=1, bias=False))

        self.final_path_bn = nn.BatchNorm2d(out_channels_left * 2, eps=0.001, momentum=0.1, affine=True)

        self.comb_iter_0_left = BranchSeparables(out_channels_right, out_channels_right, 5, 1, 2, bias=False)
        self.comb_iter_0_right = BranchSeparables(out_channels_right, out_channels_right, 3, 1, 1, bias=False)

        self.comb_iter_1_left = BranchSeparables(out_channels_right, out_channels_right, 5, 1, 2, bias=False)
        self.comb_iter_1_right = BranchSeparables(out_channels_right, out_channels_right, 3, 1, 1, bias=False)

        self.comb_iter_2_left = nn.AvgPool2d(3, stride=1, padding=1, count_include_pad=False)

        self.comb_iter_3_left = nn.AvgPool2d(3, stride=1, padding=1, count_include_pad=False)
        self.comb_iter_3_right = nn.AvgPool2d(3, stride=1, padding=1, count_include_pad=False)

        self.comb_iter_4_left = BranchSeparables(out_channels_right, out_channels_right, 3, 1, 1, bias=False)

    def forward(self, x, x_prev):
        x_relu = self.relu(x_prev)
        # path 1
        x_path1 = self.path_1(x_relu)
        # path 2
        x_path2 = self.path_2(x_relu[:, :, 1:, 1:])
        # final path
        x_left = self.final_path_bn(torch.cat([x_path1, x_path2], 1))

        x_right = self.conv_1x1(x)

        return NormalCellBranchCombine(self, x_left, x_right)


class NormalCell(nn.Module):

    def __init__(self, in_channels_left, out_channels_left, in_channels_right, out_channels_right):
        super(NormalCell, self).__init__()
        self.conv_prev_1x1 = nn.Sequential()
        self.conv_prev_1x1.add_module('relu', nn.ReLU())
        self.conv_prev_1x1.add_module('conv', nn.Conv2d(in_channels_left, out_channels_left, 1, stride=1, bias=False))
        self.conv_prev_1x1.add_module('bn', nn.BatchNorm2d(out_channels_left, eps=0.001, momentum=0.1, affine=True))

        self.conv_1x1 = nn.Sequential()
        self.conv_1x1.add_module('relu', nn.ReLU())
        self.conv_1x1.add_module('conv', nn.Conv2d(in_channels_right, out_channels_right, 1, stride=1, bias=False))
        self.conv_1x1.add_module('bn', nn.BatchNorm2d(out_channels_right, eps=0.001, momentum=0.1, affine=True))

        self.comb_iter_0_left = BranchSeparables(out_channels_right, out_channels_right, 5, 1, 2, bias=False)
        self.comb_iter_0_right = BranchSeparables(out_channels_left, out_channels_left, 3, 1, 1, bias=False)

        self.comb_iter_1_left = BranchSeparables(out_channels_left, out_channels_left, 5, 1, 2, bias=False)
        self.comb_iter_1_right = BranchSeparables(out_channels_left, out_channels_left, 3, 1, 1, bias=False)

        self.comb_iter_2_left = nn.AvgPool2d(3, stride=1, padding=1, count_include_pad=False)

        self.comb_iter_3_left = nn.AvgPool2d(3, stride=1, padding=1, count_include_pad=False)
        self.comb_iter_3_right = nn.AvgPool2d(3, stride=1, padding=1, count_include_pad=False)

        self.comb_iter_4_left = BranchSeparables(out_channels_right, out_channels_right, 3, 1, 1, bias=False)

    def forward(self, x, x_prev):
        x_left = self.conv_prev_1x1(x_prev)
        x_right = self.conv_1x1(x)

        return NormalCellBranchCombine(self, x_left, x_right)


class NASNet(nn.Module):
    def __init__(self, num_stem_features, num_normal_cells, filters, scaling, skip_reduction, use_aux=True,
                 num_classes=1000):
        super(NASNet, self).__init__()
        self.num_normal_cells = num_normal_cells
        self.skip_reduction = skip_reduction
        self.use_aux = use_aux
        self.num_classes = num_classes

        self.conv0 = nn.Sequential(OrderedDict([
            ('conv', nn.Conv2d(3, num_stem_features, kernel_size=3, stride=2, bias=False)),
            ('bn', nn.BatchNorm2d(num_stem_features, eps=0.001, momentum=0.1, affine=True))
        ]))

        self.cell_stem_0 = CellStem0(in_channels=num_stem_features,
                                     out_channels=int(filters * scaling ** (-2)))
        self.cell_stem_1 = CellStem1(in_channels_x=int(4 * filters * scaling ** (-2)),
                                     in_channels_h=num_stem_features,
                                     out_channels=int(filters * scaling ** (-1)))

        x_channels = int(4 * filters * scaling ** (-1))
        h_channels = int(4 * filters * scaling ** (-2))
        cell_id = 0
        branch_out_channels = filters
        for i in range(3):
            self.add_module('cell_{:d}'.format(cell_id), FirstCell(
                in_channels_left=h_channels, out_channels_left=branch_out_channels // 2, in_channels_right=x_channels,
                out_channels_right=branch_out_channels))
            cell_id += 1
            h_channels = x_channels
            x_channels = 6 * branch_out_channels  # normal: concat 6 branches
            for _ in range(num_normal_cells - 1):
                self.add_module('cell_{:d}'.format(cell_id), NormalCell(
                    in_channels_left=h_channels, out_channels_left=branch_out_channels, in_channels_right=x_channels,
                    out_channels_right=branch_out_channels))
                h_channels = x_channels
                cell_id += 1
            if i == 1 and self.use_aux:
                self.aux_features = nn.Sequential(
                    nn.ReLU(),
                    nn.AvgPool2d(kernel_size=(5, 5), stride=(3, 3),
                                 padding=(2, 2), count_include_pad=False),
                    nn.Conv2d(in_channels=x_channels, out_channels=128, kernel_size=1, bias=False),
                    nn.BatchNorm2d(num_features=128, eps=0.001, momentum=0.1, affine=True),
                    nn.ReLU(),
                    nn.Conv2d(in_channels=128, out_channels=768,
                              kernel_size=((14 + 2) // 3, (14 + 2) // 3), bias=False),
                    nn.BatchNorm2d(num_features=768, eps=1e-3, momentum=0.1, affine=True),
                    nn.ReLU()
                )
                self.aux_linear = nn.Linear(768, num_classes)
            # scaling
            branch_out_channels *= scaling
            if i < 2:
                self.add_module('reduction_cell_{:d}'.format(i), ReductionCell(
                    in_channels_left=h_channels, out_channels_left=branch_out_channels,
                    in_channels_right=x_channels, out_channels_right=branch_out_channels))
                x_channels = 4 * branch_out_channels  # reduce: concat 4 branches

        self.linear = nn.Linear(x_channels, self.num_classes)  # large: 4032; mobile: 1056

        self.num_params = sum([param.numel() for param in self.parameters()])
        if self.use_aux:
            self.num_params -= sum([param.numel() for param in self.aux_features.parameters()])
            self.num_params -= sum([param.numel() for param in self.aux_linear.parameters()])

    def features(self, x):
        x_conv0 = self.conv0(x)
        x_stem_0 = self.cell_stem_0(x_conv0)
        x_stem_1 = self.cell_stem_1(x_conv0, x_stem_0)
        prev_x, x = x_stem_0, x_stem_1
        cell_id = 0
        for i in range(3):
            for _ in range(self.num_normal_cells):
                new_x = self._modules['cell_{:d}'.format(cell_id)](x, prev_x)
                prev_x, x = x, new_x
                cell_id += 1
            if i == 1 and self.training and self.use_aux:
                x_aux = self.aux_features(x)
            if i < 2:
                new_x = self._modules['reduction_cell_{:d}'.format(i)](x, prev_x)
                prev_x = x if not self.skip_reduction else prev_x
                x = new_x
        if self.training and self.use_aux:
            return [x, x_aux]
        return [x]

    def logits(self, features):
        x = F.relu(features, inplace=False)
        x = F.adaptive_avg_pool2d(x, 1).view(x.size(0), -1)
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.linear(x)
        return x

    def forward(self, x):
        x = self.features(x)
        output = self.logits(x[0])
        if self.training and self.use_aux:
            x_aux = x[1].view(x[1].size(0), -1)
            aux_output = self.aux_linear(x_aux)
            return [output, aux_output]
        return [output]


def NASNetALarge(num_classes=1000):
    return NASNet(96, 6, 168, 2, skip_reduction=True, use_aux=True, num_classes=num_classes)


if __name__ == '__main__':
    from rfa_toolbox import input_resolution_range, create_graph_from_pytorch_model, visualize_architecture
    model = NASNetALarge()
    model.eval()
    res = 315
    for _ in range(30):
        try:
            input = torch.zeros((1, 3, res, res))
            model(input)

        except RuntimeError:
            print(res, "did not work")
            res += 1
    print("Final res is:", res)
    graph = create_graph_from_pytorch_model(model, input_res=(1, 3, res, res))
    print(input_resolution_range(graph))
    visualize_architecture(graph, "NASNET", input_res=331).view()