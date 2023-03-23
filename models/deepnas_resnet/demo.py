# Copyright (c) Alibaba, Inc. and its affiliates.

import ast
import os
import argparse
import torch
from timm.models import register_model

from cnnnet import CnnNet


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--structure_txt', type=str)
    parser.add_argument('--pretrained', type=str, default=None)
    args = parser.parse_args()
    return args


def resnet18_50000(
        pretrained=False,
        network_id=0,
        classification=True,  # False for detetion
        num_classes=1000,
        **kwargs

):
    cfg = {
        "best_structures": [
            [
                {
                    "class": "ConvKXBNRELU",
                    "in": 3,
                    "k": 3,
                    "out": 24,
                    "s": 2
                },
                {
                    "L": 1,
                    "btn": 16,
                    "class": "SuperResK1KXK1",
                    "in": 24,
                    "inner_class": "ResK1KXK1",
                    "k": 5,
                    "out": 88,
                    "s": 2
                },
                {
                    "L": 7,
                    "btn": 32,
                    "class": "SuperResK1KXK1",
                    "in": 88,
                    "inner_class": "ResK1KXK1",
                    "k": 5,
                    "out": 360,
                    "s": 2
                },
                {
                    "L": 12,
                    "btn": 64,
                    "class": "SuperResK1KXK1",
                    "in": 360,
                    "inner_class": "ResK1KXK1",
                    "k": 5,
                    "out": 656,
                    "s": 2
                },
                {
                    "L": 12,
                    "btn": 72,
                    "class": "SuperResK1KXK1",
                    "in": 656,
                    "inner_class": "ResK1KXK1",
                    "k": 5,
                    "out": 736,
                    "s": 1
                },
                {
                    "L": 1,
                    "btn": 512,
                    "class": "SuperResK1KXK1",
                    "in": 736,
                    "inner_class": "ResK1KXK1",
                    "k": 3,
                    "out": 2048,
                    "s": 2
                }
            ]
        ],
        "space_arch": "CnnNet"
    }
    # load best structures

    network_arch = cfg['space_arch']
    best_structures = cfg['best_structures']

    # If task type is classification, param num_classes is required
    out_indices = (1, 2, 3, 4) if not classification else (4,)
    backbone = CnnNet(
        structure_info=best_structures[network_id],
        out_indices=out_indices,
        num_classes=num_classes,
        classfication=classification)
    backbone.init_weights(pretrained)

    return backbone


@register_model
def better_resnet18like(*, progress: bool = True, **kwargs):
    return resnet18_50000(**kwargs)


if __name__ == '__main__':
    # make input
    x = torch.randn(1, 3, 224, 224)

    # instantiation
    backbone = better_resnet18like(num_classes=42)

    print(backbone)
    # forward
    input_data = [x]
    backbone.eval()
    pred = backbone(*input_data)

    # print output
    for o in pred:
        print(o.size())

    from rfa_toolbox import create_graph_from_pytorch_model, input_resolution_range, visualize_architecture

    model = backbone
    graph = create_graph_from_pytorch_model(model, input_res=(1, 3, 224, 224))
    # set filter_all_inf_rf to True, if your model contains Squeeze-And-Excitation-Modules
    min_res, max_res = input_resolution_range(graph, filter_all_inf_rf=False)  # (75, 75), (427, 427)
    print("Mininum Resolution:", min_res, "Maximum Resolution:", max_res)
    visualize_architecture(graph, "VGG16", input_res=224).view()
