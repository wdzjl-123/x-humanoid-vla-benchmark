# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
"""
Backbone modules.
"""
from collections import OrderedDict

import torch
import torch.nn.functional as F
import torchvision
from torch import nn
from torchvision.models._utils import IntermediateLayerGetter
from typing import Dict, List
import numpy as np

import math
from torchvision.models import (
    efficientnet_b0,
    EfficientNet_B0_Weights,
    efficientnet_b3,
    EfficientNet_B3_Weights,
    EfficientNet_B5_Weights,
    resnet18,
    ResNet18_Weights,
    resnet34,
    ResNet34_Weights,
    resnet50,
    ResNet50_Weights,
)
from .efficientnet import (
    film_efficientnet_b0,
    film_efficientnet_b3,
    film_efficientnet_b5,
)
from .resnet import film_resnet18, film_resnet34, film_resnet50

from ..util.misc import NestedTensor, is_main_process

from .position_encoding import build_position_encoding


# import IPython
# e = IPython.embed

class FrozenBatchNorm2d(torch.nn.Module):
    """
    BatchNorm2d where the batch statistics and the affine parameters are fixed.

    Copy-paste from torchvision.misc.ops with added eps before rqsrt,
    without which any other policy_models than torchvision.policy_models.resnet[18,34,50,101]
    produce nans.
    """

    def __init__(self, n):
        super(FrozenBatchNorm2d, self).__init__()
        self.register_buffer("weight", torch.ones(n))
        self.register_buffer("bias", torch.zeros(n))
        self.register_buffer("running_mean", torch.zeros(n))
        self.register_buffer("running_var", torch.ones(n))

    def _load_from_state_dict(self, state_dict, prefix, local_metadata, strict,
                              missing_keys, unexpected_keys, error_msgs):
        num_batches_tracked_key = prefix + 'num_batches_tracked'
        if num_batches_tracked_key in state_dict:
            del state_dict[num_batches_tracked_key]

        super(FrozenBatchNorm2d, self)._load_from_state_dict(
            state_dict, prefix, local_metadata, strict,
            missing_keys, unexpected_keys, error_msgs)

    def forward(self, x):
        # move reshapes to the beginning
        # to make it fuser-friendly
        w = self.weight.reshape(1, -1, 1, 1)
        b = self.bias.reshape(1, -1, 1, 1)
        rv = self.running_var.reshape(1, -1, 1, 1)
        rm = self.running_mean.reshape(1, -1, 1, 1)
        eps = 1e-5
        scale = w * (rv + eps).rsqrt()
        bias = b - rm * scale
        return x * scale + bias


class BackboneBase(nn.Module):

    def __init__(self, backbone: nn.Module, train_backbone: bool, num_channels: int, return_interm_layers: bool):
        super().__init__()
        # for name, parameter in backbone.named_parameters(): # only train later layers # TODO do we want this?
        #     if not train_backbone or 'layer2' not in name and 'layer3' not in name and 'layer4' not in name:
        #         parameter.requires_grad_(False)
        if return_interm_layers:
            return_layers = {"layer1": "0", "layer2": "1", "layer3": "2", "layer4": "3"}
        else:
            return_layers = {'layer4': "0"}
        self.body = IntermediateLayerGetter(backbone, return_layers=return_layers)
        self.num_channels = num_channels

    def forward(self, tensor):
        xs = self.body(tensor)
        return xs
        # out: Dict[str, NestedTensor] = {}
        # for name, x in xs.items():
        #     m = tensor_list.mask
        #     assert m is not None
        #     mask = F.interpolate(m[None].float(), size=x.shape[-2:]).to(torch.bool)[0]
        #     out[name] = NestedTensor(x, mask)
        # return out


class Backbone(BackboneBase):
    """ResNet backbone with frozen BatchNorm."""
    def __init__(self, name: str,
                 train_backbone: bool,
                 return_interm_layers: bool,
                 dilation: bool):
        backbone = getattr(torchvision.models, name)(
            replace_stride_with_dilation=[False, False, dilation],
            pretrained=is_main_process(), norm_layer=FrozenBatchNorm2d) # pretrained # TODO do we want frozen batch_norm??

        num_channels = 512 if name in ('resnet18', 'resnet34') else 2048

        super().__init__(backbone, train_backbone, num_channels, return_interm_layers)


class Joiner(nn.Sequential):
    def __init__(self, backbone, position_embedding):
        super().__init__(backbone, position_embedding)

    def forward(self, tensor_list: NestedTensor):
        xs = self[0](tensor_list)
        out: List[NestedTensor] = []
        pos = []
        for name, x in xs.items():
            out.append(x)
            # position encoding
            pos.append(self[1](x).to(x.dtype))

        return out, pos

class FilMedBackbone(torch.nn.Module):
    """FiLMed image encoder backbone."""

    def __init__(self, name: str):
        super().__init__()
        # Load pretrained weights.
        if name == "resnet18film":
            weights = ResNet18_Weights.DEFAULT
            self.num_channels = 512
            self.backbone = film_resnet18(weights=weights, use_film=True)
        elif name == "resnet34film":
            weights = ResNet34_Weights.DEFAULT
            self.num_channels = 512
            self.backbone = film_resnet34(weights=weights, use_film=True)
        elif name == "resnet50film":
            weights = ResNet50_Weights.DEFAULT
            self.num_channels = 2048
            self.backbone = film_resnet50(weights=weights, use_film=True)
        elif name == "efficientnet_b0film":
            weights = EfficientNet_B0_Weights.DEFAULT
            self.num_channels = 1280
            self.backbone = film_efficientnet_b0(weights=weights, use_film=True)
        elif name == "efficientnet_b3film":
            weights = EfficientNet_B3_Weights.DEFAULT
            self.num_channels = 1536
            self.backbone = film_efficientnet_b3(weights=weights, use_film=True)
        elif name == "efficientnet_b5film":
            weights = EfficientNet_B5_Weights.DEFAULT
            self.num_channels = 2048
            self.backbone = film_efficientnet_b5(weights=weights, use_film=True)
        else:
            raise ValueError
        # Remove final average pooling and classification layers.
        if "resnet" in name:
            self.backbone.avgpool = nn.Sequential()  # remove average pool layer
            self.backbone.fc = nn.Sequential()  # remove classification layer
        else:  # efficientnet
            self.backbone.avgpool = nn.Sequential()  # remove average pool layer
            self.backbone.classifier = nn.Sequential()  # remove classification layer
        # Get image preprocessing function.
        self.preprocess = (
            weights.transforms()
        )  # Use this to preprocess images the same way as the pretrained model (e.g., ResNet-18).
        # self.preprocess = transforms.Compose([ # Use this if you don't want to resize images to 224x224.
        #     transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        # ])

    def forward(self, img_obs, language_embed):
        # img_obs shape: (batch_size, 3, H, W)
        img_obs = img_obs.float()  # cast to float type
        img_obs = self.preprocess(img_obs)
        out = self.backbone(
            img_obs, language_embed
        )  # shape (B, C_final * H_final * W_final) or (B, C_final, H_final, W_final)
        # If needed, unflatten output tensor from (B, C_final * H_final * W_final) to (B, C_final, H_final, W_final)
        if len(out.shape) == 2:
            H_final = W_final = int(math.sqrt(out.shape[-1] // self.num_channels))
            out = torch.unflatten(out, -1, (self.num_channels, H_final, W_final))
        # film backbone size: torch.Size([16, 2048, 15, 15])
        # print(f"film backbone size: {out.size()}")
        return out

class FiLMedJoiner(nn.Sequential):
    def __init__(self, backbone, position_embedding):
        super().__init__(backbone, position_embedding)

    def forward(self, img_obs, language_embed):
        """
        Args:
            img_obs: torch.Tensor of shape (batch_size, C, H, W)
            language_embed: torch.Tensor of shape (batch_size, language_embed_size)

        Returns:
            out: 1-length list of torch.Tensor of shape (batch_size, C_final, H_final, W_final).
            pos: 1-length list of torch.Tensor of shape (1, hidden_dim, H_final, W_final).
        """
        # self[0]: backbone, self[1]: position_embedding
        out = [self[0](img_obs, language_embed)]
        pos = [self[1](out[0]).to(out[0].dtype)]
        return out, pos

class ClipBackbone(torch.nn.Module):
    """FiLMed image encoder backbone."""

    def __init__(self, name: str):
        super().__init__()
        from PIL import Image
        import requests
        from transformers import CLIPProcessor, CLIPModel, CLIPVisionModel, CLIPImageProcessor
        self.name = name
        model_name = "/home/jz08/wk/github/hugging_face_model/clip-vit-base-patch16"
        self.model = CLIPVisionModel.from_pretrained(model_name)
        self.processor = CLIPImageProcessor.from_pretrained(model_name)
        self.num_channels = 768 #512
        print(f"processor: {self.processor}")
        print(f"model: {self.model}")

        # self.model = CLIPModel.from_pretrained("openai/clip-vit-base-patch16")
        # self.processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch16")

        # model_name = "openai/clip-vit-base-patch16"
        # model = CLIPModel.from_pretrained(model_name)
        # processor = CLIPProcessor.from_pretrained(model_name)

        # url = "http://images.cocodataset.org/val2017/000000039769.jpg"
        # image = Image.open(requests.get(url, stream=True).raw)
        # inputs = processor(text=["a photo of a cat", "a photo of a dog"], images=image, return_tensors="pt", padding=True)
        # outputs = model(**inputs)
        # logits_per_image = outputs.logits_per_image # this is the image-text similarity score
        # probs = logits_per_image.softmax(dim=1) # we can take the softmax to get the label probabilities

    def forward(self, img_obs):
        # [4, 3, 270, 480]
        # img_obs = img_obs * 255.0
        # print(f"Clip img_obs size: {img_obs.size()}")
        # img_obs = img_obs.cpu().numpy()
        # print(f"img_obs max: {np.max(img_obs)}")
        # print(f"img_obs min: {np.min(img_obs)}")
        # inputs = self.processor(images=img_obs, return_tensors="pt", padding=True, do_rescale=False)
        inputs = self.processor(images=img_obs, return_tensors="pt", padding=False)
        # print(f"Clip inputs size: {inputs}")
        # out = self.model(**inputs)
        # inputs = inputs.cuda()
        inputs = {key: val.to("cuda") for key, val in inputs.items()}
        # out = self.model.get_image_features(**inputs)
        # [4, 512]
        # print(f"Clip out size: {out.size()}")
        # print(f"inputs: {inputs.keys()}")
        out = self.model(**inputs, output_hidden_states=True)
        # print(f"out: {out.keys()}")
        # do not need class token
        out = out['last_hidden_state'][:, 1:]
        # out: odict_keys(['last_hidden_state', 'pooler_output', 'hidden_states'])
        # out: torch.Size([4, 196, 768])
        # print(f"out: {out.size()}")

        return out

class ClipJoiner(nn.Sequential):
    def __init__(self, backbone, position_embedding):
        super().__init__(backbone, position_embedding)

    def forward(self, img_obs):
        """
        Args:
            img_obs: torch.Tensor of shape (batch_size, C, H, W)
            language_embed: torch.Tensor of shape (batch_size, language_embed_size)

        Returns:
            out: 1-length list of torch.Tensor of shape (batch_size, C_final, H_final, W_final).
            pos: 1-length list of torch.Tensor of shape (1, hidden_dim, H_final, W_final).
        """
        # self[0]: backbone, self[1]: position_embedding
        out = [self[0](img_obs)]
        pos = [self[1](out[0]).to(out[0].dtype)]
        return out, pos

def build_backbone(args):
    position_embedding = build_position_encoding(args)
    train_backbone = args.lr_backbone > 0
    return_interm_layers = args.masks
    if "clip" in args.backbone:
        print("Using Clip backbone.")
        backbone = ClipBackbone(args.backbone)
        model = ClipJoiner(backbone, position_embedding)
    elif "film" in args.backbone:
        print("Using FiLMed backbone.")
        backbone = FilMedBackbone(args.backbone)
        model = FiLMedJoiner(backbone, position_embedding)
    else:
        backbone = Backbone(args.backbone, train_backbone, return_interm_layers, args.dilation)
        model = Joiner(backbone, position_embedding)
    model.num_channels = backbone.num_channels

    # backbone = Backbone(args.backbone, train_backbone, return_interm_layers, args.dilation)
    # model = Joiner(backbone, position_embedding)
    # model.num_channels = backbone.num_channels
    return model


class RestNetBasicBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride):
        super(RestNetBasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=stride, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        output = self.conv1(x)
        output = F.relu(self.bn1(output))
        output = self.conv2(output)
        output = self.bn2(output)
        return F.relu(x + output)


class RestNetDownBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride):
        super(RestNetDownBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride[0], padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=stride[1], padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.extra = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride[0], padding=0),
            nn.BatchNorm2d(out_channels)
        )

    def forward(self, x):
        extra_x = self.extra(x)
        output = self.conv1(x)
        out = F.relu(self.bn1(output))

        out = self.conv2(out)
        out = self.bn2(out)
        return F.relu(extra_x + out)


class DepthNet(nn.Module):
    def __init__(self):
        super(DepthNet, self).__init__()
        self.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3)
        # self.bn1 = nn.BatchNorm2d(64)
        # self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        # self.layer1 = nn.Sequential(RestNetBasicBlock(64, 64, 1),
        #                             RestNetBasicBlock(64, 64, 1))

        self.layer2 = nn.Sequential(RestNetDownBlock(64, 128, [4, 1]),
                                    RestNetBasicBlock(128, 128, 1))

        self.layer3 = nn.Sequential(RestNetDownBlock(128, 256, [4, 1]),
                                    RestNetBasicBlock(256, 256, 1))
        self.num_channels = 256
        # self.layer4 = nn.Sequential(RestNetDownBlock(256, 512, [2, 1]),
        #                             RestNetBasicBlock(512, 512, 1))

        # self.avgpool = nn.AdaptiveAvgPool2d(output_size=(1, 1))
        #
        # self.fc = nn.Linear(512, 10)

    def forward(self, x):
        out = self.conv1(x)
        # out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        # out = self.layer4(out)
        # out = self.avgpool(out)
        # out = out.reshape(x.shape[0], -1)
        # out = self.fc(out)
        return out
