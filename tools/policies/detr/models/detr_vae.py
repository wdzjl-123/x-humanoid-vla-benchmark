# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
"""
DETR model and criterion classes.
"""
import torch
from torch import nn
from torch.autograd import Variable
from .backbone import build_backbone, DepthNet
from .transformer import build_transformer, TransformerEncoder, TransformerEncoderLayer
import numpy as np
import torch.nn.functional as F

def reparametrize(mu, logvar):
    std = logvar.div(2).exp()
    eps = Variable(std.data.new(std.size()).normal_())
    return mu + std * eps


def get_sinusoid_encoding_table(n_position, d_hid):
    def get_position_angle_vec(position):
        return [position / np.power(10000, 2 * (hid_j // 2) / d_hid) for hid_j in range(d_hid)]

    sinusoid_table = np.array([get_position_angle_vec(pos_i) for pos_i in range(n_position)])
    sinusoid_table[:, 0::2] = np.sin(sinusoid_table[:, 0::2])  # dim 2i
    sinusoid_table[:, 1::2] = np.cos(sinusoid_table[:, 1::2])  # dim 2i+1

    return torch.FloatTensor(sinusoid_table).unsqueeze(0)


class DETRVAE(nn.Module):
    """ This is the DETR module that performs object detection """
    def __init__(self, backbones, depth_backbones, transformer, encoder, args_dict):
        """ Initializes the model.
        Parameters:
            backbones: torch module of the backbone to be used. See backbone.py
            transformer: torch module of the transformer architecture. See transformer.py
            state_dim: robot state dimension of the environment
            num_queries: number of object queries, ie detection slot. This is the maximal number of objects
                         DETR can detect in a single image. For COCO, we recommend 100 queries.
            aux_loss: True if auxiliary decoding losses (loss at each decoder layer) are to be used.
        """
        super().__init__()
        self.state_dim = args_dict['state_dim']
        self.num_queries = args_dict['num_queries']
        self.camera_names = args_dict['camera_names']
        self.kl_weight = args_dict['kl_weight']
        self.use_vq = args_dict['use_vq']
        self.vq_class = args_dict['vq_class']
        self.vq_dim = args_dict['vq_dim']
        self.state_dim = args_dict['state_dim']
        self.action_dim = args_dict['action_dim']
        self.input_state_acthead = args_dict['input_state_acthead']
        self.no_sepe_backbone = args_dict['no_sepe_backbone']
        self.use_lang = args_dict['use_lang']
        self.use_film = False
        if "film" in args_dict['backbone']:
            print(f"detr file is True!!!!")
            self.use_film = True

        self.transformer = transformer
        self.encoder = encoder
        hidden_dim = transformer.d_model

        self.backbone_name = args_dict['backbone']

        if self.input_state_acthead:
            # print(f"input_state_acthead is true!")
            last_input_dim = hidden_dim+self.state_dim
            self.action_head = nn.Linear(last_input_dim, self.action_dim)
            self.is_pad_head = nn.Linear(last_input_dim, 1)
        else:
            self.action_head = nn.Linear(hidden_dim, self.action_dim)
            self.is_pad_head = nn.Linear(hidden_dim, 1)


        # self.action_head = nn.Linear(hidden_dim, self.state_dim)
        self.query_embed = nn.Embedding(self.num_queries, hidden_dim)
        if backbones is not None:
            if "clip" in self.backbone_name:
                if depth_backbones is not None:
                    self.depth_backbones = nn.ModuleList(depth_backbones)
                    self.input_proj = nn.Linear(backbones[0].num_channels + depth_backbones[0].num_channels, hidden_dim)
                else:
                    self.depth_backbones = None
                    self.input_proj = nn.Linear(backbones[0].num_channels, hidden_dim)
            else:
                if depth_backbones is not None:
                    self.depth_backbones = nn.ModuleList(depth_backbones)
                    # RGB and depth features are concatenated before this
                    # projection, so the input channel count must include
                    # both encoders. The transformer still consumes hidden_dim.
                    self.input_proj = nn.Conv2d(
                        backbones[0].num_channels + depth_backbones[0].num_channels,
                        hidden_dim,
                        kernel_size=1,
                    )
                else:
                    self.depth_backbones = None
                    self.input_proj = nn.Conv2d(backbones[0].num_channels, hidden_dim, kernel_size=1)
            self.backbones = nn.ModuleList(backbones)
            self.input_proj_robot_state = nn.Linear(self.state_dim, hidden_dim)
            # language_distilbert: 768
        else:
            # input_dim = 14 + 7 # robot_state + env_state
            self.input_proj_robot_state = nn.Linear(self.state_dim, hidden_dim)
            self.input_proj_env_state = nn.Linear(7, hidden_dim)
            self.pos = torch.nn.Embedding(2, hidden_dim)
            self.backbones = None

        # encoder extra parameters
        self.latent_dim = 32  # final size of latent z # TODO tune
        self.cls_embed = nn.Embedding(1, hidden_dim)  # extra cls token embedding
        self.encoder_action_proj = nn.Linear(self.action_dim, hidden_dim) # project action to embedding
        self.encoder_joint_proj = nn.Linear(self.state_dim, hidden_dim)  # project qpos to embedding

        print(f'Use VQ: {self.use_vq}, {self.vq_class}, {self.vq_dim}')
        if self.use_vq:
            self.latent_proj = nn.Linear(hidden_dim, self.vq_class * self.vq_dim)
        else:
            self.latent_proj = nn.Linear(hidden_dim, self.latent_dim*2) # project hidden state to latent std, var
        self.register_buffer('pos_table', get_sinusoid_encoding_table(1+1+self.num_queries, hidden_dim)) # [CLS], qpos, a_seq

        # decoder extra parameters
        if self.use_vq:
            self.latent_out_proj = nn.Linear(self.vq_class * self.vq_dim, hidden_dim)
        else:
            self.latent_out_proj = nn.Linear(self.latent_dim, hidden_dim) # project latent sample to embedding
        # self.additional_pos_embed = nn.Embedding(2, hidden_dim) # learned position embedding for proprio and latent

        if self.use_lang:
            # print(f'use_lang is true!')
            self.input_proj_lang_embed = nn.Linear(768, hidden_dim)
            # one more for lang
            self.additional_pos_embed = nn.Embedding(3, hidden_dim)
        else:
            self.additional_pos_embed = nn.Embedding(2, hidden_dim)

    def encode(self, qpos, actions=None, is_pad=None, vq_sample=None):
        bs, _ = qpos.shape
        if self.encoder is None:
            latent_sample = torch.zeros([bs, self.latent_dim], dtype=torch.float32).to(qpos.device)
            latent_input = self.latent_out_proj(latent_sample)
            probs = binaries = mu = logvar = None
        else:
            # cvae encoder
            is_training = actions is not None # train or val
            ### Obtain latent z from action sequence
            if is_training:
                # project action sequence to embedding dim, and concat with a CLS token
                action_embed = self.encoder_action_proj(actions) # (bs, seq, hidden_dim)
                qpos_embed = self.encoder_joint_proj(qpos)  # (bs, hidden_dim)
                qpos_embed = torch.unsqueeze(qpos_embed, axis=1)  # (bs, 1, hidden_dim)
                cls_embed = self.cls_embed.weight # (1, hidden_dim)
                cls_embed = torch.unsqueeze(cls_embed, axis=0).repeat(bs, 1, 1) # (bs, 1, hidden_dim)
                encoder_input = torch.cat([cls_embed, qpos_embed, action_embed], axis=1) # (bs, seq+2, hidden_dim)
                encoder_input = encoder_input.permute(1, 0, 2) # (seq+2, bs, hidden_dim)
                # do not mask cls token
                cls_joint_is_pad = torch.full((bs, 2), False).to(qpos.device) # False: not a padding
                is_pad = torch.cat([cls_joint_is_pad, is_pad], axis=1)  # (bs, seq+2)
                # obtain position embedding
                pos_embed = self.pos_table.clone().detach()
                pos_embed = pos_embed.permute(1, 0, 2)  # (seq+2, 1, hidden_dim)
                # query model
                encoder_output = self.encoder(encoder_input, pos=pos_embed, src_key_padding_mask=is_pad)
                encoder_output = encoder_output[0] # take cls output only
                latent_info = self.latent_proj(encoder_output)

                if self.use_vq:
                    logits = latent_info.reshape([*latent_info.shape[:-1], self.vq_class, self.vq_dim])
                    probs = torch.softmax(logits, dim=-1)
                    binaries = F.one_hot(torch.multinomial(probs.view(-1, self.vq_dim), 1).squeeze(-1), self.vq_dim).view(-1, self.vq_class, self.vq_dim).float()
                    binaries_flat = binaries.view(-1, self.vq_class * self.vq_dim)
                    probs_flat = probs.view(-1, self.vq_class * self.vq_dim)
                    straigt_through = binaries_flat - probs_flat.detach() + probs_flat
                    latent_input = self.latent_out_proj(straigt_through)
                    mu = logvar = None
                else:
                    probs = binaries = None
                    mu = latent_info[:, :self.latent_dim]
                    logvar = latent_info[:, self.latent_dim:]
                    latent_sample = reparametrize(mu, logvar)
                    latent_input = self.latent_out_proj(latent_sample)

            else:
                mu = logvar = binaries = probs = None
                if self.use_vq:
                    latent_input = self.latent_out_proj(vq_sample.view(-1, self.vq_class * self.vq_dim))
                else:
                    latent_sample = torch.zeros([bs, self.latent_dim], dtype=torch.float32).to(qpos.device)
                    latent_input = self.latent_out_proj(latent_sample)

        return latent_input, probs, binaries, mu, logvar

    def forward(self, qpos, image, depth_image, env_state, actions=None, is_pad=None, vq_sample=None, lang_embed=None):
        """
        qpos: batch, qpos_dim
        image: batch, num_cam, channel, height, width
        env_state: None                                   没有值
        actions: batch, seq, action_dim
        """
        latent_input, probs, binaries, mu, logvar = self.encode(qpos, actions, is_pad, vq_sample)

        # Image observation features and position embeddings
        all_cam_features = []
        all_cam_depth_features = []
        all_cam_pos = []
        for cam_id, cam_name in enumerate(self.camera_names):
            # print(f'cam_name: {cam_name}')
            if self.no_sepe_backbone:
                print(f'no_sepe_backbone is True')
                if self.use_film and lang_embed is not None:
                    features, pos = self.backbones[0](image[:, cam_id], lang_embed)
                else:
                    features, pos = self.backbones[0](image[:, cam_id])
            else:
                if self.use_film and lang_embed is not None:
                    features, pos = self.backbones[cam_id](image[:, cam_id], lang_embed)
                else:
                    features, pos = self.backbones[cam_id](image[:, cam_id])

            features = features[0]  # take the last layer feature
            pos = pos[0]
            if self.no_sepe_backbone:
                if self.depth_backbones is not None and depth_image is not None:
                    features_depth = self.depth_backbones[0](depth_image[:, cam_id].unsqueeze(dim=1))
                    if features_depth.shape[-2:] != features.shape[-2:]:
                        features_depth = F.interpolate(
                            features_depth,
                            size=features.shape[-2:],
                            mode="bilinear",
                            align_corners=False,
                        )
                    all_cam_features.append(self.input_proj(torch.cat([features, features_depth], axis=1)))
                else:
                    all_cam_features.append(self.input_proj(features))
            else:
                if self.depth_backbones is not None and depth_image is not None:
                    features_depth = self.depth_backbones[cam_id](depth_image[:, cam_id].unsqueeze(dim=1))
                    if features_depth.shape[-2:] != features.shape[-2:]:
                        features_depth = F.interpolate(
                            features_depth,
                            size=features.shape[-2:],
                            mode="bilinear",
                            align_corners=False,
                        )
                    all_cam_features.append(self.input_proj(torch.cat([features, features_depth], axis=1)))
                else:
                    all_cam_features.append(self.input_proj(features))
            all_cam_pos.append(pos)

        if self.use_lang and lang_embed is not None:
            # print(f'use_lang is true!')
            # print(f'use_lang is true!, lang_embed size: {lang_embed.shape}')
            lang_embed = self.input_proj_lang_embed(lang_embed)
            # print(f'lang_embed: {lang_embed.shape}')
        # proprioception features
        proprio_input = self.input_proj_robot_state(qpos)
        # fold camera dimension into width dimension
        if "clip" in self.backbone_name:
            src = torch.cat(all_cam_features, axis=1)
            pos = torch.cat(all_cam_pos, axis=1)
        else:
            src = torch.cat(all_cam_features, axis=3)
            pos = torch.cat(all_cam_pos, axis=3)
            # image: (h, w) (480, 640)
            # efficientnet_b3film src size: torch.Size([2, 512, 10, 30])
            # efficientnet_b3film pos size: torch.Size([1, 512, 10, 30])
            # resnet18 src size: torch.Size([2, 512, 9, 36])
            # resnet18 pos size: torch.Size([1, 512, 9, 36])
            # print(f"src size: {src.size()}")
            # print(f"pos size: {pos.size()}")
        hs = self.transformer(src, None, self.query_embed.weight, pos, latent_input, proprio_input, self.additional_pos_embed.weight, lang_embed)[-1]

        # 1. hs: torch.Size([8, 100, 512])
        # print(f"1. hs: {hs.size()}")
        if self.input_state_acthead:
            qpos = qpos.unsqueeze(1).repeat(1, self.num_queries, 1)
            hs = torch.cat((hs, qpos), axis=-1)
            # print(f"2. hs: {hs.size()}")
        a_hat = self.action_head(hs)
        is_pad_hat = self.is_pad_head(hs)

        return a_hat, is_pad_hat, [mu, logvar], probs, binaries

class CNNMLP(nn.Module):
    def __init__(self, backbones, depth_backbones, state_dim, camera_names):
        """ Initializes the model.
        Parameters:
            backbones: torch module of the backbone to be used. See backbone.py
            transformer: torch module of the transformer architecture. See transformer.py
            state_dim: robot state dimension of the environment
            num_queries: number of object queries, ie detection slot. This is the maximal number of objects
                         DETR can detect in a single image. For COCO, we recommend 100 queries.
            aux_loss: True if auxiliary decoding losses (loss at each decoder layer) are to be used.
        """
        super().__init__()
        self.camera_names = camera_names
        self.depth_backbones = depth_backbones
        self.action_head = nn.Linear(1000, state_dim) # TODO add more

        if backbones is not None:
            self.backbones = nn.ModuleList(backbones)
            backbone_down_projs = []

            for i, backbone in enumerate(backbones):
                num_channels = backbone.num_channels
                if self.depth_backbones is not None:
                    num_channels += depth_backbones[i].num_channels
                down_proj = nn.Sequential(
                    nn.Conv2d(num_channels, 128, kernel_size=5),
                    nn.Conv2d(128, 64, kernel_size=5),
                    nn.Conv2d(64, 32, kernel_size=5)
                )
                backbone_down_projs.append(down_proj)
            self.backbone_down_projs = nn.ModuleList(backbone_down_projs)

            mlp_in_dim = 768 * len(backbones) + state_dim
            self.mlp = mlp(input_dim=mlp_in_dim, hidden_dim=1024, output_dim=state_dim, hidden_depth=2)
        else:
            raise NotImplementedError

    def forward(self, image, depth_image, robot_state, actions=None, action_is_pad=None):
        """
        qpos: batch, qpos_dim
        image: batch, num_cam, channel, height, width
        env_state: None
        actions: batch, seq, action_dim
        """
        bs, _ = robot_state.shape
        # Image observation features and position embeddings
        all_cam_features = []
        for cam_id, cam_name in enumerate(self.camera_names):
            features, pos = self.backbones[cam_id](image[:, cam_id])
            features = features[0]  # take the last layer feature
            if self.depth_backbones is not None and depth_image is not None:
                features_depth = self.depth_backbones[cam_id](depth_image[:, cam_id].unsqueeze(dim=1))
                all_cam_features.append(self.backbone_down_projs[cam_id](torch.cat([features, features_depth], axis=1)))
            else:
                all_cam_features.append(self.backbone_down_projs[cam_id](features))
        # flatten everything
        flattened_features = []
        for cam_feature in all_cam_features:
            flattened_features.append(cam_feature.reshape([bs, -1]))
        flattened_features = torch.cat(flattened_features, axis=1)  # 768 each
        features = torch.cat([flattened_features, robot_state], axis=1)  # qpos: 14
        a_hat = self.mlp(features)
        return a_hat


def mlp(input_dim, hidden_dim, output_dim, hidden_depth):
    if hidden_depth == 0:
        mods = [nn.Linear(input_dim, output_dim)]
    else:
        mods = [nn.Linear(input_dim, hidden_dim), nn.ReLU(inplace=True)]
        for i in range(hidden_depth - 1):
            mods += [nn.Linear(hidden_dim, hidden_dim), nn.ReLU(inplace=True)]
        mods.append(nn.Linear(hidden_dim, output_dim))
    trunk = nn.Sequential(*mods)
    return trunk


def build_encoder(args):

    d_model = args.hidden_dim  # 256
    dropout = args.dropout     # 0.1
    nhead = args.nheads        # 8
    dim_feedforward = args.dim_feedforward  # 2048
    num_encoder_layers = args.enc_layers  # 4 # TODO shared with VAE decoder
    normalize_before = args.pre_norm  # False
    activation = "relu"

    encoder_layer = TransformerEncoderLayer(d_model, nhead, dim_feedforward,
                                            dropout, activation, normalize_before)

    encoder_norm = nn.LayerNorm(d_model) if normalize_before else None

    encoder = TransformerEncoder(encoder_layer, num_encoder_layers, encoder_norm)

    return encoder


def build(args):
    state_dim = args.state_dim
    # print('state_dim:',state_dim)
    # From state
    # backbone = None # from state for now, no need for conv nets
    # From image
    backbones = []
    depth_backbones = None
    if args.use_depth_image:
        depth_backbones = []

    if args.no_sepe_backbone:
        backbone = build_backbone(args)
        backbones.append(backbone)
        if args.use_depth_image:
            depth_backbones.append(DepthNet())
    else:
        for _ in args.camera_names:
            backbone = build_backbone(args)
            backbones.append(backbone)
            if args.use_depth_image:
                depth_backbones.append(DepthNet())

    transformer = build_transformer(args)  # 构建trans层

    # encoder = None
    # if args.kl_weight != 0:
    #     encoder = build_encoder(args)          # 构建编码成和解码层

    if args.no_encoder:
        encoder = None
    else:
        encoder = build_encoder(args)

    args_dict = vars(args)
    model = DETRVAE(
        backbones,
        depth_backbones,
        transformer,
        encoder,
        args_dict,
    )

    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print("number of parameters: %.2fM" % (n_parameters/1e6,))

    return model


def build_cnnmlp(args):
    if args.use_robot_base:
        state_dim = 16  # TODO hardcode
    else:
        state_dim = 14

    # From state
    # backbone = None # from state for now, no need for conv nets
    # From image
    backbones = []   # 空的网络list
    depth_backbones = None
    if args.use_depth_image:
        depth_backbones = []

    # backbone = build_backbone(args)  # 位置编码和主干网络组合成特征提取器
    # backbones.append(backbone)
    # if args.use_depth_image:
    #     depth_backbones.append(DepthNet())

    for _ in args.camera_names:
        backbone = build_backbone(args)
        backbones.append(backbone)
        if args.use_depth_image:
            depth_backbones.append(DepthNet())

    model = CNNMLP(
        backbones,
        depth_backbones,
        state_dim=state_dim,
        camera_names=args.camera_names,
    )

    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print("number of parameters: %.2fM" % (n_parameters/1e6,))

    return model
