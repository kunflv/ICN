# ------------------------------------------------------------------------
# INTR
# Copyright (c) 2023 Imageomics Paul. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
# Copied from DETR (https://github.com/facebookresearch/detr)
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
# ------------------------------------------------------------------------

"""
INTR model and loss.
"""
import torch
from torch import nn
import torch.nn.functional as F

import random
from .backbone_intr import build_backbone
from .k_transformer import build_transformer
from util.misc import (NestedTensor, nested_tensor_from_tensor_list,
                       )

import copy
from torch.nn.functional import normalize
from use_module.kan import KAN

def orthogonal_query_init(x):
    """
        计算正交矩阵 A
        返回:
        A (torch.Tensor): 正交矩阵
    """
    I = torch.eye(x.shape[-1], device=x.device)
    A = copy.deepcopy(I)
    x = normalize(x, p=2.0, dim=-1)
    # for i in range(self.feat_channels):
    for v in x:
        A = A @ (I - 2 * torch.outer(v, v))
    return A[:x.shape[0]]

class INTR(nn.Module):
    """ This is the INTR module that performs explainable image classification """
    def __init__(self, args, backbone, transformer, num_queries, num_feature_levels=4):
        """ Initializes the model.
        Parameters:
            backbone: torch module of the backbone to be used. See backbone_intr.py
            transformer: torch module of the transformer architecture. See k_transformer.py (no pos_embed in decoder)
            num_queries: number of classes in the dataset
        """
        super().__init__()
        self.args = args
        self.num_queries = num_queries
        self.transformer = transformer
        hidden_dim = transformer.d_model

        # INTR classification head presence vector
        # self.presence_vector = nn.Linear(hidden_dim, 1)
        self.kan_cls = KAN([hidden_dim, 1])

        self.query_embed = nn.Embedding(num_queries, hidden_dim)
        self.input_proj = nn.Conv2d(backbone.num_channels, hidden_dim, kernel_size=1)
        self.backbone = backbone

        self.num_feature_levels = num_feature_levels

        if self.num_queries < hidden_dim:
            # 创建可学习的正交初始化query_feat
            self.query_embed.weight = nn.Parameter(orthogonal_query_init(self.query_embed.weight))
        # self.norm = nn.LayerNorm(hidden_dim)

    def forward(self, samples: NestedTensor):

        """  The forward expects a NestedTensor, which consists of:
               - samples.tensor: batched images, of shape [batch_size x 3 x H x W]

            It returns the following elements:
               - "out": it is a dictnary which currently contains all logit values for for all queries.
                                Shape= [batch_size x num_queries x 1]
               - "encoder_output": it is the output of the transformer encoder which is basically feature map. 
                                Shape= [batch_size x num_features x height x weight]
               - "hs": it is the output of the transformer decoder. These are learned class specific queries. 
                                Shape= [dec_layers x batch_size x num_queries x num_features]
               - "attention_scores": it is attention weight corresponding to each pixel in the encoder  for all heads. 
                                Shape= [dec_layers x batch_size x num_heads x num_queries x height*weight]
               - "avg_attention_scores": it is attention weight corresponding to each pixel in the encoder for avg of all heads. 
                                Shape= [dec_layers x batch_size x num_queries x height*weight]

        """
        if isinstance(samples, (list, torch.Tensor)):
            samples = nested_tensor_from_tensor_list(samples)
        features, pos = self.backbone(samples)

        src, mask = features[-1].decompose()

        query = self.query_embed.weight  # (num_queries, hidden_dim)
        # print(query[0,:10])

        assert mask is not None
        hs, encoder_output, attention_scores, avg_attention_scores = self.transformer(self.input_proj(src), mask, query, pos[-1])

        # query = hs[-1]

        """FFN_head"""
        # query_logits = self.presence_vector(hs[-1])  # Correspond to Equation (6)
        """KAN_pred_head"""
        b, q, c = hs[-1].shape
        kan_input = hs[-1].reshape(-1, c)  # 等价 decoder_out.reshape(-1, c)
        # kan_input = normalize(kan_input, p=2, dim=-1)
        # kan_input = self.norm(kan_input)
        query_logits = self.kan_cls(kan_input).view(b, q, -1)

        out = {'query_logits': query_logits.squeeze(dim=-1)}


        return out, query, encoder_output, hs, attention_scores, avg_attention_scores

class SetCriterion(nn.Module):
    """ This class computes the loss for INTR.
        INTR uses only one type of loss i.e., cross entropy loss.
    """
    def __init__(self, args,  model): # weight_dict, losses,
        """ Create the criterion.
        """
        super().__init__()
        self.args = args
        self.model = model

    def get_loss(self, outputs, targets, query_pred, model):
        """ CE Classification loss
        targets dicts must contain the key "image_label".
        """
        assert 'query_logits' in outputs
        query_logits = outputs['query_logits']
        device = query_logits.device

        target_classes = torch.cat([t['image_label'] for t in targets]) 
        
        criterion = torch.nn.CrossEntropyLoss()
        classification_loss=criterion(query_logits, target_classes)

        losses = {'CE_loss': classification_loss}

        """Ort_loss"""
        l2_loss = nn.MSELoss()
        query_pred = normalize(query_pred, p=2, dim=-1)
        y_pred = torch.matmul(query_pred, query_pred.transpose(-1, -2))
        y_true = torch.eye(query_pred.shape[0], device=query_pred.device)
        # y_true = y_true.unsqueeze(0).repeat(query_pred.shape[0], 1, 1)
        """MSE loss"""
        # query_loss = l2_loss(y_pred, y_true) * 1.0
        """Frobenius loss"""
        query_loss = torch.norm(y_pred - y_true, p='fro') ** 2

        losses['Ort_loss'] = query_loss

        return losses

    def forward(self, outputs, targets, query, model):
        """ This performs the loss computation.
        Parameters:
             outputs: dict of tensors, see the output specification of the model for the format.
             targets: list of dicts, such that len(targets) == batch_size.
                      The expected keys in each dict depends on the losses applied. Here we have used only CE loss.
        """
        losses = {}
        losses.update(self.get_loss(outputs, targets, query, model))
        return losses


def build(args):
    """
    In INTR, each query is responsible for learning class specific information.
    So, the `num_queries` here is actually the number of classes in the dataset.
    """

    if args.dataset_name== 'cub':
        args.num_queries=200
    elif args.dataset_name== 'bird525':
        args.num_queries=525
    elif args.dataset_name== 'fish':
        args.num_queries=183
    elif args.dataset_name== 'dog':
        args.num_queries=120
    elif args.dataset_name== 'butterfly':
        args.num_queries=65
    elif args.dataset_name== 'oxford_iiit_pet':
        args.num_queries=37
    elif args.dataset_name== 'car':
        args.num_queries=196
    elif args.dataset_name== 'craft':
        args.num_queries=100
    else:
        print ("Enter a test dataset")
        exit()

    device = torch.device(args.device)

    backbone = build_backbone(args)
    transformer = build_transformer(args)

    model = INTR(
        args,
        backbone,
        transformer,
        num_queries=args.num_queries,
        )

    criterion = SetCriterion(args, model=model)
    criterion.to(device)

    return model, criterion
