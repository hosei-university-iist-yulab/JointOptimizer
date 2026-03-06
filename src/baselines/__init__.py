#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
🚀 Created on 02/03/2025🚀

Author: Franck Aboya
Email: mesabo18@gmail.com / messouaboya17@gmail.com
Github: https://github.com/mesabo
Univ: Hosei University, PhD
Dept: Science and Engineering
Lab: Prof YU Keping's Lab
"""

"""
Baseline Models for Comparison

Required baselines from REQUIREMENTS.md:
- B1: Sequential OPF + QoS (decoupled optimization)
- B2: MLP Joint (simple feedforward)
- B3: GNN-only (no attention)
- B4: LSTM Joint (recurrent)
- B5: CNN Joint (convolutional)
- B6: Vanilla Transformer (standard attention)
- B7: Transformer (no L_coupling loss)

V2 additions:
- B8: Heterogeneous GNN (Q3.1)
- B9: DeepOPF (Q5.2)
"""

from .sequential_opf import SequentialOPFQoS
from .mlp_joint import MLPJoint
from .gnn_only import GNNOnly
from .lstm_joint import LSTMJoint
from .cnn_joint import CNNJoint
from .vanilla_transformer import VanillaTransformer
from .transformer_no_coupling import TransformerNoCoupling
from .heterogeneous_gnn import HeterogeneousGNN
from .deepopf import DeepOPF

__all__ = [
    "SequentialOPFQoS",
    "MLPJoint",
    "GNNOnly",
    "LSTMJoint",
    "CNNJoint",
    "VanillaTransformer",
    "TransformerNoCoupling",
    "HeterogeneousGNN",
    "DeepOPF",
]
