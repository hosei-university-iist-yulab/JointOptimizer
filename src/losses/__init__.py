#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
🚀 Created on 01/26/2025🚀

Author: Franck Aboya
Email: mesabo18@gmail.com / messouaboya17@gmail.com
Github: https://github.com/mesabo
Univ: Hosei University, PhD
Dept: Science and Engineering
Lab: Prof YU Keping's Lab
"""

"""Loss functions for Energy-Information Co-Optimization."""

from .coupling_loss import CouplingLoss, LogBarrierStabilityLoss
from .energy_loss import EnergyLoss
from .communication_loss import CommunicationLoss
from .contrastive import PhysicsAwareContrastiveLoss
from .combined import JointLoss

__all__ = [
    "CouplingLoss",
    "LogBarrierStabilityLoss",
    "EnergyLoss",
    "CommunicationLoss",
    "PhysicsAwareContrastiveLoss",
    "JointLoss",
]
