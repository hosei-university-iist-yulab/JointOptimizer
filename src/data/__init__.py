#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
🚀 Created on 01/20/2025🚀

Author: Franck Aboya
Email: mesabo18@gmail.com / messouaboya17@gmail.com
Github: https://github.com/mesabo
Univ: Hosei University, PhD
Dept: Science and Engineering
Lab: Prof YU Keping's Lab
"""

"""Data loading and preprocessing for power systems."""

from .ieee_cases import IEEECaseLoader, load_ieee_case, get_all_cases
from .synthetic_delays import SyntheticDelayGenerator, DelayConfig, CorrelatedDelayGenerator
from .dataset import PowerGridDataset, MultiCaseDataset, create_dataloaders
from .synthetic_grid import (
    SyntheticGridGenerator,
    SyntheticGridConfig,
    SyntheticGridState,
    SyntheticCaseLoader,
    generate_synthetic_grid,
    validate_synthetic_grid,
)


def load_case(case_id: int):
    """Load any case (IEEE or synthetic) by ID."""
    if case_id >= 1000:
        return SyntheticCaseLoader(case_id)
    return IEEECaseLoader(case_id)


__all__ = [
    # IEEE cases
    "IEEECaseLoader",
    "load_ieee_case",
    "get_all_cases",
    # Delays
    "SyntheticDelayGenerator",
    "DelayConfig",
    "CorrelatedDelayGenerator",
    # Dataset
    "PowerGridDataset",
    "MultiCaseDataset",
    "create_dataloaders",
    # Synthetic large-scale grid (V2)
    "SyntheticGridGenerator",
    "SyntheticGridConfig",
    "SyntheticGridState",
    "SyntheticCaseLoader",
    "generate_synthetic_grid",
    "validate_synthetic_grid",
    "load_case",
]
