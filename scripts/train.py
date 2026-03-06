#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
🚀 Created on 01/18/2025🚀

Author: Franck Aboya
Email: mesabo18@gmail.com / messouaboya17@gmail.com
Github: https://github.com/mesabo
Univ: Hosei University, PhD
Dept: Science and Engineering
Lab: Prof YU Keping's Lab
"""

"""
Training Script for Energy-Information Co-Optimization

Main entry point for training the joint optimizer model.

Usage:
    python train.py --config configs/default.yaml
    python train.py --case 14 --epochs 500 --batch_size 32
"""

import argparse
import os
import sys
import yaml
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts, ReduceLROnPlateau
from pathlib import Path
from datetime import datetime
from typing import Dict, Optional, Any
from tqdm import tqdm

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.models import JointOptimizer
from src.losses import JointLoss
from src.data import create_dataloaders, DelayConfig, PowerGridDataset
from src.utils.statistical_tests import restrict_gpus


def load_config(config_path: str) -> Dict[str, Any]:
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def setup_device(config: Dict) -> torch.device:
    """Setup compute device."""
    if config.get('hardware', {}).get('device') == 'cuda' and torch.cuda.is_available():
        device = torch.device('cuda')
        print(f"Using GPU: {torch.cuda.get_device_name(0)}")
    else:
        device = torch.device('cpu')
        print("Using CPU")
    return device


def setup_wandb(config: Dict, args: argparse.Namespace) -> bool:
    """Setup Weights & Biases logging."""
    if args.no_wandb:
        return False

    try:
        import wandb
        wandb.init(
            project=config.get('wandb', {}).get('project', 'energy-info-coopt'),
            config=config,
            tags=config.get('wandb', {}).get('tags', []),
            name=f"run_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
        )
        return True
    except ImportError:
        print("Warning: wandb not installed, skipping logging")
        return False
    except Exception as e:
        print(f"Warning: Could not initialize wandb: {e}")
        return False


def create_model(config: Dict, dataset: PowerGridDataset, device: torch.device) -> JointOptimizer:
    """Create model from configuration."""
    model_config = config.get('model', {})
    stability_config = config.get('stability', {})

    base_case = dataset.get_base_case()
    n_generators = base_case['n_generators']

    # For large grids (>=1000 buses), disable attention to avoid OOM
    n_buses = base_case.get('n_buses', 0)
    large_grid = n_buses >= 1000
    use_cross_attn = not large_grid and model_config.get('use_cross_attention', True)
    use_causal = not large_grid and model_config.get('use_causal_mask', True)

    # K init scale: "auto" uses grid-aware scaling, else use explicit value
    k_init_raw = stability_config.get('k_init_scale', 0.1)
    k_init_scale = 0.1 if k_init_raw == 'auto' else float(k_init_raw)
    lambda_min_0 = base_case.get('lambda_min', None) if k_init_raw == 'auto' else None

    model = JointOptimizer(
        n_generators=n_generators,
        energy_input_dim=int(model_config.get('energy_input_dim', 5)),
        comm_input_dim=int(model_config.get('comm_input_dim', 3)),
        embed_dim=int(model_config.get('embed_dim', 128)),
        hidden_dim=int(model_config.get('hidden_dim', 256)),
        num_heads=int(model_config.get('num_heads', 8)),
        gnn_layers=int(model_config.get('gnn_layers', 3)),
        decoder_layers=int(model_config.get('decoder_layers', 2)),
        dropout=float(model_config.get('attention_dropout', 0.1)),
        physics_gamma=float(model_config.get('physics_mask_gamma', 1.0)),
        k_init_scale=k_init_scale,
        lambda_min_0=lambda_min_0,
        learnable_k=True,
        use_cross_attention=use_cross_attn,
        use_causal_mask=use_causal,
        use_physics_mask=not large_grid,
    )

    return model.to(device)


def create_loss(config: Dict) -> JointLoss:
    """Create loss function from configuration."""
    loss_config = config.get('loss', {})
    delays_config = config.get('delays', {})

    return JointLoss(
        cost_weight=loss_config.get('cost_weight', 1.0),
        voltage_weight=loss_config.get('voltage_weight', 10.0),
        frequency_weight=loss_config.get('frequency_weight', 100.0),
        latency_weight=loss_config.get('latency_weight', 1.0),
        bandwidth_weight=loss_config.get('bandwidth_penalty', 10.0),
        alpha=loss_config.get('alpha', 1.0),
        beta=loss_config.get('beta', 0.1),
        rho_min=loss_config.get('rho_min', 0.01),
        contrastive_weight=loss_config.get('contrastive_weight', 0.1),
        temperature=loss_config.get('temperature', 0.07),
        tau_budget=delays_config.get('mean_ms', 50),
        tau_max=delays_config.get('max_ms', 500),
    )


def create_optimizer(model: nn.Module, config: Dict) -> optim.Optimizer:
    """Create optimizer from configuration."""
    train_config = config.get('training', {})

    optimizer_name = train_config.get('optimizer', 'adamw').lower()
    lr = float(train_config.get('learning_rate', 1e-4))
    weight_decay = float(train_config.get('weight_decay', 1e-5))

    if optimizer_name == 'adamw':
        return optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    elif optimizer_name == 'adam':
        return optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    elif optimizer_name == 'sgd':
        return optim.SGD(model.parameters(), lr=lr, weight_decay=weight_decay, momentum=0.9)
    else:
        raise ValueError(f"Unknown optimizer: {optimizer_name}")


def create_scheduler(optimizer: optim.Optimizer, config: Dict):
    """Create learning rate scheduler."""
    train_config = config.get('training', {})
    scheduler_name = train_config.get('scheduler', 'cosine').lower()
    epochs = train_config.get('epochs', 500)

    if scheduler_name == 'cosine':
        return CosineAnnealingWarmRestarts(optimizer, T_0=50, T_mult=2)
    elif scheduler_name == 'plateau':
        return ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=10)
    else:
        return None


def train_epoch(
    model: JointOptimizer,
    dataloader: torch.utils.data.DataLoader,
    criterion: JointLoss,
    optimizer: optim.Optimizer,
    device: torch.device,
    impedance_matrix: torch.Tensor,
    config: Dict,
) -> Dict[str, float]:
    """Train for one epoch."""
    model.train()
    total_loss = 0.0
    all_components = {}
    num_batches = 0

    grad_clip = config.get('training', {}).get('gradient_clip', 1.0)

    pbar = tqdm(dataloader, desc="Training", leave=False)
    for batch in pbar:
        # Move to device
        energy_x = batch['energy_x'].to(device)
        comm_x = batch['comm_x'].to(device)
        energy_edge_index = batch['energy_edge_index'].to(device)
        comm_edge_index = batch['comm_edge_index'].to(device)
        tau = batch['tau'].to(device)
        tau_max = batch['tau_max'].to(device)
        lambda_min_0 = batch['lambda_min_0'].to(device)
        u_prev = batch['u_prev'].to(device)
        P_load = batch['P_load'].to(device)

        # Flatten node features for GNN
        batch_size = energy_x.shape[0]
        n_nodes = energy_x.shape[1]

        # Reshape for model (assuming same graph structure in batch)
        energy_x_flat = energy_x.reshape(-1, energy_x.shape[-1])
        comm_x_flat = comm_x.reshape(-1, comm_x.shape[-1])

        # Create batch tensor
        batch_tensor = torch.arange(batch_size, device=device).repeat_interleave(n_nodes)

        # Forward pass
        optimizer.zero_grad()

        outputs = model(
            energy_x=energy_x_flat,
            energy_edge_index=energy_edge_index,
            comm_x=comm_x_flat,
            comm_edge_index=comm_edge_index,
            tau=tau,
            tau_max=tau_max[0],  # Same for all in batch
            lambda_min_0=lambda_min_0,
            impedance_matrix=impedance_matrix.to(device) if impedance_matrix is not None else None,
            batch=batch_tensor,
        )

        # Extract P_gen from control output
        n_gen = tau.shape[1]
        P_gen = outputs['u'][:, :n_gen]  # First half is P

        # Compute loss
        loss, components = criterion(
            u=outputs['u'],
            rho=outputs['rho'],
            h_E=outputs['h_E'],
            h_I=outputs['h_I'],
            P_gen=P_gen,
            tau=tau,
            lambda_min_0=lambda_min_0,
            u_prev=u_prev,
            P_load=P_load,
            impedance_matrix=impedance_matrix.to(device) if impedance_matrix is not None else None,
        )

        # Backward pass
        loss.backward()

        # Gradient clipping
        if grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)

        optimizer.step()

        # Accumulate metrics
        total_loss += loss.item()
        for key, value in components.items():
            if key not in all_components:
                all_components[key] = 0.0
            all_components[key] += value

        num_batches += 1
        pbar.set_postfix({'loss': loss.item()})

    # Average metrics
    metrics = {'train_loss': total_loss / num_batches}
    for key, value in all_components.items():
        metrics[f'train_{key}'] = value / num_batches

    return metrics


@torch.no_grad()
def validate(
    model: JointOptimizer,
    dataloader: torch.utils.data.DataLoader,
    criterion: JointLoss,
    device: torch.device,
    impedance_matrix: torch.Tensor,
) -> Dict[str, float]:
    """Validate model."""
    model.eval()
    total_loss = 0.0
    all_components = {}
    num_batches = 0

    for batch in tqdm(dataloader, desc="Validating", leave=False):
        # Move to device
        energy_x = batch['energy_x'].to(device)
        comm_x = batch['comm_x'].to(device)
        energy_edge_index = batch['energy_edge_index'].to(device)
        comm_edge_index = batch['comm_edge_index'].to(device)
        tau = batch['tau'].to(device)
        tau_max = batch['tau_max'].to(device)
        lambda_min_0 = batch['lambda_min_0'].to(device)
        u_prev = batch['u_prev'].to(device)
        P_load = batch['P_load'].to(device)

        batch_size = energy_x.shape[0]
        n_nodes = energy_x.shape[1]

        energy_x_flat = energy_x.reshape(-1, energy_x.shape[-1])
        comm_x_flat = comm_x.reshape(-1, comm_x.shape[-1])
        batch_tensor = torch.arange(batch_size, device=device).repeat_interleave(n_nodes)

        outputs = model(
            energy_x=energy_x_flat,
            energy_edge_index=energy_edge_index,
            comm_x=comm_x_flat,
            comm_edge_index=comm_edge_index,
            tau=tau,
            tau_max=tau_max[0],
            lambda_min_0=lambda_min_0,
            impedance_matrix=impedance_matrix.to(device) if impedance_matrix is not None else None,
            batch=batch_tensor,
        )

        n_gen = tau.shape[1]
        P_gen = outputs['u'][:, :n_gen]

        loss, components = criterion(
            u=outputs['u'],
            rho=outputs['rho'],
            h_E=outputs['h_E'],
            h_I=outputs['h_I'],
            P_gen=P_gen,
            tau=tau,
            lambda_min_0=lambda_min_0,
            u_prev=u_prev,
            P_load=P_load,
            impedance_matrix=impedance_matrix.to(device) if impedance_matrix is not None else None,
        )

        total_loss += loss.item()
        for key, value in components.items():
            if key not in all_components:
                all_components[key] = 0.0
            all_components[key] += value
        num_batches += 1

    metrics = {'val_loss': total_loss / num_batches}
    for key, value in all_components.items():
        metrics[f'val_{key}'] = value / num_batches

    return metrics


def save_checkpoint(
    model: JointOptimizer,
    optimizer: optim.Optimizer,
    epoch: int,
    metrics: Dict[str, float],
    save_dir: str,
    is_best: bool = False,
):
    """Save model checkpoint."""
    os.makedirs(save_dir, exist_ok=True)

    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'metrics': metrics,
        'config': model.config,
    }

    # Save latest
    torch.save(checkpoint, os.path.join(save_dir, 'latest.pt'))

    # Save best
    if is_best:
        torch.save(checkpoint, os.path.join(save_dir, 'best.pt'))

    # Save periodic
    torch.save(checkpoint, os.path.join(save_dir, f'epoch_{epoch}.pt'))


def main():
    restrict_gpus()
    parser = argparse.ArgumentParser(description="Train Energy-Info Co-Optimizer")
    parser.add_argument('--config', type=str, default='configs/default.yaml',
                        help='Path to config file')
    parser.add_argument('--case', type=int, default=None,
                        help='IEEE case number (overrides config)')
    parser.add_argument('--epochs', type=int, default=None,
                        help='Number of epochs (overrides config)')
    parser.add_argument('--batch_size', type=int, default=None,
                        help='Batch size (overrides config)')
    parser.add_argument('--lr', type=float, default=None,
                        help='Learning rate (overrides config)')
    parser.add_argument('--no_wandb', action='store_true',
                        help='Disable wandb logging')
    parser.add_argument('--checkpoint', type=str, default=None,
                        help='Resume from checkpoint')
    parser.add_argument('--save_dir', type=str, default='checkpoints',
                        help='Directory to save checkpoints')

    args = parser.parse_args()

    # Load config
    config = load_config(args.config)

    # Override with command line args
    if args.case is not None:
        config['data']['ieee_cases'] = [args.case]
    if args.epochs is not None:
        config['training']['epochs'] = args.epochs
    if args.batch_size is not None:
        config['training']['batch_size'] = args.batch_size
    if args.lr is not None:
        config['training']['learning_rate'] = args.lr

    # Setup
    device = setup_device(config)
    use_wandb = setup_wandb(config, args)

    # Set seed
    seed = config.get('reproducibility', {}).get('seed', 42)
    torch.manual_seed(seed)

    # Create data
    print("Creating dataloaders...")
    data_config = config.get('data', {})
    delays_config = config.get('delays', {})

    delay_config = DelayConfig(
        distribution=delays_config.get('distribution', 'lognormal'),
        mean_ms=delays_config.get('mean_ms', 50),
        std_ms=delays_config.get('std_ms', 20),
        min_ms=delays_config.get('min_ms', 5),
        max_ms=delays_config.get('max_ms', 500),
    )

    case_id = data_config.get('ieee_cases', [14])[0]
    train_loader, val_loader, test_loader = create_dataloaders(
        case_id=case_id,
        num_scenarios=data_config.get('num_scenarios', 1000),
        train_split=data_config.get('train_split', 0.7),
        val_split=data_config.get('val_split', 0.15),
        batch_size=config.get('training', {}).get('batch_size', 32),
        num_workers=config.get('hardware', {}).get('num_workers', 4),
        seed=seed,
        delay_config=delay_config,
    )

    # Get dataset info for model creation
    dataset = train_loader.dataset.dataset  # Access underlying dataset
    impedance_matrix = dataset.get_impedance_matrix()

    # Create model
    print("Creating model...")
    model = create_model(config, dataset, device)
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

    # Create loss
    criterion = create_loss(config)

    # Create optimizer
    optimizer = create_optimizer(model, config)
    scheduler = create_scheduler(optimizer, config)

    # Resume from checkpoint
    start_epoch = 0
    best_val_loss = float('inf')

    if args.checkpoint:
        print(f"Loading checkpoint: {args.checkpoint}")
        checkpoint = torch.load(args.checkpoint, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint['epoch'] + 1
        best_val_loss = checkpoint['metrics'].get('val_loss', float('inf'))

    # Training loop
    epochs = config.get('training', {}).get('epochs', 500)
    patience = config.get('training', {}).get('patience', 50)
    patience_counter = 0

    print(f"\nStarting training for {epochs} epochs...")
    print(f"  Case: IEEE {case_id}")
    print(f"  Train samples: {len(train_loader.dataset)}")
    print(f"  Val samples: {len(val_loader.dataset)}")
    print(f"  Test samples: {len(test_loader.dataset)}")
    print()

    for epoch in range(start_epoch, epochs):
        print(f"Epoch {epoch + 1}/{epochs}")

        # Train
        train_metrics = train_epoch(
            model, train_loader, criterion, optimizer, device,
            impedance_matrix, config
        )

        # Validate
        val_metrics = validate(
            model, val_loader, criterion, device, impedance_matrix
        )

        # Update scheduler
        if scheduler is not None:
            if isinstance(scheduler, ReduceLROnPlateau):
                scheduler.step(val_metrics['val_loss'])
            else:
                scheduler.step()

        # Log metrics
        current_lr = optimizer.param_groups[0]['lr']
        K_values = model.get_coupling_constants()

        print(f"  Train Loss: {train_metrics['train_loss']:.4f}")
        print(f"  Val Loss: {val_metrics['val_loss']:.4f}")
        print(f"  LR: {current_lr:.2e}")
        print(f"  K_i mean: {K_values.mean().item():.4f}")

        # Wandb logging
        if use_wandb:
            import wandb
            wandb.log({
                **train_metrics,
                **val_metrics,
                'lr': current_lr,
                'K_mean': K_values.mean().item(),
                'K_max': K_values.max().item(),
                'epoch': epoch,
            })

        # Check for improvement
        is_best = val_metrics['val_loss'] < best_val_loss
        if is_best:
            best_val_loss = val_metrics['val_loss']
            patience_counter = 0
        else:
            patience_counter += 1

        # Save checkpoint
        all_metrics = {**train_metrics, **val_metrics}
        save_checkpoint(
            model, optimizer, epoch, all_metrics,
            args.save_dir, is_best=is_best
        )

        # Early stopping
        if patience_counter >= patience:
            print(f"\nEarly stopping after {patience} epochs without improvement")
            break

        print()

    # Final evaluation on test set
    print("\nFinal evaluation on test set...")
    test_metrics = validate(model, test_loader, criterion, device, impedance_matrix)
    print(f"Test Loss: {test_metrics['val_loss']:.4f}")

    if use_wandb:
        import wandb
        wandb.log({'test_loss': test_metrics['val_loss']})
        wandb.finish()

    print("\nTraining complete!")
    print(f"Best validation loss: {best_val_loss:.4f}")
    print(f"Checkpoints saved to: {args.save_dir}")


if __name__ == '__main__':
    main()
