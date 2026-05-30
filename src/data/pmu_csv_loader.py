"""Loader for the IEEE DataPort PMU FREQUENCY CASE I-IV CSV archive.

The archive ships 4 CSV files (Case I MITM, Case II Resembling, Case III
Repetition, Case IV Missing) of 8-PMU frequency time-series sampled at the
field-deployed PMU rate. Each row is one timestamp; each column past the
timestamp is the per-PMU frequency in Hz. We treat each PMU as a bus and
zero-pad the missing channels (P, Q, V, theta on the energy side and tau,
R, B on the communication side) so the trained JointOptimizer can run a
forward pass without retraining.

Returns a case-dict that drops in to the framework's IEEECaseLoader path:
``{'n_buses', 'n_generators', 'edge_index', 'lambda_min', 'impedance_matrix',
   'energy_x', 'comm_x', 'tau', 'tau_max', 'lambda_min_0'}``.
"""
from __future__ import annotations
from pathlib import Path
import csv
import math
import torch

CSV_FILES = {
    'mitm': 'Case I MITM_Hz.csv',
    'resembling': 'Case II Data Resembling_Hz.csv',
    'repetition': 'Case III Data Repetition_Hz.csv',
    'missing': 'Case IV Data Missing_Hz.csv',
}

DEFAULT_DIR = Path(__file__).resolve().parents[2] / 'data/DATASETS PMU FREQUENCY CASE I TO CASE IV'
NOMINAL_HZ = 50.0
DEFAULT_TAU_MAX_MS = 500.0


def _build_ring_edges(n: int) -> torch.Tensor:
    """Bidirectional ring topology over n buses (placeholder; the CSV archive
    does not ship a topology). Replace with the real adjacency once available."""
    src, dst = [], []
    for i in range(n):
        j = (i + 1) % n
        src.append(i); dst.append(j)
        src.append(j); dst.append(i)
    return torch.tensor([src, dst], dtype=torch.long)


def load_pmu_csv(case_name: str = 'mitm', data_dir: Path | str | None = None,
                 max_timesteps: int | None = 256, batch_size: int = 8,
                 default_lambda_min: float = 0.4) -> dict:
    """Load one of the 4 PMU FREQUENCY CSVs into the framework's case-dict shape.

    Args:
        case_name: one of ``'mitm', 'resembling', 'repetition', 'missing'``.
        data_dir: parent directory holding the 4 CSV files (defaults to
            ``data/DATASETS PMU FREQUENCY CASE I TO CASE IV/``).
        max_timesteps: cap on rows read; ``None`` reads the whole file.
        batch_size: how many timesteps to bundle into a forward-pass batch.
        default_lambda_min: undelayed stability margin to use; the CSV does
            not provide a system Jacobian, so we fall back to the framework's
            small-signal default.

    Returns:
        A dict matching the IEEECaseLoader.load() shape, plus a ``'tau'`` and
        ``'tau_max'`` tensor pre-built so the JointOptimizer forward pass can
        consume the dict directly.
    """
    if case_name not in CSV_FILES:
        raise KeyError(f"Unknown PMU case '{case_name}'; expected one of {list(CSV_FILES)}")
    base = Path(data_dir) if data_dir is not None else DEFAULT_DIR
    path = base / CSV_FILES[case_name]
    if not path.exists():
        raise FileNotFoundError(f"PMU CSV not found at {path}")

    rows: list[list[float]] = []
    with path.open('r', newline='') as fh:
        reader = csv.reader(fh)
        header = next(reader)
        # First column is timestamp (string), remaining are PMU frequencies.
        n_pmu = len(header) - 1
        for i, row in enumerate(reader):
            if max_timesteps is not None and i >= max_timesteps:
                break
            try:
                rows.append([float(x) for x in row[1:1 + n_pmu]])
            except ValueError:
                continue

    if not rows:
        raise ValueError(f"No numeric rows parsed from {path}")

    omega_hz = torch.tensor(rows, dtype=torch.float32)              # [T, n_pmu]
    omega = omega_hz - NOMINAL_HZ                                    # frequency deviation in Hz

    # Bundle T rows into B batches of length batch_size for the model.
    T, n_pmu = omega.shape
    n_batches = max(1, T // batch_size)
    omega = omega[:n_batches * batch_size].reshape(n_batches, batch_size, n_pmu)
    # Use the LAST timestep per batch as the operating point.
    omega_op = omega[:, -1, :]                                       # [B, n_pmu]

    # Build [B, n_pmu, 5] energy_x = (P, Q, V, theta, omega) with omega real.
    B = omega_op.shape[0]
    energy_x = torch.zeros(B, n_pmu, 5)
    energy_x[..., 4] = omega_op
    # Voltage magnitude default 1.0 p.u. (the archive is freq-only); P/Q/theta zero-padded.
    energy_x[..., 2] = 1.0

    # Communication features (tau, R, B). The archive's filename tags the
    # attack scenario; encode that as a categorical perturbation in tau:
    #   MITM       -> base tau
    #   Resembling -> 1.5x base tau (subtle replay)
    #   Repetition -> 2.0x base tau (stale data)
    #   Missing    -> 3.0x base tau (worst case)
    tau_multiplier = {'mitm': 1.0, 'resembling': 1.5, 'repetition': 2.0, 'missing': 3.0}.get(
        case_name, 1.0)
    base_tau_ms = 50.0 * tau_multiplier
    tau = torch.full((B, n_pmu), base_tau_ms / 1000.0)               # seconds, [B, n_pmu]
    tau_max = torch.full((n_pmu,), DEFAULT_TAU_MAX_MS / 1000.0)      # seconds
    R = torch.full((B, n_pmu), 1.0)                                  # bandwidth utilization
    Bcomm = torch.full((B, n_pmu), 1.0)                              # available bandwidth
    comm_x = torch.stack([tau, R, Bcomm], dim=-1)                    # [B, n_pmu, 3]

    # Ring topology placeholder.
    edge_index = _build_ring_edges(n_pmu)
    impedance_matrix = torch.full((n_pmu, n_pmu), 1e6)
    for r, c in zip(*edge_index.tolist()):
        impedance_matrix[r, c] = 1.0
    impedance_matrix.fill_diagonal_(0)

    return {
        'n_buses': n_pmu,
        'n_generators': n_pmu,
        'edge_index': edge_index,
        'energy_x': energy_x,
        'comm_x': comm_x,
        'tau': tau,
        'tau_max': tau_max,
        'lambda_min': default_lambda_min,
        'lambda_min_0': torch.full((B,), default_lambda_min),
        'impedance_matrix': impedance_matrix,
        'omega_full_series': omega.reshape(-1, n_pmu),
        'case_name': case_name,
        'n_timesteps': T,
        'tau_multiplier': tau_multiplier,
    }
