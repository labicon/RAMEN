import argparse
import numpy as np
from typing import Optional


def load_positions_from_traj(traj_path: str) -> np.ndarray:
    """
    traj.txt: each line has 16 floats (row-major), representing 4x4 c2w matrix.
    Returns: (N, 3) positions (x,y,z) from the last column.
    """
    positions = []
    with open(traj_path, "r") as f:
        for ln, line in enumerate(f, start=1):
            s = line.strip()
            if not s:
                continue
            parts = s.split()
            if len(parts) != 16:
                raise ValueError(f"{traj_path}:{ln}: expected 16 floats, got {len(parts)}")
            vals = np.array([float(x) for x in parts], dtype=np.float64)
            T = vals.reshape(4, 4)  # row-major
            t = T[:3, 3]
            positions.append(t)
    if not positions:
        raise ValueError(f"No valid poses found in {traj_path}")
    return np.vstack(positions)


def compute_bounds(
    positions: np.ndarray,
    margin: float = 0.0,
    robust_quantile: Optional[float] = None,
) -> np.ndarray:
    """
    positions: (N, 3)
    margin: expand bound by +/- margin (meters)
    robust_quantile: if set (e.g. 0.01), use [q, 1-q] percentiles per axis instead of min/max.
    Returns: (3, 2) [[xmin,xmax],[ymin,ymax],[zmin,zmax]]
    """
    if robust_quantile is None:
        lo = positions.min(axis=0)
        hi = positions.max(axis=0)
    else:
        q = float(robust_quantile)
        lo = np.quantile(positions, q, axis=0)
        hi = np.quantile(positions, 1.0 - q, axis=0)

    lo = lo - margin
    hi = hi + margin

    out = np.stack([lo, hi], axis=1)  # (3,2)
    return out


def shrink_bounds(bounds: np.ndarray, shrink: float) -> np.ndarray:
    """
    bounds: (3,2)
    shrink: positive value shrinks each side inward by shrink (meters).
    """
    b = bounds.copy()
    b[:, 0] = b[:, 0] + shrink
    b[:, 1] = b[:, 1] - shrink
    if np.any(b[:, 1] <= b[:, 0]):
        raise ValueError("Shrink too large: marching_cubes_bound became invalid.")
    return b


def format_yaml_bounds(bounds: np.ndarray) -> str:
    arr = bounds.tolist()
    return f"[[{arr[0][0]:.2f},{arr[0][1]:.2f}],[{arr[1][0]:.2f},{arr[1][1]:.2f}],[{arr[2][0]:.2f},{arr[2][1]:.2f}]]"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--traj", required=True)
    ap.add_argument("--margin", type=float, default=0.0)
    ap.add_argument("--robust_q", type=float, default=None)
    ap.add_argument("--mc_shrink", type=float, default=0.1)

    # ADDED: axis handling + vertical padding
    ap.add_argument("--up_axis", choices=["y", "z"], default="y",
                    help="Which axis is vertical in your dataset/world.")
    ap.add_argument("--down_pad", type=float, default=2.0,
                    help="Extend bound downward along up_axis (meters).")
    ap.add_argument("--up_pad", type=float, default=1.0,
                    help="Extend bound upward along up_axis (meters).")

    # ADDED: uniform padding for all axes
    ap.add_argument("--pad", type=float, default=2.0,
                    help="Uniform +/- padding added to all axes (meters). E.g. 3.0")

    args = ap.parse_args()

    pos = load_positions_from_traj(args.traj)
    bound = compute_bounds(pos, margin=args.margin, robust_quantile=args.robust_q)

    # Uniform padding on x/y/z
    if args.pad != 0.0:
        bound[:, 0] -= float(args.pad)
        bound[:, 1] += float(args.pad)


    mc_bound = shrink_bounds(bound, shrink=args.mc_shrink)

    print("bound:", format_yaml_bounds(bound))
    print("marching_cubes_bound:", format_yaml_bounds(mc_bound))


if __name__ == "__main__":
    main()