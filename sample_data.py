from __future__ import annotations

import argparse
import sys
from pathlib import Path

import joblib
import numpy as np


def main() -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    parser = argparse.ArgumentParser(description="Tạo hai file PKL demo theo cấu trúc WHAM-like.")
    parser.add_argument("--output", default="samples")
    parser.add_argument("--frames", type=int, default=8)
    args = parser.parse_args()

    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    cam1, cam2 = make_sample(args.frames)
    joblib.dump([cam1], output / "wham_cam1.pkl")
    joblib.dump([cam2], output / "wham_cam2.pkl")
    print(f"Đã tạo: {(output / 'wham_cam1.pkl').resolve()}")
    print(f"Đã tạo: {(output / 'wham_cam2.pkl').resolve()}")
    return 0


def make_sample(frame_count: int) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
    rng = np.random.default_rng(42)
    frame_ids = np.arange(frame_count)
    pose1 = rng.normal(0.0, 0.035, size=(frame_count, 72))
    pose2 = pose1 + rng.normal(0.0, 0.018, size=(frame_count, 72))
    trans1 = np.column_stack(
        [
            np.linspace(0.0, 0.04, frame_count),
            np.zeros(frame_count),
            np.full(frame_count, 2.0),
        ]
    )
    trans2 = trans1 + np.asarray([0.03, 0.0, -0.04])

    if frame_count > 2:
        pose2[2, 17 * 3 + 1] += 2.8
        pose2[2, 19 * 3 + 2] -= 2.2
    if frame_count > 5:
        pose2[5, 4 * 3 + 0] += 1.4
        pose2[5, 7 * 3 + 1] -= 1.2

    betas = np.zeros((frame_count, 10), dtype=float)
    return (
        {"frame_ids": frame_ids, "pose": pose1, "betas": betas, "trans": trans1},
        {"frame_ids": frame_ids, "pose": pose2, "betas": betas, "trans": trans2},
    )


if __name__ == "__main__":
    raise SystemExit(main())
