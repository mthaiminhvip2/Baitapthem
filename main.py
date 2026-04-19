from __future__ import annotations

import argparse
import json
import sys
import threading
from dataclasses import asdict
from pathlib import Path
from typing import Any

from html_report import write_html_report
from inference import InferenceConfig, LearnableInverseKinematicSolver
from pkl_io import load_pose_sequence
from pose_analysis import AnalysisSettings, analyze_sequences, sort_frame_ids


DEFAULT_FILE_1 = "wham_cam1.pkl"
DEFAULT_FILE_2 = "wham_cam2.pkl"


def configure_console() -> None:
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            getattr(stream, "reconfigure")(encoding="utf-8", errors="replace")


def main() -> int:
    configure_console()
    args = parse_args()
    if not args.no_prompt:
        args = fill_from_prompts(args)
    args = normalize_args(args)

    settings = AnalysisSettings(
        orientation_epsilon=args.orientation_epsilon,
        distance_rule=args.distance_rule,
        mad_factor=args.mad_factor,
    )

    try:
        seq_a = load_pose_sequence(
            args.file1, args.person_id, args.smpl_model_dir, args.gender
        )
        seq_b = load_pose_sequence(
            args.file2, args.person_id, args.smpl_model_dir, args.gender
        )
    except Exception as exc:
        print(f"Lỗi đọc PKL: {exc}", file=sys.stderr)
        print(
            "Gợi ý: đặt hai file mặc định wham_cam1.pkl và wham_cam2.pkl trong thư mục này, "
            "hoặc chạy `python sample_data.py --output samples` để tạo dữ liệu demo.",
            file=sys.stderr,
        )
        return 2

    before = analyze_sequences(seq_a.frames, seq_b.frames, settings)
    solver = LearnableInverseKinematicSolver(
        InferenceConfig(
            checkpoint=args.checkpoint,
            learnable_smplify_src=args.learnable_smplify_src,
            smpl_model_dir=args.smpl_model_dir,
            device=args.device,
            fallback_refiner=args.fallback_refiner,
            smooth_window=args.smooth_window,
        )
    )
    refined_a = solver.refine(seq_a)
    refined_b = solver.refine(seq_b)
    after = analyze_sequences(refined_a.frames, refined_b.frames, settings)

    common_frame_ids = sort_frame_ids(set(before).intersection(after))
    try:
        selected_frame_ids = select_frames(args.frame_id, common_frame_ids)
    except Exception as exc:
        print(f"Lỗi chọn frame: {exc}", file=sys.stderr)
        return 3
    print_report(before, after, selected_frame_ids, args.reduced)

    write_json_report(
        args.output_json,
        before,
        after,
        selected_frame_ids,
        seq_a,
        seq_b,
        refined_a,
        refined_b,
    )
    html_path = write_html_report(
        args.output_html, before, after, selected_frame_ids
    )
    print(f"\nĐã ghi HTML: {html_path.resolve()}")
    print(f"Đã ghi JSON: {Path(args.output_json).resolve()}")
    if solver.load_error:
        print(f"Lưu ý inference: {solver.load_error}")
    return 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Phân tích hai file SMPL/pose PKL trước và sau bước Learnable IK refinement."
    )
    parser.add_argument(
        "--file1",
        default=None,
        help=f"PKL camera/video 1, mặc định {DEFAULT_FILE_1}",
    )
    parser.add_argument(
        "--file2",
        default=None,
        help=f"PKL camera/video 2, mặc định {DEFAULT_FILE_2}",
    )
    parser.add_argument(
        "--reduced",
        default=None,
        choices=("true", "false"),
        help="true: chỉ in kết quả chính",
    )
    parser.add_argument(
        "--frame-id",
        default=None,
        help="Frame cần in, hoặc 'all' để in tất cả",
    )
    parser.add_argument(
        "--no-prompt",
        action="store_true",
        help="Không hỏi tương tác, dùng tham số/default",
    )
    parser.add_argument(
        "--timeout", type=float, default=5.0, help="Số giây chờ nhập liệu"
    )
    parser.add_argument("--person-id", type=int, default=0)
    parser.add_argument(
        "--gender", default="neutral", choices=("neutral", "male", "female")
    )
    parser.add_argument(
        "--smpl-model-dir",
        default=None,
        help="Thư mục chứa SMPL_FEMALE/MALE/NEUTRAL.pkl",
    )
    parser.add_argument(
        "--checkpoint",
        default=None,
        help="Checkpoint pretrained của Learnable-SMPLify",
    )
    parser.add_argument(
        "--learnable-smplify-src",
        default=None,
        help="Đường dẫn tới thư mục src của repo Learnable-SMPLify",
    )
    parser.add_argument("--device", default="auto", help="auto, cpu hoặc cuda")
    parser.add_argument(
        "--fallback-refiner", default="smooth", choices=("smooth", "none")
    )
    parser.add_argument("--smooth-window", type=int, default=3)
    parser.add_argument(
        "--distance-rule", default="hybrid", choices=("hybrid", "mad", "q3")
    )
    parser.add_argument("--mad-factor", type=float, default=2.0)
    parser.add_argument("--orientation-epsilon", type=float, default=1e-2)
    parser.add_argument("--output-html", default="results.html")
    parser.add_argument("--output-json", default="results.json")
    return parser.parse_args()


def fill_from_prompts(args: argparse.Namespace) -> argparse.Namespace:
    timeout = args.timeout
    if args.file1 is None:
        args.file1 = (
            input_with_timeout(
                f"Nhập file .pkl thứ nhất [{DEFAULT_FILE_1}] sau {timeout:g}s dùng mặc định: ",
                timeout,
            )
            or DEFAULT_FILE_1
        )
    if args.file2 is None:
        args.file2 = (
            input_with_timeout(
                f"Nhập file .pkl thứ hai [{DEFAULT_FILE_2}] sau {timeout:g}s dùng mặc định: ",
                timeout,
            )
            or DEFAULT_FILE_2
        )
    if args.reduced is None:
        reduced_text = input_with_timeout(
            f"In giản lược reduced=True? [Y/n] sau {timeout:g}s dùng Y: ",
            timeout,
        )
        args.reduced = (
            "false"
            if reduced_text.lower() in ("n", "no", "false", "0")
            else "true"
        )
    if args.frame_id is None:
        args.frame_id = (
            input_with_timeout(
                f"Nhập frame_id hoặc all [all] sau {timeout:g}s dùng all: ",
                timeout,
            )
            or "all"
        )
    return args


def normalize_args(args: argparse.Namespace) -> argparse.Namespace:
    args.file1 = args.file1 or DEFAULT_FILE_1
    args.file2 = args.file2 or DEFAULT_FILE_2
    args.frame_id = args.frame_id or "all"
    args.reduced = parse_bool(args.reduced)
    return args


def input_with_timeout(prompt: str, timeout: float) -> str:
    print(prompt, end="", flush=True)
    result: list[str | None] = [None]

    def read_input() -> None:
        try:
            result[0] = sys.stdin.readline().rstrip("\r\n")
        except Exception:
            result[0] = ""

    thread = threading.Thread(target=read_input, daemon=True)
    thread.start()
    thread.join(timeout)
    if thread.is_alive():
        print("\n[Hết thời gian] Dùng giá trị mặc định.")
        return ""
    return result[0] or ""


def parse_bool(value: bool | str | None) -> bool:
    if isinstance(value, bool):
        return value
    if value is None:
        return True
    return str(value).lower() in ("true", "1", "yes", "y", "co", "có")


def select_frames(
    frame_id: str | None, common_frame_ids: list[int | str]
) -> list[int | str]:
    if not common_frame_ids:
        raise ValueError("Hai file không có frame_id chung.")
    if frame_id is None or str(frame_id).lower() == "all":
        return common_frame_ids
    if str(frame_id).lower() == "first_500":
        return common_frame_ids[:500]

    if str(frame_id).lstrip("-").isdigit():
        wanted: int | str = int(str(frame_id))
    else:
        wanted = frame_id

    if wanted not in common_frame_ids:
        raise ValueError(f"Frame {frame_id!r} không có trong cả hai file.")
    return [wanted]


def print_report(
    before: dict[Any, Any],
    after: dict[Any, Any],
    frame_ids: list[int | str],
    reduced: bool,
) -> None:
    for frame_id in frame_ids:
        print(f"\nFrame {frame_id}")
        print_analysis("Trước sửa", before[frame_id], reduced)
        print_analysis("Sau sửa", after[frame_id], reduced)


def print_analysis(label: str, analysis: Any, reduced: bool) -> None:
    print(f"  {label}:")
    print(f"    M = {analysis.m_conflicts}")
    print(f"    D = {analysis.d_distance_errors}")
    print(
        f"    Q1={fmt(analysis.q1)} | Q3={fmt(analysis.q3)} | mean={fmt(analysis.mean)}"
    )
    if not reduced:
        print(f"    Ngưỡng D = {analysis.threshold:.6f}")
        print("    Sai số theo pose:")
        for name, error in sorted(
            analysis.per_joint_errors.items(),
            key=lambda item: item[1],
            reverse=True,
        ):
            print(f"      {name:<18} {error:.6f}")


def fmt(value: float | None) -> str:
    return "N/A" if value is None else f"{value:.6f}"


def write_json_report(
    output_path: str,
    before: dict[Any, Any],
    after: dict[Any, Any],
    frame_ids: list[int | str],
    seq_a: Any,
    seq_b: Any,
    refined_a: Any,
    refined_b: Any,
) -> None:
    def convert(analysis: Any) -> dict[str, Any]:
        data = asdict(analysis)
        data["pair_errors"] = list(analysis.pair_errors)
        return data

    payload = {
        "input_a": str(seq_a.source_path),
        "input_b": str(seq_b.source_path),
        "metadata_a": seq_a.metadata,
        "metadata_b": seq_b.metadata,
        "refined_metadata_a": refined_a.metadata,
        "refined_metadata_b": refined_b.metadata,
        "frames": {
            str(frame_id): {
                "before": convert(before[frame_id]),
                "after": convert(after[frame_id]),
            }
            for frame_id in frame_ids
        },
    }
    Path(output_path).write_text(
        json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8"
    )


if __name__ == "__main__":
    raise SystemExit(main())
