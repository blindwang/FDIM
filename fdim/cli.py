"""Command line interface for installed FDIM packages."""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path


PACKAGE_ROOT = Path(__file__).resolve().parent
DEFAULT_MODEL_PATH = PACKAGE_ROOT / "dist" / "checkpoints" / "dist_5.0.0.ckpt"


def _ensure_legacy_import_path() -> None:
    """Keep existing intra-package absolute imports working after pip install."""
    package_root = str(PACKAGE_ROOT)
    dist_root = str(PACKAGE_ROOT / "dist")
    for path in (package_root, dist_root):
        if path not in sys.path:
            sys.path.insert(0, path)


def _str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ("yes", "true", "t", "y", "1"):
        return True
    if v.lower() in ("no", "false", "f", "n", "0"):
        return False
    raise argparse.ArgumentTypeError("Boolean value expected.")


def _add_common_infer_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--model_path", type=str, default=str(DEFAULT_MODEL_PATH))
    parser.add_argument("--ref_fmt", type=str, default="rgb")
    parser.add_argument("--dis_fmt", type=str, default="rgb")
    parser.add_argument("--ref_bit_depth", type=int, default=8)
    parser.add_argument("--dis_bit_depth", type=int, default=8)
    parser.add_argument("--ref_width", type=int, default=None)
    parser.add_argument("--ref_height", type=int, default=None)
    parser.add_argument("--dis_width", type=int, default=None)
    parser.add_argument("--dis_height", type=int, default=None)
    parser.add_argument("--frame_rate", type=int, default=None)
    parser.add_argument("--input_resolution", type=str, default="ori")
    parser.add_argument("--calc_interval", type=int, default=1)
    parser.add_argument("--preprocess", type=str, default="none", choices=["none", "pu21"])
    parser.add_argument("--display_model", type=str, default=None)
    parser.add_argument("--resize", type=_str2bool, default=False)
    parser.add_argument("--resize_method", type=str, default="default")
    parser.add_argument("--n_threads", type=int, default=32)
    parser.add_argument("--vmaf_version", type=str, default=None)
    parser.add_argument("--video_temp_path", type=str, default="./data/video_temp/")
    parser.add_argument("--gpu_idx", type=int, default=0)


def _single_parser(subparsers) -> None:
    parser = subparsers.add_parser("single", help="Infer one reference/distorted video pair.")
    parser.add_argument("--ref", "--ref_video_root", dest="ref_video_root", required=True)
    parser.add_argument("--dis", "--dis_video_root", dest="dis_video_root", required=True)
    _add_common_infer_args(parser)
    parser.set_defaults(func=_run_single)


def _batch_parser(subparsers) -> None:
    parser = subparsers.add_parser("batch", help="Infer videos listed in a CSV file.")
    parser.add_argument("--csv_path", required=True)
    parser.add_argument("--ref_dir", required=True)
    parser.add_argument("--dis_dir", required=True)
    parser.add_argument("--save_dir", type=str, default="data/result")
    parser.add_argument("--save_name", type=str, default="fdim_results")
    parser.add_argument("--dis_column", type=str, default="dis_name")
    parser.add_argument("--ref_column", type=str, default="ref_name")
    parser.add_argument("--mos_column", type=str, default="mos")
    parser.add_argument("--ref_fmt", type=str, default="rgb")
    parser.add_argument("--dis_fmt", type=str, default="rgb")
    parser.add_argument("--ref_bit_depth_column", type=str, default="ref_bit_depth")
    parser.add_argument("--dis_bit_depth_column", type=str, default="dis_bit_depth")
    parser.add_argument("--ref_width_column", type=str, default=None)
    parser.add_argument("--ref_height_column", type=str, default=None)
    parser.add_argument("--dis_width_column", type=str, default=None)
    parser.add_argument("--dis_height_column", type=str, default=None)
    parser.add_argument("--frame_rate_col", type=str, default=None)
    parser.add_argument("--model_path", type=str, default=str(DEFAULT_MODEL_PATH))
    parser.add_argument("--input_resolution", type=str, default="ori")
    parser.add_argument("--calc_interval", type=int, default=1)
    parser.add_argument("--preprocess", type=str, default="none", choices=["none", "pu21"])
    parser.add_argument("--display_model", type=str, default=None)
    parser.add_argument("--resize", type=_str2bool, default=False)
    parser.add_argument("--resize_method", type=str, default="default")
    parser.add_argument("--n_threads", type=int, default=32)
    parser.add_argument("--vmaf_version", type=str, default=None)
    parser.add_argument("--video_temp_path", type=str, default="./data/video_temp/")
    parser.add_argument("--gpu_idx", type=int, default=0)
    parser.add_argument("--correlation", default=False, action="store_true")
    parser.set_defaults(func=_run_batch)


def _run_single(args: argparse.Namespace) -> int:
    _ensure_legacy_import_path()
    from fdim.run_fdim_single_infer import main

    main(args)
    return 0


def _run_batch(args: argparse.Namespace) -> int:
    _ensure_legacy_import_path()
    from common_utils import corr
    from fdim.run_fdim_dataset_infer import main

    results_info = main(args)
    if args.correlation and not (results_info["mos"] == 0).all():
        corr.calc_correlation(results_info, args.save_dir, args.save_name, "fdim")
    print("Finished!")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="fdim", description="FDIM video quality assessment CLI.")
    parser.add_argument("--version", action="version", version="FDIM 0.2.0")
    subparsers = parser.add_subparsers(dest="command", required=True)
    _single_parser(subparsers)
    _batch_parser(subparsers)
    return parser


def main(argv: list[str] | None = None) -> int:
    # Make VMAF download fallback use the package directory as its natural base.
    os.environ.setdefault("FDIM_HOME", str(PACKAGE_ROOT))
    parser = build_parser()
    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
