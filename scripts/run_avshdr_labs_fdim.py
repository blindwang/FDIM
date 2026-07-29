from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

import pandas as pd
import torch


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT / "fdim") not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT / "fdim"))
if str(PROJECT_ROOT / "fdim" / "dist") not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT / "fdim" / "dist"))

from dist.deep_model import Model
from run_fdim import run_model
from utils.utils import load_checkpoint


LABS = ("HW", "MLS", "WHU", "ZJU")

DISPLAY_MODELS = {
    ("HW", "PQ"): "avs_hdr_set2_pq",
    ("MLS", "PQ"): "avs_hdr_set3_pq",
    ("WHU", "PQ"): "avs_hdr_set2_pq",
    ("WHU", "HLG"): "avs_hdr_set2_hlg",
    # ZJU 子集没有单独实验室显示模型，使用标准 HDR TV 口径。
    ("ZJU", "PQ"): "standard_hdr_pq_tv",
    ("ZJU", "HLG"): "standard_hdr_hlg_tv",
}


def _clear_temp_dir(path: Path) -> None:
    if not path.exists():
        return
    for child in path.iterdir():
        if child.is_dir():
            shutil.rmtree(child, ignore_errors=True)
        else:
            child.unlink(missing_ok=True)


def _parse_labs(raw_labs: list[str]) -> list[str]:
    labs: list[str] = []
    for item in raw_labs:
        labs.extend(part.strip().upper() for part in item.split(",") if part.strip())
    if not labs or "ALL" in labs:
        return list(LABS)
    unknown = sorted(set(labs) - set(LABS))
    if unknown:
        raise ValueError(f"Unknown AVS HDR labs: {unknown}. Available labs: {list(LABS)}")
    return labs


def _load_lab_data(csv_dir: Path, labs: list[str]) -> pd.DataFrame:
    frames = []
    for lab in labs:
        csv_path = csv_dir / f"AVSHDR-info-mp4-{lab}.csv"
        if not csv_path.exists():
            raise FileNotFoundError(f"Missing AVS HDR Dataset Info CSV: {csv_path}")
        df = pd.read_csv(csv_path)
        df["lab"] = lab
        df["source_csv"] = str(csv_path)
        df["display_model"] = [
            DISPLAY_MODELS[(lab, str(trc).upper())] for trc in df["trc"]
        ]
        frames.append(df)
    data = pd.concat(frames, ignore_index=True)
    data["case_key"] = (
        data["lab"].astype(str)
        + ":"
        + data["dis_name"].astype(str)
        + ":"
        + data["display_model"].astype(str)
    )
    data["compute_key"] = (
        data["ref_name"].astype(str)
        + ":"
        + data["dis_name"].astype(str)
        + ":"
        + data["display_model"].astype(str)
    )
    return data


def _compute_key(row: dict) -> str:
    return f"{row['ref_name']}:{row['dis_name']}:{row['display_model']}"


def _score_cache(rows: list[dict]) -> dict[str, tuple[float, float, float]]:
    cache = {}
    for row in rows:
        if not {"fdim_score", "vmaf_score", "deep_score"}.issubset(row):
            continue
        if pd.isna(row["fdim_score"]) or pd.isna(row["vmaf_score"]) or pd.isna(row["deep_score"]):
            continue
        compute_key = row.get("compute_key") or _compute_key(row)
        cache[compute_key] = (
            float(row["fdim_score"]),
            float(row["vmaf_score"]),
            float(row["deep_score"]),
        )
    return cache


def _result_row(row: dict, scores: tuple[float, float, float]) -> dict:
    fdim_score, vmaf_score, deep_score = scores
    return {
        "case_key": row["case_key"],
        "compute_key": row["compute_key"],
        "lab": row["lab"],
        "source_csv": row["source_csv"],
        "ref_name": row["ref_name"],
        "dis_name": row["dis_name"],
        "mos": row["mos"],
        "trc": row["trc"],
        "display_model": row["display_model"],
        "codec": row["codec"],
        "ref_width": row["ref_width"],
        "ref_height": row["ref_height"],
        "ref_bits": row["ref_bits"],
        "dis_width": row["dis_width"],
        "dis_height": row["dis_height"],
        "dis_bits": row["dis_bits"],
        "fdim_score": fdim_score,
        "vmaf_score": vmaf_score,
        "deep_score": deep_score,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--csv-dir", type=Path, default=PROJECT_ROOT / "data" / "dataset")
    parser.add_argument("--labs", nargs="+", default=["all"], help="Labs to run: all, HW, MLS, WHU, ZJU")
    parser.add_argument("--out-dir", type=Path, default=PROJECT_ROOT / "outputs" / "avshdr_labs_fdim")
    parser.add_argument("--temp-root", type=Path, required=True)
    parser.add_argument("--gpu-idx", type=int, default=0)
    parser.add_argument("--calc-interval", type=int, default=1)
    parser.add_argument("--input-resolution", type=str, default="ori")
    parser.add_argument("--n-threads", type=int, default=32)
    parser.add_argument("--keep-temp", action="store_true")
    args = parser.parse_args()

    labs = _parse_labs(args.labs)
    ref_dir = args.data_root / "ref_mp4"
    dis_dir = args.data_root / "val_mp4"
    args.out_dir.mkdir(parents=True, exist_ok=True)
    temp_dir = args.temp_root
    temp_dir.mkdir(parents=True, exist_ok=True)
    vmaf_temp_dir = Path(f"{temp_dir}_vmaf_result")
    if not args.keep_temp:
        _clear_temp_dir(temp_dir)
        _clear_temp_dir(vmaf_temp_dir)

    data = _load_lab_data(args.csv_dir, labs)
    missing = []
    for row in data.to_dict("records"):
        if not (ref_dir / row["ref_name"]).exists() or not (dis_dir / row["dis_name"]).exists():
            missing.append((row["lab"], row["ref_name"], row["dis_name"]))
    if missing:
        raise FileNotFoundError(f"Missing AVS HDR video pairs: {missing[:20]} total={len(missing)}")

    result_csv = args.out_dir / "avshdr_labs_fdim.csv"
    if result_csv.exists():
        result = pd.read_csv(result_csv)
        rows = result.to_dict("records")
        done = set(result["case_key"].astype(str))
    else:
        rows = []
        done = set()
    score_cache = _score_cache(rows)

    device = torch.device(f"cuda:{args.gpu_idx}" if torch.cuda.is_available() else "cpu")
    model = load_checkpoint(
        Model(),
        str(PROJECT_ROOT / "fdim" / "dist" / "checkpoints" / "dist_5.0.0.ckpt"),
        device,
    )

    for compute_key, group in data.groupby("compute_key", sort=False):
        group_rows = group.to_dict("records")
        pending_rows = [row for row in group_rows if row["case_key"] not in done]
        if not pending_rows:
            print(f"skip done group: {compute_key}", flush=True)
            continue

        row = group_rows[0]
        ref_path = ref_dir / row["ref_name"]
        dis_path = dis_dir / row["dis_name"]
        if compute_key in score_cache:
            scores = score_cache[compute_key]
            print(f"reuse computed score: {compute_key}", flush=True)
        else:
            try:
                # 同一 ref/dis/display_model 的 FDIM 输出一致；不同实验室 MOS 只影响落表，不重复进算。
                scores = run_model(
                    str(ref_path),
                    str(dis_path),
                    model=model,
                    device=device,
                    ref_fmt="rgb",
                    dis_fmt="rgb",
                    ref_width=int(row["ref_width"]),
                    ref_height=int(row["ref_height"]),
                    dis_width=int(row["dis_width"]),
                    dis_height=int(row["dis_height"]),
                    ref_bit_depth=int(row["ref_bits"]),
                    dis_bit_depth=int(row["dis_bits"]),
                    input_resolution=args.input_resolution,
                    calc_interval=args.calc_interval,
                    n_threads=args.n_threads,
                    video_temp_path=str(temp_dir),
                    preprocess="pu21",
                    display_model=row["display_model"],
                )[:3]
                score_cache[compute_key] = scores
            finally:
                # 4K 10-bit 中间 YUV 文件很大，每条样本后立即清理，避免临时目录被打满。
                if not args.keep_temp:
                    _clear_temp_dir(temp_dir)
                    _clear_temp_dir(vmaf_temp_dir)

        for pending in pending_rows:
            rows.append(_result_row(pending, scores))
            done.add(pending["case_key"])
        pd.DataFrame(rows).to_csv(result_csv, index=False)
        print(f"saved group: {compute_key} rows={len(pending_rows)} -> {result_csv}", flush=True)

    pd.DataFrame(rows).to_csv(result_csv, index=False)
    print(f"saved: {result_csv}", flush=True)


if __name__ == "__main__":
    main()
