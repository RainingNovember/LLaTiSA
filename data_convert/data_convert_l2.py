# render_dataset_with_images_and_prompt.py
# -*- coding: utf-8 -*-

import os
import re
import json
import math
import argparse
import random
import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from multiprocessing import Pool, cpu_count
from tqdm import tqdm


# -----------------------------
# Timeseries extraction
# -----------------------------
_ts_k_re = re.compile(r"^timeseries_(\d+)$")


def to_matrix(timeseries):
    """
    Convert sample['timeseries'] to (T, D) float64 matrix.
    Supports:
      - [T]
      - [[T]]  (common)
      - [[T],[T],...] treated as (D,T) if it looks like few-rows-many-cols
      - (T,D) already
    """
    ts = timeseries

    # [T]
    if isinstance(ts, list) and len(ts) > 0 and not isinstance(ts[0], list):
        return np.asarray(ts, dtype=np.float64)[:, None]

    arr = np.asarray(ts, dtype=np.float64)

    if arr.ndim == 1:
        return arr[:, None]

    if arr.ndim == 2:
        # Heuristic: (D,T) when rows are small and cols are large
        if arr.shape[0] <= 8 and arr.shape[1] > arr.shape[0]:
            return arr.T
        return arr

    raise ValueError(f"Unsupported timeseries shape: {arr.shape}")


def extract_ts_matrix_and_names(item, fallback_idx=0):
    """
    Supports:
      1) 'timeseries'
      2) 'timeseries_0', 'timeseries_1', ...
    Returns (ts_matrix (T,D), col_names, sid)
    """
    sid = item.get("id", fallback_idx)

    if "timeseries" in item:
        ts = to_matrix(item["timeseries"])
        col_names = [f"Series {i+1}" for i in range(ts.shape[1])]
        return ts, col_names, sid

    pairs = []
    for k in item.keys():
        m = _ts_k_re.match(k)
        if m:
            pairs.append((int(m.group(1)), k))

    if pairs:
        pairs.sort(key=lambda x: x[0])
        series_list = [np.asarray(item[k], dtype=np.float64).flatten() for _, k in pairs]
        min_len = min(len(s) for s in series_list)
        ts = np.column_stack([s[:min_len] for s in series_list])
        col_names = [f"Series {i+1}" for i in range(ts.shape[1])]
        return ts, col_names, sid

    raise KeyError(f"No timeseries found. keys={list(item.keys())}")


# -----------------------------
# Prompt/Answer selection
# -----------------------------
def pick_prompt_answer(item: dict, idx: int, seed: int):
    """
    Randomly pick among:
      - (prompt_1, answer_1)
      - (prompt_2, answer_2)
      - (prompt_3, truth_1)

    Then final_prompt = item['2img_prompt'] + chosen_prompt
    """
    rng = random.Random(seed + idx)

    choices = []
    if "prompt_1" in item and "answer_1" in item:
        choices.append(("prompt_1", "answer_1"))
    if "prompt_2" in item and "answer_2" in item:
        choices.append(("prompt_2", "answer_2"))
    if "prompt_3" in item and "truth_1" in item:
        choices.append(("prompt_3", "truth_1"))

    if not choices:
        raise KeyError("No valid (prompt,answer) pairs found among "
                       "(prompt_1,answer_1)/(prompt_2,answer_2)/(prompt_3,truth_1).")

    pk, ak = rng.choice(choices)
    prefix = item.get("2img_prompt", "") or ""
    prompt = (item.get(pk, "") or "")
    answer = (item.get(ak, "") or "")

    return prefix + prompt, answer


# -----------------------------
# Rendering
# -----------------------------
def render_line_plots(data: np.ndarray, save_path: str, col_names: list, dpi: int = 150):
    """Line plots (supports multivariate by stacked subplots)."""
    try:
        T, D = data.shape
        fig, axes = plt.subplots(nrows=D, ncols=1, figsize=(10, max(2.5 * D, 3)), dpi=dpi)
        if D == 1:
            axes = [axes]

        for i in range(D):
            ts = data[:, i]
            axes[i].plot(ts, color="blue", linewidth=1.2)
            axes[i].set_xlabel("Index")
            axes[i].set_ylabel("Value")
            title = col_names[i] if i < len(col_names) else f"Series {i+1}"
            axes[i].set_title(title)
            axes[i].grid(True, alpha=0.3)

        plt.tight_layout()
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, format="png", dpi=dpi)
        plt.close(fig)
        return True
    except Exception as e:
        print(f"[render_line_plots] Error: {e}")
        return False


def fmt_fidelity(v: float) -> str:
    # 保真输出（不再三位小数）
    return format(float(v), ".17g")


def render_numeric_table(data: np.ndarray, save_path: str, col_names: list,
                         max_rows_per_col: int = 50, dpi: int = 150):
    """
    High-density numeric OCR image, no 3-decimal rounding.
    """
    try:
        T, D = data.shape
        num_time_cols = int(math.ceil(T / max_rows_per_col))

        # stringify values once + compute dynamic width
        str_vals = [[fmt_fidelity(data[t, d]) for d in range(D)] for t in range(T)]
        max_val_len = max((len(str_vals[t][d]) for t in range(T) for d in range(D)), default=10)

        # geometry (in "character units")
        W_IDX = len(str(T)) + 1
        W_VAL = max(14, max_val_len + 2)
        W_GAP = 3.0
        single_block_w = W_IDX + D * (W_VAL + W_GAP) + W_GAP
        BLOCK_SPACING = 3.0

        total_w = num_time_cols * single_block_w + (num_time_cols - 1) * BLOCK_SPACING
        total_h = max_rows_per_col + 2

        fig_w = total_w * (11 * 0.6 / 72.0)
        fig_h = total_h * (11 * 1.3 / 72.0)

        fig = plt.figure(figsize=(fig_w, fig_h), dpi=dpi)
        ax = fig.add_axes([0, 0, 1, 1])
        ax.set_axis_off()
        ax.set_xlim(0, total_w)
        ax.set_ylim(0, total_h)

        font_cfg = dict(fontsize=11, fontfamily="monospace", va="center")
        bg_zebra = "#f2f2f2"

        for col_idx in range(num_time_cols):
            start_t = col_idx * max_rows_per_col
            end_t = min((col_idx + 1) * max_rows_per_col, T)
            base_x = col_idx * (single_block_w + BLOCK_SPACING)

            header_y = total_h - 0.5
            line_y = total_h - 1.0

            ax.text(base_x + W_IDX - 0.5, header_y, "T", ha="left",
                    weight="bold", color="#404040", **font_cfg)

            for d in range(D):
                v_right = base_x + W_IDX + W_GAP + (d + 1) * (W_VAL + W_GAP) - W_GAP
                name = col_names[d] if d < len(col_names) else f"Series {d+1}"
                disp_name = name[:int(W_VAL) + 1]
                ax.text(v_right - 0.5, header_y, disp_name, ha="right",
                        weight="bold", color="blue", **font_cfg)

            ax.plot([base_x, base_x + single_block_w - W_GAP],
                    [line_y, line_y], color="black", linewidth=1.0)

            for t in range(start_t, end_t):
                rel_r = t - start_t
                row_y = line_y - (rel_r + 0.5)

                if rel_r % 2 == 0:
                    rect = plt.Rectangle((base_x, row_y - 0.5),
                                         single_block_w - W_GAP, 1.0,
                                         color=bg_zebra, zorder=0, ec=None)
                    ax.add_patch(rect)

                ax.text(base_x + W_IDX - 0.5, row_y, str(t),
                        ha="right", color="#606060", **font_cfg)

                for d in range(D):
                    v_right = base_x + W_IDX + W_GAP + (d + 1) * (W_VAL + W_GAP) - W_GAP
                    ax.text(v_right - 0.5, row_y, str_vals[t][d],
                            ha="right", color="black", **font_cfg)

            if col_idx < num_time_cols - 1:
                div_x = base_x + single_block_w + BLOCK_SPACING / 2
                ax.plot([div_x, div_x], [0, total_h],
                        color="#aaaaaa", linewidth=1.2, linestyle="--")

        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, format="png", dpi=dpi)
        plt.close(fig)
        return True
    except Exception as e:
        print(f"[render_numeric_table] Error: {e}")
        return False


# -----------------------------
# multiprocessing job
# -----------------------------
def process_one(args):
    (item, idx,
     plot_dir, num_dir, plot_prefix, num_prefix,
     dpi, max_rows_per_col, seed,
     use_idx_filename) = args

    # 1) extract series
    try:
        ts, col_names, sid = extract_ts_matrix_and_names(item, fallback_idx=idx)
    except Exception as e:
        print(f"[skip] idx={idx} id={item.get('id')} err={e}")
        return idx, dict(item)

    # 2) decide filename key (avoid overwrite when ids repeat)
    file_key = (idx + 1) if use_idx_filename else sid

    plot_name = f"plot_{file_key}.png"
    num_name = f"num_{file_key}.png"
    plot_path = os.path.join(plot_dir, plot_name)
    num_path = os.path.join(num_dir, num_name)

    ok1 = render_line_plots(ts, plot_path, col_names, dpi=dpi)
    ok2 = render_numeric_table(ts, num_path, col_names, max_rows_per_col=max_rows_per_col, dpi=dpi)

    if not (ok1 and ok2):
        return idx, dict(item)

    # 3) pick prompt/answer and prefix 2img_prompt
    try:
        final_prompt, final_answer = pick_prompt_answer(item, idx, seed=seed)
    except Exception as e:
        print(f"[warn] idx={idx} id={item.get('id')} prompt/answer pick failed: {e}")
        final_prompt, final_answer = item.get("2img_prompt", ""), ""

    # 4) build new item
    new_item = dict(item)
    new_item["images"] = [
        f"{plot_prefix.rstrip('/')}/{plot_name}",
        f"{num_prefix.rstrip('/')}/{num_name}",
    ]
    new_item["prompt"] = final_prompt
    new_item["answer"] = final_answer

    return idx, new_item


# -----------------------------
# main
# -----------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True, help="Input JSON (NOT jsonl), top-level list.")
    ap.add_argument("--output", required=True, help="Output JSON with images/prompt/answer added.")
    ap.add_argument("--plot_dir", required=True)
    ap.add_argument("--num_dir", required=True)
    ap.add_argument("--plot_prefix", required=True, help="URL prefix for plot images.")
    ap.add_argument("--num_prefix", required=True, help="URL prefix for numeric grid images.")
    ap.add_argument("--dpi", type=int, default=150)
    ap.add_argument("--max_rows_per_col", type=int, default=50)
    ap.add_argument("--workers", type=int, default=max(1, cpu_count() - 2))

    # sampling + id renumber
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--sample_ratio", type=float, default=1.0, help="Sample ratio (0-1). Use 0.5 for half.")
    ap.add_argument("--renumber_sampled_ids", action="store_true",
                    help="Renumber sampled items id=1..N after sampling.")

    # filename safety
    ap.add_argument("--use_idx_filename", action="store_true",
                    help="Use dataset index (idx+1) for image filenames to avoid overwriting when ids duplicate.")
    args = ap.parse_args()

    os.makedirs(args.plot_dir, exist_ok=True)
    os.makedirs(args.num_dir, exist_ok=True)

    # load
    with open(args.input, "r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise ValueError("Input JSON must be a list of samples.")

    # sample first
    if args.sample_ratio < 1.0:
        rng = random.Random(args.seed)
        N = len(data)
        k = int(N * args.sample_ratio)
        k = max(1, k)  # at least 1
        sel = set(rng.sample(range(N), k))
        data = [data[i] for i in range(N) if i in sel]
        print(f"Sampled {len(data)}/{N} items (ratio={args.sample_ratio}).")

    # renumber ids (for sampled subset only)
    if args.renumber_sampled_ids:
        for new_id, item in enumerate(data, start=1):
            item["id"] = new_id
        print(f"Renumbered sampled ids: 1..{len(data)}")

    # tasks
    tasks = [
        (item, i,
         args.plot_dir, args.num_dir, args.plot_prefix, args.num_prefix,
         args.dpi, args.max_rows_per_col, args.seed,
         args.use_idx_filename)
        for i, item in enumerate(data)
    ]

    out = [None] * len(data)
    with Pool(processes=args.workers) as pool:
        for res in tqdm(pool.imap_unordered(process_one, tasks), total=len(tasks)):
            if res is None:
                continue
            idx, new_item = res
            out[idx] = new_item

    # fill failures with originals
    final = []
    for i, item in enumerate(out):
        final.append(item if item is not None else data[i])

    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(final, f, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    main()
