import os
import json
import math
import argparse
import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from multiprocessing import Pool, cpu_count
from tqdm import tqdm
import random


# -----------------------------
# helpers
# -----------------------------
def to_matrix(timeseries):
    """
    Convert sample['timeseries'] to (T, D) float64 matrix.
    Supports:
      - [T]
      - [[T]]  (common in your sample)
      - [[T],[T],...]  interpreted as (D,T) then transpose to (T,D)
      - (T,D) already
    """
    ts = timeseries

    # [T]
    if isinstance(ts, list) and len(ts) > 0 and not isinstance(ts[0], list):
        arr = np.asarray(ts, dtype=np.float64)[:, None]
        return arr

    arr = np.asarray(ts, dtype=np.float64)

    if arr.ndim == 1:
        return arr[:, None]

    if arr.ndim == 2:
        # Heuristic: if rows are few and cols are many, treat as (D,T)
        if arr.shape[0] <= 8 and arr.shape[1] > arr.shape[0]:
            return arr.T  # (T,D)
        return arr  # (T,D)

    raise ValueError(f"Unsupported timeseries shape: {arr.shape}")


import re

_ts_k_re = re.compile(r"^timeseries_(\d+)$")


def extract_ts_matrix_and_names(item, fallback_idx=0):
    """
    Return: (ts_matrix (T,D), col_names, sid)

    Supports:
      1) item["timeseries"]  (could be [T], [[T]], [[T],[T],...], (T,D))
      2) item["timeseries_0"], item["timeseries_1"], ... (unknown count)
      3) optional: item["raw_data"] = [ts1, ts2, ...]
    """
    sid = item.get("id", fallback_idx + 1)

    # case 1: standard key
    if "timeseries" in item:
        ts = to_matrix(item["timeseries"])
        T, D = ts.shape
        col_names = [f"Series {i + 1}" for i in range(D)]
        return ts, col_names, sid

    # case 2: split keys timeseries_0..N
    pairs = []
    for k in item.keys():
        m = _ts_k_re.match(k)
        if m:
            pairs.append((int(m.group(1)), k))

    if pairs:
        pairs.sort(key=lambda x: x[0])
        series_list = []
        for _, k in pairs:
            arr = np.asarray(item[k], dtype=np.float64).flatten()
            series_list.append(arr)

        min_len = min(len(s) for s in series_list)
        ts = np.column_stack([s[:min_len] for s in series_list])  # (T,D)
        col_names = [f"Series {i + 1}" for i in range(ts.shape[1])]
        return ts, col_names, sid

    # case 3: optional compatibility with old schema
    if (
        "raw_data" in item
        and isinstance(item["raw_data"], list)
        and len(item["raw_data"]) > 0
    ):
        series_list = [
            np.asarray(s, dtype=np.float64).flatten() for s in item["raw_data"]
        ]
        min_len = min(len(s) for s in series_list)
        ts = np.column_stack([s[:min_len] for s in series_list])
        col_names = [f"Series {i + 1}" for i in range(ts.shape[1])]
        return ts, col_names, sid

    raise KeyError(
        f"No timeseries found (need 'timeseries' or 'timeseries_\\d+' or 'raw_data'). keys={list(item.keys())}"
    )


def fmt_fidelity(v: float) -> str:
    # 高保真：尽量不丢 double 精度（比 str(v) 更可控）
    return format(float(v), ".17g")


# -----------------------------
# renderers (CPU)
# -----------------------------
def render_line_plots(
    data: np.ndarray, save_path: str, col_names: list, dpi: int = 150
):
    """Line plots (supports multivariate by stacked subplots)."""
    try:
        T, D = data.shape
        n = D

        fig, axes = plt.subplots(
            nrows=n, ncols=1, figsize=(10, max(2.5 * n, 3)), dpi=dpi
        )
        if n == 1:
            axes = [axes]

        for i in range(D):
            ts = data[:, i]
            axes[i].plot(ts, color="blue", linewidth=1.2)
            axes[i].set_xlabel("Index")
            axes[i].set_ylabel("Value")
            title = col_names[i] if i < len(col_names) else f"Series {i + 1}"
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


def render_numeric_table(
    data: np.ndarray,
    save_path: str,
    col_names: list,
    max_rows_per_col: int = 50,
    dpi: int = 150,
):
    """
    High-density numeric image.
    保真：不再 {:.3f}，用 .17g。
    多变量：每列一个 series。
    """
    try:
        T, D = data.shape
        num_time_cols = int(np.ceil(T / max_rows_per_col))

        # 先把所有值转为字符串，计算最大宽度，避免溢出
        str_vals = [[fmt_fidelity(data[t, d]) for d in range(D)] for t in range(T)]
        max_val_len = (
            max(len(str_vals[t][d]) for t in range(T) for d in range(D))
            if T > 0
            else 10
        )

        W_IDX = len(str(T)) + 1
        W_VAL = max(14, max_val_len + 2)  # 动态列宽
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

            ax.text(
                base_x + W_IDX - 0.5,
                header_y,
                "T",
                ha="left",
                weight="bold",
                color="#404040",
                **font_cfg,
            )
            for d in range(D):
                v_right = base_x + W_IDX + W_GAP + (d + 1) * (W_VAL + W_GAP) - W_GAP
                disp_name = (col_names[d] if d < len(col_names) else f"Series {d + 1}")[
                    : int(W_VAL) + 1
                ]
                ax.text(
                    v_right - 0.5,
                    header_y,
                    disp_name,
                    ha="right",
                    weight="bold",
                    color="blue",
                    **font_cfg,
                )

            ax.plot(
                [base_x, base_x + single_block_w - W_GAP],
                [line_y, line_y],
                color="black",
                linewidth=1.0,
            )

            for t in range(start_t, end_t):
                rel_r = t - start_t
                row_y = line_y - (rel_r + 0.5)

                if rel_r % 2 == 0:
                    rect = plt.Rectangle(
                        (base_x, row_y - 0.5),
                        single_block_w - W_GAP,
                        1.0,
                        color=bg_zebra,
                        zorder=0,
                        ec=None,
                    )
                    ax.add_patch(rect)

                ax.text(
                    base_x + W_IDX - 0.5,
                    row_y,
                    str(t),
                    ha="right",
                    color="#606060",
                    **font_cfg,
                )

                for d in range(D):
                    v_right = base_x + W_IDX + W_GAP + (d + 1) * (W_VAL + W_GAP) - W_GAP
                    ax.text(
                        v_right - 0.5,
                        row_y,
                        str_vals[t][d],
                        ha="right",
                        color="black",
                        **font_cfg,
                    )

            if col_idx < num_time_cols - 1:
                div_x = base_x + single_block_w + BLOCK_SPACING / 2
                ax.plot(
                    [div_x, div_x],
                    [0, total_h],
                    color="#aaaaaa",
                    linewidth=1.2,
                    linestyle="--",
                )

        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, format="png", dpi=dpi)
        plt.close(fig)
        return True
    except Exception as e:
        print(f"[render_numeric_table] Error: {e}")
        return False


def pick_prompt_answer(item, idx, seed=42):
    """
    Deterministic random choice per-sample (stable under multiprocessing):
    choose (prompt_1, answer_1) or (prompt_2, answer_2),
    then prefix with 2img_prompt.
    """
    rng = random.Random(seed + idx)  # 每条样本独立随机源，保证可复现
    choose_1 = rng.random() < 0.5

    if choose_1:
        p = item.get("prompt_1", "")
        a = item.get("answer_1", "")
    else:
        p = item.get("prompt_2", "")
        a = item.get("answer_2", "")

    prefix = item.get("2img_prompt", "")
    final_prompt = (prefix or "") + (p or "")

    return final_prompt, a


# -----------------------------
# multiprocessing job
# -----------------------------
def process_one(args):
    (
        item,
        idx,
        plot_dir,
        num_dir,
        plot_prefix,
        num_prefix,
        dpi,
        max_rows_per_col,
        seed,
    ) = args
    ts, col_names, sid = extract_ts_matrix_and_names(item, fallback_idx=idx)

    plot_name = f"plot_{sid}.png"
    num_name = f"num_{sid}.png"
    plot_path = os.path.join(plot_dir, plot_name)
    num_path = os.path.join(num_dir, num_name)

    ok1 = render_line_plots(ts, plot_path, col_names, dpi=dpi)
    ok2 = render_numeric_table(
        ts, num_path, col_names, max_rows_per_col=max_rows_per_col, dpi=dpi
    )
    if not (ok1 and ok2):
        return None

    # 新增 images 字段（前缀可配置）
    new_item = dict(item)  # shallow copy
    new_item["images"] = [
        f"{plot_prefix.rstrip('/')}/{plot_name}",
        f"{num_prefix.rstrip('/')}/{num_name}",
    ]
    final_prompt, final_answer = pick_prompt_answer(item, idx, seed=seed)
    new_item["prompt"] = final_prompt
    new_item["answer"] = final_answer

    return idx, new_item


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--input",
        required=True,
        help="Input JSON (NOT jsonl). Top-level should be a list.",
    )
    ap.add_argument(
        "--output", required=True, help="Output JSON with added 'images' field."
    )
    ap.add_argument("--plot_dir", required=True)
    ap.add_argument("--num_dir", required=True)
    ap.add_argument(
        "--plot_prefix", required=True, help="URL prefix for plot images (editable)."
    )
    ap.add_argument(
        "--num_prefix", required=True, help="URL prefix for numeric images (editable)."
    )
    ap.add_argument("--dpi", type=int, default=150)
    ap.add_argument("--max_rows_per_col", type=int, default=50)
    ap.add_argument("--workers", type=int, default=max(1, cpu_count() - 2))
    ap.add_argument(
        "--seed",
        type=int,
        default=414,
        help="Random seed for picking (prompt_1/2, answer_1/2)",
    )
    ap.add_argument(
        "--sample_ratio", type=float, default=0.5, help="抽样比例，默认一半=0.5"
    )
    ap.add_argument(
        "--renumber_sampled_ids",
        action="store_true",
        help="对抽中的样本重新编号 id=1..N",
    )
    args = ap.parse_args()

    os.makedirs(args.plot_dir, exist_ok=True)
    os.makedirs(args.num_dir, exist_ok=True)
    import random

    random.seed(args.seed)

    import random

    with open(args.input, "r", encoding="utf-8") as f:
        data = json.load(f)
    assert isinstance(data, list), "Input JSON must be a list."

    rng = random.Random(args.seed)

    N = len(data)
    k = int(N * args.sample_ratio)
    k = max(1, k)  # 至少取 1 条（你也可改成允许 0）

    # 随机抽取一半（不放回）
    selected_indices = rng.sample(range(N), k)
    selected_indices_set = set(selected_indices)

    # 生成子集（保持抽样顺序稳定：按原顺序保留）
    sampled = [data[i] for i in range(N) if i in selected_indices_set]

    # 可选：对抽中的一半样本重新编号 id
    if args.renumber_sampled_ids:
        for new_id, item in enumerate(sampled, start=1):
            item["id"] = new_id

    data = sampled
    print(
        f"Sampled {len(data)}/{N} items (ratio={args.sample_ratio}). Renumber={args.renumber_sampled_ids}"
    )

    tasks = [
        (
            item,
            i,
            args.plot_dir,
            args.num_dir,
            args.plot_prefix,
            args.num_prefix,
            args.dpi,
            args.max_rows_per_col,
            args.seed,
        )
        for i, item in enumerate(data)
    ]

    out = [None] * len(data)
    with Pool(processes=args.workers) as pool:
        for res in tqdm(pool.imap_unordered(process_one, tasks), total=len(tasks)):
            if res is None:
                continue
            idx, new_item = res
            out[idx] = new_item

    # 若有失败，保留原条目或过滤（这里选择保留原条目并不加 images）
    final = []
    for i, item in enumerate(out):
        if item is None:
            final.append(data[i])
        else:
            final.append(item)

    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(final, f, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    main()
