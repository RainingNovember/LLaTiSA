  # Copyright 2024 Tsinghua University and ByteDance.
#
# Licensed under the MIT License (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://opensource.org/license/mit
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import numpy as np
import random
from tqdm import tqdm
import re
import json
from typing import *
from ts_generator.trend_utils import generate_random_points, generate_trend_prompt, generate_trend_curve, generate_trend_list
from ts_generator.local_changes import generate_local_chars
from ts_generator.change_utils import generate_ts_change
import yaml


# Config
ENABLE_MULTIPLE_TREND = yaml.safe_load(open("config/datagen_config.yaml"))["enable_multiple_trend"]  # Enable or disable multiple trend types
ENABLE_MULTIPLE_SEASONAL = yaml.safe_load(open("config/datagen_config.yaml"))["enable_multiple_seasonal"]  # Enable or disable multiple seasonal types
ENABLE_MULTIPLE_NOISE = yaml.safe_load(open("config/datagen_config.yaml"))["enable_multiple_noise"]  # Enable or disable multiple noise types

# All Config for TS Attributes
# Notes
# 1. Seaonal and Frequency can be combined (e.g., high frequency-sin periodic fluctuation, so there should be 7 types of seasonal attributes)
# 2. We implemented 2 types of noise types (e.g., random noise, sin noise), see the codes below for more information
# 3. The value following the overall attribute is the probability of the attribute to be selected
# All Attribute Set

all_attribute_set = {
    "overall_attribute": {
        "seasonal": {
            "no periodic fluctuation": 0.25,
            "sin periodic fluctuation": 0.25,
            "square periodic fluctuation": 0.25,
            "triangle periodic fluctuation": 0.25
        },
        "trend": {
            "decrease": 0.3,
            "increase": 0.3,
            "keep steady": 0.1,
            "multiple": 0.3
        },
        "frequency": {
            "high frequency": 0.5,
            "low frequency": 0.5
        },
        "noise": {
            "noisy": 0.2,
            "almost no noise": 0.8
        }
    },
    "change": {
        "shake": 2,
        "upward spike": 12,
        "downward spike": 10,
        "continuous upward spike": 3,
        "continuous downward spike": 3,
        "upward convex": 2,
        "downward conv se": 10,
        "sudden decrease": 10,
        "rapid rise followed by slow decline": 2,
        "slow rise followed by rapid decline": 2,
        "rapid decline followed by slow rise": 2,
        "slow decline followed by rapid rise": 2,
        "decrease after upward spike": 1,
        "increase after downward spike": 1,
        "increase after upward spike": 1,
        "decrease after downward spike": 1,
        "wide upward spike": 2,
        "wide downward spike": 2
    }
}

# 新增互斥 annotation 分组映射表
# 映射原始trend / seasonal到互斥类别
ANNOTATION_MAP = {
    "trend": {
        "increase": ["increase"],
        "decrease": ["decrease"],
        "steady": ["keep steady"],
        "multiple": ["multiple"],
    },

    "seasonal": {
        "none": ["no periodic fluctuation"],
        "sin": ["sin periodic fluctuation"],
        "square": ["square periodic fluctuation"],
        "triangle": ["triangle periodic fluctuation"],
        "periodic": ["periodic fluctuation"]
    }
}

# Local原始->互斥类别映射表
# ====== 新 LOCAL_MUTUALLY_EXCLUSIVE_MAP（只保留 spike）======
LOCAL_MUTUALLY_EXCLUSIVE_MAP = {
    "upward spike": "spike_up",
    "continuous upward spike": "spike_up",
    "wide upward spike": "spike_up",
    "downward spike": "spike_down",
    "continuous downward spike": "spike_down",
    "wide downward spike": "spike_down",
    # 转折类
    "rapid rise followed by slow decline": "turn_up_down",
    "slow rise followed by rapid decline": "turn_up_down",
    "rapid decline followed by slow rise": "turn_down_up",
    "slow decline followed by rapid rise": "turn_down_up",
    # 其它全部归为none
}
# ==========================================================

def determine_exclusive_local(local_events):
    local_counts = {"spike_up": 0, "spike_down": 0}
    for ev in local_events:
        mapped = LOCAL_MUTUALLY_EXCLUSIVE_MAP.get(ev["type"], "none")
        if mapped in local_counts:
            local_counts[mapped] += 1
    if local_counts["spike_up"] == 0 and local_counts["spike_down"] == 0:
        return "none"
    return "spike_up" if local_counts["spike_up"] >= local_counts["spike_down"] else "spike_down"

def build_count_local_detail(local_events):
    if not local_events:
        return "This time series has no significant upward spikes or downward spikes."
    counts = {}
    for ev in local_events:
        exclusive_type = LOCAL_MUTUALLY_EXCLUSIVE_MAP.get(ev["type"], "none")
        counts[exclusive_type] = counts.get(exclusive_type, 0) + 1
    detail_parts = []
    for ev_type, num in counts.items():
        if num > 0 and ev_type != "none":
            detail_parts.append(f"{num} significant {ev_type}")
    return ", ".join(detail_parts) if detail_parts else "No significant local events."

def determine_trend_from_series(y, ratio_threshold=0.1, slope_threshold=0.02):
    """
    组合中点均值差法 + 线性回归斜率法
    ratio_threshold: 半段均值差占总波动的比例阈值
    slope_threshold: 线性回归斜率占总波动的比例阈值
    """
    total_range = np.max(y) - np.min(y)
    if total_range < 1e-8:
        return "steady"

    # 前半段 & 后半段均值
    mid = len(y) // 2
    mean_diff = np.mean(y[mid:]) - np.mean(y[:mid])
    mean_diff_ratio = mean_diff / total_range

    # 全程线性回归
    x = np.arange(len(y))
    slope, _ = np.polyfit(x, y, 1)
    slope_ratio = slope * len(y) / total_range

    if abs(mean_diff_ratio) >= ratio_threshold and abs(slope_ratio) >= slope_threshold:
        if mean_diff_ratio > 0:
            return "increase"
        else:
            return "decrease"
    else:
        return "steady"

def remap_to_annotation(attribute_pool, y):
    ann = {"trend": None, "seasonal": None, "local": None, "numerical": None}

    # trend
    real_trend = determine_trend_from_series(y)
    ann["trend"] = real_trend

    # seasonal
    s_type = attribute_pool["seasonal"]["type"].strip().lower()
    ann["seasonal"] = "other"
    for k, v in ANNOTATION_MAP["seasonal"].items():
        if s_type in [vv.lower() for vv in v]:
            ann["seasonal"] = k
            break

    # 保存周期到 annotation_mapped
    if ann["seasonal"] in ("sin", "triangle", "square") and "period" in attribute_pool.get("frequency", {}):
        ann["seasonal_period"] = attribute_pool["frequency"]["period"]
    else:
        ann["seasonal_period"] = None

    # ===== spike / local =====
    if ann["seasonal"] == "triangle":
        # triangle → 清空 spike 相关
        ann["local"] = "none"
        ann["numerical"] = {
            "max_point": {"pos": int(np.argmax(y)) + 1, "value": round(float(np.max(y)), 4)},
            "min_point": {"pos": int(np.argmin(y)) + 1, "value": round(float(np.min(y)), 4)},
            "point_compare": "equal" if y[-1] == y[0] else ("up" if y[-1] > y[0] else "down"),
            "count_local": {"count": 0, "detail": "No significant local events."}
        }
        ann["spike_details"] = []
    else:
        # 正常统计 spike
        spike_events = [ev for ev in attribute_pool["local"]
                        if LOCAL_MUTUALLY_EXCLUSIVE_MAP.get(ev["type"], "none") != "none"]
        spike_count = len(spike_events)
        ann["local"] = "none"
        if spike_count > 0:
            local_counts = {"spike_up": 0, "spike_down": 0}
            for ev in spike_events:
                mapped = LOCAL_MUTUALLY_EXCLUSIVE_MAP.get(ev["type"], "none")
                local_counts[mapped] += 1
            ann["local"] = "spike_up" if local_counts["spike_up"] >= local_counts["spike_down"] else "spike_down"

        start_val = y[0]
        end_val = y[-1]
        point_cmp = "equal"
        if end_val > start_val:
            point_cmp = "up"
        elif end_val < start_val:
            point_cmp = "down"

        ann["numerical"] = {
            "max_point": {"pos": int(np.argmax(y)) + 1, "value": round(float(np.max(y)), 4)},
            "min_point": {"pos": int(np.argmin(y)) + 1, "value": round(float(np.min(y)), 4)},
            "point_compare": point_cmp,
            "count_local": {"count": spike_count, "detail": build_count_local_detail(spike_events)}
        }

        spike_details = []
        for ev in attribute_pool["local"]:
            mapped_type = LOCAL_MUTUALLY_EXCLUSIVE_MAP.get(ev["type"], "none")
            if mapped_type in ("spike_up", "spike_down"):
                spike_details.append({
                    "type": mapped_type,
                    "start": ev.get("position_start") + 1 if ev.get("position_start") is not None else None,
                    "end": ev.get("position_end") + 1 if ev.get("position_end") is not None else None,
                    "amplitude": ev.get("amplitude")
                })
        ann["spike_details"] = spike_details

    # turning points
    turning_types_map = {
        "rapid rise followed by slow decline": "turn_up_down",
        "slow rise followed by rapid decline": "turn_up_down",
        "rapid decline followed by slow rise": "turn_down_up",
        "slow decline followed by rapid rise": "turn_down_up"
    }
    turning_points = []
    for ev in attribute_pool["local"]:
        if ev["type"] in turning_types_map:
            turning_points.append({
                "type": turning_types_map[ev["type"]],
                "start": ev.get("position_start") + 1 if ev.get("position_start") is not None else None,
                "end": ev.get("position_end") + 1 if ev.get("position_end") is not None else None
            })

    ann["turning_points"] = {
        "count": len(turning_points),
        "points": turning_points
    }
    return ann

def mask_to_intervals(mask: np.ndarray) -> List[List[int]]:
    intervals = []
    L = len(mask)
    i = 0
    while i < L:
        if mask[i]:
            s = i
            while i + 1 < L and mask[i + 1]:
                i += 1
            e = i
            intervals.append([int(s), int(e)])
        i += 1
    return intervals

def merge_and_sanitize_intervals(intervals: List[List[int]], L: int, merge_adjacent: bool = True, min_len: int = 1) -> List[List[int]]:
    if not intervals:
        return []

    ints = []
    for s, e in intervals:
        s = max(0, min(int(s), L - 1))
        e = max(0, min(int(e), L - 1))
        if s > e:
            s, e = e, s
        if (e - s + 1) >= min_len:
            ints.append([s, e])
    if not ints:
        return []
    ints.sort(key=lambda x: (x[0], x[1]))

    merged = [ints[0]]
    for s, e in ints[1:]:
        ps, pe = merged[-1]
        if s <= pe + (1 if merge_adjacent else 0):
            merged[-1][1] = max(pe, e)
        else:
            merged.append([s, e])
    return merged

def generate_random_attributes(overall_attribute: Dict[str, Dict[str, float]] = all_attribute_set["overall_attribute"], change_attribute: Dict[str, float] = all_attribute_set["change"], change_positions: Optional[List[Tuple[Optional[int], Optional[float]]]] = None, seq_len: int = 512):
    if change_positions is None:
        change_positions = [(None, None) for _ in range(random.randint(0, 3))]
    attribute_pool = {}

    if seq_len >= 24:
        attribute_pool["seasonal"] = {
            "type": np.random.choice(list(overall_attribute['seasonal']), p=np.array(list(overall_attribute['seasonal'].values()))/sum(list(overall_attribute['seasonal'].values())))
        }
    else:
        # The length is too short, we can only use no periodic fluctuation
        attribute_pool["seasonal"] = {
            "type": "no periodic fluctuation"
        }

    trend_candidates = overall_attribute['trend'].copy()
    trend_candidates.pop("multiple", None)
    if not ENABLE_MULTIPLE_TREND and "multiple" in trend_candidates:
        trend_candidates.pop("multiple")
    trend_char = np.random.choice(list(trend_candidates), p=np.array(list(trend_candidates.values()))/sum(list(trend_candidates.values())))
    
    attribute_pool["trend"] = {
        "type": trend_char
    }

    num_local_chars = len(change_positions)

    # If the length is too short (<=64 and the trend is multiple, we should omit some types of local changes)
    if seq_len <= 64 and trend_char == "multiple":
        change_attribute = change_attribute.copy()
        change_attribute.pop("upward convex", None)
        change_attribute.pop("downward convex", None)
        change_attribute.pop("rapid rise followed by slow decline", None)
        change_attribute.pop("slow rise followed by rapid decline", None)
        change_attribute.pop("rapid decline followed by slow rise", None)
        change_attribute.pop("slow decline followed by rapid rise", None)
        change_attribute.pop("decrease after upward spike", None)
        change_attribute.pop("increase after downward spike", None)
        change_attribute.pop("increase after upward spike", None)
        change_attribute.pop("decrease after downward spike", None)
        change_attribute.pop("wide upward spike", None)
        change_attribute.pop("wide downward spike", None)
    if seq_len <= 8:
        change_attribute.pop("shake", None)
        change_attribute.pop("sudden increase", None)
        change_attribute.pop("sudden decrease", None)

    # ====== 新增过滤步骤 ======
    change_attribute = change_attribute.copy()
    if "sudden increase" in change_attribute:
        change_attribute.pop("sudden increase")
    if "sudden decrease" in change_attribute:
        change_attribute.pop("sudden decrease")
    # =========================


    SPIKE_TYPES = ["upward spike", "continuous upward spike", "wide upward spike",
                "downward spike", "continuous downward spike", "wide downward spike"]


    change_attribute_spike = {k: v for k, v in change_attribute.items() if k in SPIKE_TYPES}


    seasonal_type = attribute_pool["seasonal"]["type"].lower()
    if "triangle" in seasonal_type:

        local_chars = []
    else:

        local_chars = list(np.random.choice(
            list(change_attribute_spike),
            size=num_local_chars,
            p=np.array(list(change_attribute_spike.values())) / sum(list(change_attribute_spike.values()))
        ))


    attribute_pool["local"] = []
    for char in local_chars:
        local_position, local_amplitude = change_positions.pop()
        attribute_pool["local"].append({
            "type": char,
            "position_start": local_position,
            "amplitude": local_amplitude
        })


    if 'no periodic fluctuation' not in attribute_pool["seasonal"]['type'] and seq_len >= 24:
        # If the seq_len is too short (<64, we can only use low frequency)
        if seq_len <= 64:
            attribute_pool["frequency"] = {'type': 'low frequency'}
        else:
            attribute_pool["frequency"] = {'type': np.random.choice(list(overall_attribute['frequency']), p=np.array(list(overall_attribute['frequency'].values()))/sum(list(overall_attribute['frequency'].values())))}
    else:
        attribute_pool["frequency"] = {'type': 'no periodicity'}
    
    # If the seq_len is too short (<=32, the noise is of no meaning, we just use no noise) if period not in attribute_pool['frequency']
    if seq_len <= 32:
        attribute_pool["noise"] = {'type': 'almost no noise'}
    else:
        attribute_pool["noise"] = {'type': np.random.choice(list(overall_attribute['noise']), p=np.array(list(overall_attribute['noise'].values()))/sum(list(overall_attribute['noise'].values())))}
    attribute_pool["seq_len"] = seq_len

    return attribute_pool

def generate_controlled_attributes(attribute_set, change_positions: Optional[List[Tuple[Optional[int], Optional[float]]]] = None, seq_len: int = 512):
    if change_positions is None:
        change_positions = [(None, None) for _ in range(random.randint(0, 3))]
    description = {}

    seasonal_p = [all_attribute_set['overall_attribute']['seasonal'][i] for i in attribute_set['seasonal']['attributes']]
    description["seasonal"] = {
        "type": np.random.choice(list(attribute_set['seasonal']['attributes']), p=np.array(seasonal_p)/sum(seasonal_p)),
        "amplitude": random.uniform(attribute_set['seasonal']['amplitude']['min'], attribute_set['seasonal']['amplitude']['max'])
    }
    
    if not ENABLE_MULTIPLE_TREND:
        if "multiple" in attribute_set['trend']['attributes']:
            attribute_set['trend']['attributes'].remove("multiple")
            if len(attribute_set['trend']['attributes']) == 0:
                attribute_set['trend']['attributes'] = ['increase', 'decrease', 'keep steady']
    trend_p = [all_attribute_set['overall_attribute']['trend'][i] for i in attribute_set['trend']['attributes']]
    description["trend"] = {
        "type": np.random.choice(list(attribute_set['trend']['attributes']), p=np.array(trend_p)/sum(trend_p)),
        "start": random.uniform(attribute_set['trend']['start']['min'], attribute_set['trend']['start']['max']),
        "amplitude": random.uniform(attribute_set['trend']['amplitude']['min'], attribute_set['trend']['amplitude']['max'])
    }

    num_local_chars = len(change_positions)
    change_attrs_filtered = [a for a in attribute_set['change']['attributes'] 
                            if a not in ["sudden increase", "sudden decrease"]]
    change_p = [all_attribute_set['change'][i] for i in change_attrs_filtered]
    local_chars = list(np.random.choice(change_attrs_filtered, 
                                        size=num_local_chars, 
                                        p=np.array(change_p)/sum(change_p)))
    
    description["local"] = []
    for char in local_chars:
        description["local"].append({
            "type": char,
            "position_start": None,
            "amplitude": random.uniform(attribute_set['change']['amplitude']['min'], attribute_set['change']['amplitude']['max'])
        })

    if 'no periodic fluctuation' not in description["seasonal"]['type']:
        # Generate period and then determine type
        period = max(random.uniform(attribute_set['seasonal']['period']['min'], attribute_set['seasonal']['period']['max']), 6)
        if period < seq_len // 8:
            description["frequency"] = {'type': 'high frequency', 'period': round(period, 1)}
        else:
            description["frequency"] = {'type': 'low frequency', 'period': round(period, 1)}
    else:
        description["frequency"] = {'type': 'no periodicity'}
        
    noise_p = [all_attribute_set['overall_attribute']['noise'][i] for i in attribute_set['noise']['attributes']]
    description["noise"] = {'type': np.random.choice(list(attribute_set['noise']['attributes']), p=np.array(noise_p)/sum(noise_p))}
    description["seq_len"] = seq_len

    return description

def generate_seasonal_wave(period, amplitude_list, split_points, seq_len, wave_type=None):
    # Time array
    t = np.linspace(0, seq_len, seq_len)
    data = np.zeros(seq_len)
    base_frequency = 1 / period

    # Amplitude series
    amplitude_series = np.zeros(seq_len)
    for i in range(len(amplitude_list)):
        amplitude_series[split_points[i]:split_points[i + 1]] = amplitude_list[i]

    # Smoothing amplitude_series with window size
    sliding_window = 5
    for i in range(seq_len - sliding_window):
        amplitude_series[i + sliding_window // 2] = np.mean(amplitude_series[i:i + sliding_window])

    if wave_type is None:
        wave_type = str(np.random.choice(['sin', 'square', 'triangle'], p=[0.7, 0.15, 0.15]))

    if wave_type == 'sin':
        num_harmonics = np.random.randint(1, max(2, min(period // 6, 10)))
        for n in range(1, num_harmonics + 1):
            phase = np.random.uniform(0, 2 * np.pi)
            harmonic_amplitude = amplitude_series / n * (1 + np.random.uniform(0, 0.05) * np.sin(np.random.uniform(1, 3) * np.pi * t / seq_len + np.random.uniform(0, 2 * np.pi)))
            data += harmonic_amplitude * np.sin(2 * np.pi * base_frequency * n * t + phase)
    elif wave_type == 'square':
        start = np.random.uniform(0, 0.2)
        duration = np.random.uniform(0.4, 0.7)
        for i in range(seq_len):
            cycle_pos = (t[i] % period) / period
            if start <= cycle_pos < start + duration:
                data[i] = amplitude_series[i]
            else:
                data[i] = 0.0
    else:
        start = np.random.uniform(0, 0.3)
        duration = np.random.uniform(0.1, 0.6)
        end = start + duration
        for i in range(seq_len):
            cycle_pos = (t[i] % period) / period
            if start <= cycle_pos < end:
                if cycle_pos < (start + end) / 2:
                    data[i] = amplitude_series[i] * 2 * (cycle_pos - start) / duration
                else:
                    data[i] = amplitude_series[i] * 2 * (end - cycle_pos) / duration
            else:
                data[i] = 0.0

    # normalize to amplitude
    data = data / (data.max() - data.min() + 1e-7) * max(amplitude_list)
    data -= np.mean(data)

    return data

def generate_sin_noise(amplitude, seq_len):
    # Time array
    t = np.linspace(0, seq_len, seq_len)
    data = np.zeros(seq_len)

    num_harmonics = 200
    for n in range(1, num_harmonics + 1):
        phase = np.random.uniform(0, 2 * np.pi)
        cur_freq = np.random.uniform(50 / seq_len, 200 / seq_len)
        data += np.sin(cur_freq * t + phase) * np.random.uniform(0.3, 1.0)

    # normalize to amplitude
    data = data / (data.max() - data.min() + 1e-7) * amplitude
    data -= np.mean(data)

    return data

def generate_noise(attribute_pool, y, overall_amplitude, seq_len):
    max_change = np.abs(np.max(y) - np.min(y))
    noise_level = attribute_pool["noise"]['type']
    if noise_level == "noisy":
        if random.random() > 0.5 and max_change > overall_amplitude / 2 and attribute_pool["frequency"]['type'] == "no periodicity":
            # Generate a sin type noise
            noise = generate_sin_noise(0.2 * overall_amplitude, seq_len)
            noise += np.random.normal(0, 0.03 * overall_amplitude, seq_len)
            std = round(float(np.std(noise)), 3)
            attribute_pool["noise"]["detail"] = f"There is a irregular fluctuating noise, indicating a noisy curve: "
        else:
            # Generate random type noise
            std = np.random.uniform(0.03, 0.15) * overall_amplitude
            noise = np.random.normal(0, std, seq_len)
            attribute_pool["noise"]["detail"] = f"There is a random noise, indicating a noisy curve: "

        # Apply noise segments
        num_noise_segments = 1
        if ENABLE_MULTIPLE_NOISE:
            num_noise_segments = random.randint(1, 3)
        
            # Choose segments to apply noise
            attribute_pool["noise"]["segments"] = []
            noise_segments = generate_split_points(seq_len, num_noise_segments)
            for i in range(num_noise_segments):
                noise_start = noise_segments[i]
                noise_end = noise_segments[i + 1]
                noise_std_amp = np.random.uniform(0.1, 5.0)
                noise[noise_start:noise_end] *= noise_std_amp
                attribute_pool["noise"]["segments"].append({
                    "position_start": noise_start,
                    "position_end": noise_end,
                    "amplitude": round(noise_std_amp * std, 2),
                    "description": f"the noise std is {noise_std_amp * std:.2f} between point {noise_start} and point {noise_end}"
                })
                attribute_pool["noise"]["detail"] += f"the noise std is {noise_std_amp * std:.2f} between point {noise_start} and point {noise_end}, "
            attribute_pool["noise"]["detail"] = attribute_pool["noise"]["detail"][:-2] + ". "
        else:
            noise_std_amp = np.random.uniform(0.1, 5.0)
            noise *= noise_std_amp
            attribute_pool["noise"]["std"] = round(noise_std_amp * std, 2)
            attribute_pool["noise"]["detail"] = f"The overall noise standard deviation is around {noise_std_amp * std:.2f}, indicating a large noisy curve."
    elif noise_level == "almost no noise":
        if max_change > overall_amplitude / 2:
            std = np.random.uniform(0.0, 0.001) * overall_amplitude
        else:
            std = 0.0
        noise = np.random.normal(0, std, seq_len)
        attribute_pool["noise"]["std"] = round(std, 3)
        attribute_pool["noise"]["detail"] = f"The overall noise standard deviation is around {std:.2f}, very small compared the overall change of the curve. The curve is overall smooth with almost no noise. "
    
    return noise

def generate_seasonal(attribute_pool, overall_amplitude, seq_len):
        y = np.zeros(seq_len)
        if "no period" not in attribute_pool["seasonal"]['type']:
            if attribute_pool["seasonal"]['type'] == "periodic fluctuation":
                wave_type = None
            else:
                wave_type = attribute_pool["seasonal"]["type"].split(" ")[0]
            if 'amplitude' not in attribute_pool['seasonal']:
                # Many periods of seasonal amplitudes
                num_seasonal = 1
                if ENABLE_MULTIPLE_SEASONAL:
                    num_seasonal = random.randint(1, 3)
                amp = []
                for _ in range(num_seasonal):
                    amp.append(random.uniform(1.0, 2.0) * overall_amplitude)
                split_points = generate_split_points(seq_len, num_seasonal)
            else:
                # Only one period of seasonal amplitudes
                amp = [attribute_pool['seasonal']['amplitude']]
                split_points = [0, seq_len]
            y += generate_seasonal_wave(attribute_pool['frequency']['period'], amp, split_points, seq_len, wave_type)

            attribute_pool["seasonal"]['detail'] = f"The time series is showing {attribute_pool['seasonal']['type']}: "
            attribute_pool["seasonal"]["segments"] = []
            for i in range(len(amp)):
                attribute_pool["seasonal"]["segments"].append({
                    "amplitude": round(amp[i], 2),
                    "position_start": split_points[i],
                    "position_end": split_points[i + 1],
                    "description": f"the amplitude of the periodic fluctuation is {amp[i]:.1f} between point {split_points[i]} and point {split_points[i + 1]}"
                })
                attribute_pool["seasonal"]['detail'] += f"the amplitude of the periodic fluctuation is {amp[i]:.1f} between point {split_points[i]} and point {split_points[i + 1]}, "
            attribute_pool["seasonal"]['detail'] = attribute_pool["seasonal"]['detail'][:-2] + ". "
        elif attribute_pool["seasonal"]['type'] == "no periodic fluctuation":
            y += 0.0
            attribute_pool["seasonal"]["segments"] = []
            attribute_pool["seasonal"]['detail'] = f"No periodic fluctuations observed, showing {attribute_pool['seasonal']['type']}. "
        return y

def generate_trend(attribute_pool, y, overall_amplitude, overall_bias, seq_len):
    # Apply trend attribute
    trend = attribute_pool["trend"]["type"]

    if 'amplitude' in attribute_pool['trend']:
        amplitude = attribute_pool['trend']['amplitude']
    else:
        amplitude = random.uniform(0.8, 3.0) * overall_amplitude
    if 'start' in attribute_pool['trend']:
        bias = attribute_pool['trend']['start']
    else:
        bias = overall_bias

    if trend == "decrease":
        cur_value = generate_ts_change(seq_len, -amplitude, add_random_noise=False) + bias
        y += cur_value
        attribute_pool["trend"]["detail"] = f"From the perspective of the slope, the overall trend is decreasing. "
        attribute_pool["trend"]["trend_list"] = [("decrease", 0, seq_len - 1)]
    elif trend == "increase":
        cur_value = generate_ts_change(seq_len, amplitude, add_random_noise=False) + bias
        y += cur_value
        attribute_pool["trend"]["detail"] = f"From the perspective of the slope, the overall trend is increasing. "
        attribute_pool["trend"]["trend_list"] = [("increase", 0, seq_len - 1)]
    elif trend == "multiple":
        # Ensure the generated trend has more than one type
        while True:
            points = generate_random_points(seq_len=seq_len)[0]
            if len(generate_trend_list(points, seq_len)) > 1:
                break
        trend_ts = generate_trend_curve(seq_len=seq_len, points=points)[1]
        y += trend_ts * amplitude
        attribute_pool["trend"]["detail"] = "From the perspective of the slope, the overall trend contains multiple different segments: " + generate_trend_prompt(points)
        attribute_pool["trend"]["trend_list"] = generate_trend_list(points, seq_len)
    elif trend == "keep steady":
        y += bias
        attribute_pool["trend"]["detail"] = f"From the perspective of the slope, the overall trend is steady. "
        attribute_pool["trend"]["trend_list"] = [("keep steady", 0, seq_len - 1)]

    # Find increase or decrease in local char
    local_phase_change = [i['type'] for i in attribute_pool["local"] if 'increase' in i['type'] or 'decrease' in i['type']]
    if len(local_phase_change):
        attribute_pool["trend"]["detail"] += f"However, local phase changes were observed, including: {', '.join(local_phase_change)}. "
    attribute_pool["trend"]["start"] = round(float(y[0]), 2)
    attribute_pool["trend"]["amplitude"] = round(float(y[-1] - y[0]), 2)
    attribute_pool["trend"]["detail"] += f"The value of time series starts from around {float(y[0]):.2f} and ends at around {float(y[-1]):.2f}, with an overall amplitude of {float(y[-1] - y[0]):.2f}. "
    return y

def generate_split_points(seq_len: int, num_segments: int) -> list:
    if num_segments < 1:
        raise ValueError("Number of segments must be at least 1.")
    if seq_len < num_segments:
        raise ValueError("Sequence length must be at least equal to the number of segments.")

    min_segment_len = seq_len / num_segments / 2  # Minimum segment length
    split_points = [0]  # Start with the first point
    
    for _ in range(num_segments - 1):
        # Determine the valid range for the next split point
        min_point = split_points[-1] + min_segment_len
        max_point = seq_len - (num_segments - len(split_points)) * min_segment_len
        if min_point >= max_point:
            raise ValueError("Cannot generate split points satisfying the constraints.")
        
        # Randomly select a split point within the valid range
        split_points.append(int(random.uniform(min_point, max_point)))
    split_points.append(seq_len)
    
    return split_points

def generate_time_series(attribute_pool, seq_len=512):
    """
    Generate a time series based on the given attribute pool and sequence length.
    """
    # Adapt to legacy behavior
    if not ENABLE_MULTIPLE_TREND:
        # (Step 1) Remove seasonal type
        if "no period" not in attribute_pool["seasonal"]['type']:
            attribute_pool["seasonal"]["type"] = "periodic fluctuation"

        # (Step 2) Remove multiple trend
        if attribute_pool["trend"]["type"] == "multiple":
            attribute_pool["trend"]["type"] = random.choice(["increase", "decrease", "keep steady"])

    # Generate base arrays
    x = np.linspace(0, 10 * np.pi, seq_len)
    y = np.zeros_like(x)

    # frequency → period
    period = seq_len
    if "frequency" in attribute_pool:
        if period not in attribute_pool["frequency"]:
            if attribute_pool["frequency"]['type'] == "high frequency":
                max_period = seq_len // 8
                min_period = max(seq_len // 16, 6)
                period = random.uniform(min_period, max_period)
            elif attribute_pool["frequency"]['type'] == "low frequency":
                max_period = seq_len // 3
                min_period = max(seq_len // 8, 6)
                period = random.uniform(min_period, max_period)

        if attribute_pool["frequency"]['type'] == "no periodicity":
            attribute_pool["frequency"]['period'] = 0.0
            attribute_pool["frequency"]['detail'] = "No significant periodic fluctuations observed, overall almost no periodicity. "
        else:
            attribute_pool["frequency"]['period'] = round(period, 1)
            attribute_pool["frequency"]['detail'] = f"Each fluctuation period is approximately {period:.1f} points, thus the overall fluctuation is {attribute_pool['frequency']['type']}. "

    # Amplitude & bias
    if 'overall_amplitude' in attribute_pool and 'overall_bias' in attribute_pool:
        overall_amplitude = attribute_pool['overall_amplitude']
        overall_bias = attribute_pool['overall_bias']
    else:
        overall_amplitude_e = np.random.choice([-2, -1, 0, 1, 2, 3, 4, 5, 6, 7],
                                               p=[0.1, 0.2, 0.2, 0.3, 0.1, 0.04, 0.03, 0.02, 0.008, 0.002])
        overall_amplitude = round(np.random.uniform(10.0 ** (overall_amplitude_e - 1),
                                                    10.0 ** (overall_amplitude_e + 1)), 2)
        overall_bias = round(np.random.uniform(-(10.0 ** (overall_amplitude_e + 1)),
                                               10.0 ** (overall_amplitude_e + 1)), 2)
        attribute_pool['overall_amplitude'] = round(overall_amplitude, 2)
        attribute_pool['overall_bias'] = round(overall_bias, 2)

    # Seasonal
    y += generate_seasonal(attribute_pool, overall_amplitude, seq_len)

    seasonal_type = attribute_pool["seasonal"]["type"].lower()
    if "triangle" in seasonal_type:
        attribute_pool["local"] = []
    else:
        y += generate_local_chars(attribute_pool, overall_amplitude, seq_len)

    # Trend
    y = generate_trend(attribute_pool, y, overall_amplitude, overall_bias, seq_len)

    for local_char in attribute_pool["local"]:
        pattern = re.compile(r'<\|(\d+)\|>')
        def replacer(match):
            n = int(match.group(1))
            if n < 0 or n >= seq_len:
                print(local_char["detail"], seq_len)
            return f"{y[n]:.2f}"
        local_char['detail'] = pattern.sub(replacer, local_char['detail'])

    # Noise
    y += generate_noise(attribute_pool, y, overall_amplitude, seq_len)

    # Statistics
    attribute_pool["statistics"] = {
        "mean": round(float(np.mean(y)), 2),
        "std": round(float(np.std(y)), 2),
        "max": round(float(np.max(y)), 2),
        "min": round(float(np.min(y)), 2),
        "max_pos": int(np.argmax(y)),
        "min_pos": int(np.argmin(y))
    }
    attribute_pool["seq_len"] = seq_len

    # Local footprints
    L = seq_len
    for lc in attribute_pool.get("local", []):
        if "position_start" in lc and "position_end" in lc:
            s = int(lc["position_start"])
            e = int(lc["position_end"])
            lc["footprint"] = [max(0, min(s, L - 1)), max(0, min(e, L - 1))]
    
    # Trend segments
    trend_list = attribute_pool.get("trend", {}).get("trend_list", [])
    trend_segments = []
    for t in trend_list:
        if isinstance(t, (list, tuple)) and len(t) == 3:
            t_type, s, e = t[0], int(t[1]), int(t[2])
        elif isinstance(t, dict):
            t_type, s, e = t.get("type", "unknown"), int(t.get("start", 0)), int(t.get("end", L - 1))
        else:
            continue
        s = max(0, min(s, L - 1))
        e = max(0, min(e, L - 1))
        if s > e:
            s, e = e, s
        trend_segments.append({"type": t_type, "start": s, "end": e})
    attribute_pool["trend"]["segments"] = trend_segments
    
    # Merge annotations
    anns = {}
    local_by_type = {}
    for lc in attribute_pool.get("local", []):
        fp = lc.get("footprint")
        if fp:
            local_by_type.setdefault(lc["type"], []).append(fp)
    for k, ints in local_by_type.items():
        anns[f"local::{k}"] = merge_and_sanitize_intervals(ints, L)

    trend_by_type = {}
    for seg in trend_segments:
        trend_by_type.setdefault(seg["type"], []).append([seg["start"], seg["end"]])
    for k, ints in trend_by_type.items():
        anns[f"trend::{k}"] = merge_and_sanitize_intervals(ints, L)

    mean_v = float(np.mean(y))
    below_mask = (y < mean_v)
    anns["value_below_mean"] = merge_and_sanitize_intervals(mask_to_intervals(below_mask), L)

    attribute_pool["annotations"] = anns

    # remap → annotation_mapped
    attribute_pool["annotation_mapped"] = remap_to_annotation(attribute_pool, y)

    return y, attribute_pool

def attribute_to_text(time_series: np.ndarray, attribute_pool: dict, generate_values: bool=True, include_attributes: List[str] = ['length', 'trend', 'periodicity', 'frequency', 'noise', 'local', 'statistic']) -> str:
    """
    Generates a textual description of a time series based on various attributes and attributes.
    Args:
        time_series (np.ndarray): The time series data as a numpy array.
        attribute_pool (dict): A dictionary containing attribute details for the time series.
        generate_values (bool, optional): Deprecated. Use 'statistic' in include_attributes instead. Defaults to True.
        include_attributes (List[str], optional): A list of attributes to include in the description. Defaults to ['length', 'trend', 'periodicity', 'frequency', 'noise', 'local', 'statistic'].
    Returns:
        str: A detailed textual description of the time series.
    """
    # Adapt to legacy parameters
    if not generate_values and 'statistic' in include_attributes:
        include_attributes.remove('statistic')
    elif generate_values and 'statistic' not in include_attributes:
        include_attributes.append('statistic')

    seq_len = len(time_series)
    max_value = round(np.max(time_series), 2)
    min_value = round(np.min(time_series), 2)

    detailed_description = ''
    if 'length' in include_attributes:
        detailed_description += f"The length of the time series is {seq_len}. "
    if 'trend' in include_attributes:
        detailed_description += f"{attribute_pool['trend']['detail']}"
    if 'periodicity' in include_attributes:
        detailed_description += attribute_pool['seasonal']['detail']
    if "no" not in attribute_pool['seasonal']['type'] and 'frequency' in include_attributes:
        detailed_description += attribute_pool['frequency']['detail']
    if 'noise' in include_attributes:
        detailed_description += attribute_pool['noise']['detail']
    if 'local' in include_attributes:
        if len(attribute_pool["local"]):
            detailed_description += 'In terms of local characteristics, ' + ";".join([f"{i['detail']}, forming a {i['type']}" for i in attribute_pool['local']]) + '. '
        else:
            detailed_description += 'No local characteristics are found. '
    if 'statistic' in include_attributes:
        if seq_len >= 64:
            segments = 32
        elif seq_len >= 32:
            segments = 16
        else:
            segments = seq_len

        segment_mean = [round(np.mean(time_series[i:i + seq_len // segments]), 2) for i in range(0, seq_len, seq_len // segments)]
        detailed_description += f"Specific data details: The time series is divided into {segments} segments, with the approximate mean values for each {seq_len // segments}-point interval being: {segment_mean}. The maximum value of the entire series is {max_value}, and the minimum value is {min_value}."

    return detailed_description


def attribute_to_caption(time_series: np.ndarray, attribute_pool: dict, generate_values: bool=True) -> str:
    """
        Compared with text, caption is in a more natural and fluent way that combines the trend with the local flucations.
    """
    seq_len = len(time_series)
    if seq_len >= 64:
        segments = 32
    elif seq_len >= 32:
        segments = 16
    else:
        segments = seq_len

    segment_mean = [round(np.mean(time_series[i:i + seq_len // segments]), 2) for i in range(0, seq_len, seq_len // segments)]
    max_value = round(np.max(time_series), 2)
    min_value = round(np.min(time_series), 2)

    # Some basic attribute_pool
    detailed_description = ''
    detailed_description += f"The length of the time series is {seq_len}. "
    detailed_description += attribute_pool['seasonal']['detail']
    if "no" not in attribute_pool['seasonal']['type']:
        detailed_description += attribute_pool['frequency']['detail']
    detailed_description += attribute_pool['noise']['detail']

    # Combine the multiple attribute_pool
    detailed_description += "In terms of the trend and changes of this time series: At the beginning, "
    all_local_changes = dict((int(v['position_start']), v) for v in attribute_pool['local'])
    cur_pos = 0
    while True:
        if cur_pos >= seq_len - 1:
            break

        # Find the next local change
        later_changes = sorted(k for k in all_local_changes if k >= cur_pos)
        later_trend = sorted(k[1] for k in attribute_pool["trend"]["trend_list"] if k[1] > cur_pos)
        cur_trend = [k for k in attribute_pool["trend"]["trend_list"] if (k[1] <= cur_pos < k[2])][0]

        if (len(later_changes) > 0 and len(later_trend) > 0 and later_changes[0] < later_trend[0]) or (len(later_changes) > 0 and len(later_trend) == 0):
            # Later is a change
            nxt_pos = later_changes[0]
            cur_change = [k for k in attribute_pool["local"] if k['position_start'] == nxt_pos][0]
            if nxt_pos > cur_pos:
                detailed_description += f"from point {cur_pos} to {nxt_pos}, the time series {cur_trend[0]} with values from {float(time_series[cur_pos]):.2f} to {float(time_series[nxt_pos]):.2f}; "
            detailed_description += f"from point {cur_change['position_start']} to point {cur_change['position_end']}, {cur_change['detail']}, forming a {cur_change['type']}; "
            cur_pos = cur_change['position_end']
        elif (len(later_changes) > 0 and len(later_trend) > 0 and later_changes[0] >= later_trend[0]) or (len(later_trend) > 0 and len(later_changes) == 0):
            # Later is a trend
            nxt_pos = later_trend[0]
            nxt_trend = [k for k in attribute_pool["trend"]["trend_list"] if k[1] == nxt_pos]
            if nxt_pos > cur_pos:
                detailed_description += f"from point {cur_pos} to {nxt_pos}, the time series {cur_trend[0]} with values from {float(time_series[cur_pos]):.2f} to {float(time_series[nxt_pos]):.2f}, and then the trend of the time series changes to {nxt_trend[0][0]}; "
            cur_pos = nxt_pos
        else:
            # Later is the end
            nxt_pos = seq_len - 1
            if nxt_pos > cur_pos:
                detailed_description += f"finally, from point {cur_pos} to {nxt_pos}, the time series {cur_trend[0]} with values from {float(time_series[cur_pos]):.2f} to {float(time_series[nxt_pos]):.2f}. "
            break
    
    if generate_values:
        detailed_description += f"Specific data details: The time series is divided into {segments} segments, with the approximate mean values for each {seq_len // segments}-point interval being: {segment_mean}. The maximum value of the entire series is {max_value}, and the minimum value is {min_value}. The start value is {float(time_series[0]):.2f}, the end value if {float(time_series[-1]):.2f}. "
        
        # Random choose some points
        for _ in range(5):
            cur_pos = random.choice(list(range(seq_len)))
            detailed_description += f"The value of point {cur_pos} is {float(time_series[cur_pos]):.2f}. "

    return detailed_description

def prompt_to_inference(timeseries: np.ndarray, prompt: str) -> str:
    prompt_list = prompt.split("<ts><ts/>")
    result = prompt_list[0]

    for i in range(len(prompt_list) - 1):
        cur_ts = timeseries[i]
        if type(cur_ts) == np.ndarray:
            cur_ts = cur_ts.tolist()
        cur_ts = [[round(float(v), 4) for v in item] for item in cur_ts]
        result += f"<ts>{cur_ts}<ts/>" + prompt_list[i + 1]

    return result
