# Copyright 2025 Tsinghua University and ByteDance.
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
import copy
from typing import *
from ts_generator.mmts_generate import generate_random_attributes, generate_time_series, attribute_to_text, all_attribute_set
from utils.encoding_utils import timeseries_encoding, timeseries_to_list
import yaml
import os


# CONFIG
NUM_DATA = yaml.safe_load(open("config/datagen_config.yaml"))["num_data_template_qa"]
SEQ_LEN = yaml.safe_load(open("config/datagen_config.yaml"))["seq_len"]  # Set to None for random length
ENCODING_METHOD = yaml.safe_load(open("config/datagen_config.yaml"))["encoding_method"]
OUTPUT_BASE_DIR = yaml.safe_load(open("config/datagen_config.yaml"))["data_output_dir"]
OUTPUT_PATH = f'{OUTPUT_BASE_DIR}/uts_template_{SEQ_LEN}_{NUM_DATA}_{ENCODING_METHOD}_test.jsonl'
DISABLE_EXTREME_LENGTHS = yaml.safe_load(open("config/datagen_config.yaml"))["disable_extreme_lengths"]


def attribute_pool_to_json(attribute_pool: dict) -> str:
    result = copy.deepcopy(attribute_pool)
    for i in range(len(result['local'])):
        result["local"][i]['amplitude'] = round(result["local"][i]['amplitude'], 2)
    if 'overall_amplitude' in result:
        del result['overall_amplitude']
    if 'overall_bias' in result:
        del result['overall_bias']
    if 'statistics' in result:
        del result['statistics']
    if 'trend_list' in result.get('trend', {}):
        del result['trend']['trend_list']
    return json.dumps(result, ensure_ascii=False)

def generate_single_dataset():
    if SEQ_LEN is None:
        current_seq_len = random.randint(64, 1024)
    else:
        current_seq_len = SEQ_LEN

    # 生成属性池和时间序列
    attribute_pool = generate_random_attributes(
        all_attribute_set['overall_attribute'],
        all_attribute_set['change'],
        seq_len=current_seq_len
    )
    timeseries, attribute_pool = generate_time_series(attribute_pool, current_seq_len)

    # 编码
    scaled_timeseries, cur_ts_prompt, _ = timeseries_encoding(timeseries, ENCODING_METHOD)

    # 返回单条结果
    return {
        'timeseries': timeseries_to_list([scaled_timeseries]),
        'annotation_mapped': attribute_pool.get("annotation_mapped", None)
    }


if __name__ == '__main__':
    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
    with open(OUTPUT_PATH, 'wt') as f:
        with tqdm(total=NUM_DATA, desc='Generating') as t:
            cnt = 0
            while True:
                try:
                    result = generate_single_dataset()
                except ValueError as err:
                    continue
                except IndexError as err:
                    continue
                
                item = generate_single_dataset()
                f.write(json.dumps(item, ensure_ascii=False) + '\n')
                t.update()
                cnt += 1
                if cnt >= NUM_DATA:
                    break