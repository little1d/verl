# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from collections import defaultdict

import torch

from verl import DataProto
from verl.utils.reward_score import _default_compute_score


class NaiveRewardManager:
    """The reward manager."""

    def __init__(self, tokenizer, num_examine, compute_score=None, reward_fn_key="data_source") -> None:
        self.tokenizer = tokenizer
        self.num_examine = num_examine  # the number of batches of decoded responses to print to the console
        self.compute_score = compute_score or _default_compute_score
        self.reward_fn_key = reward_fn_key

    def __call__(self, data: DataProto, return_dict=False):
        """We will expand this function gradually based on the available datasets"""

        # If there is rm score, we directly return rm score. Otherwise, we compute via rm_score_fn
        reward_extra_info = defaultdict(list)
        if return_dict:
            for key in data.non_tensor_batch.keys():
                if key.startswith("rm_"):
                    reward_extra_info[key[3:]].extend(data.non_tensor_batch[key].tolist())
            return {"reward_tensor": data.batch["rm_scores"], "reward_extra_info": reward_extra_info}
        else:
            return data.batch["rm_scores"]

        reward_tensor = torch.zeros_like(data.batch["responses"], dtype=torch.float32)
        reward_extra_info = defaultdict(list)

        already_print_data_sources = {}

        for i in range(len(data)):
            data_item = data[i]  # DataProtoItem

            prompt_ids = data_item.batch["prompts"]

            prompt_length = prompt_ids.shape[-1]

            valid_prompt_length = data_item.batch["attention_mask"][:prompt_length].sum()
            valid_prompt_ids = prompt_ids[-valid_prompt_length:]

            response_ids = data_item.batch["responses"]
            valid_response_length = data_item.batch["attention_mask"][prompt_length:].sum()
            valid_response_ids = response_ids[:valid_response_length]

            # decode
            prompt_str = self.tokenizer.decode(valid_prompt_ids, skip_special_tokens=True)
            response_str = self.tokenizer.decode(valid_response_ids, skip_special_tokens=True)

            ground_truth = data_item.non_tensor_batch["reward_model"]["ground_truth"]

            data_source = data_item.non_tensor_batch[self.reward_fn_key]

            def _scalar(val):
                try:
                    import numpy as np
                    if isinstance(val, np.ndarray):
                        if val.shape == ():
                            return val.item()
                        if len(val) > 0:
                            return val[0]
                    if isinstance(val, (list, tuple)) and len(val) > 0:
                        return val[0]
                except Exception:
                    pass
                return val

            extra_info = data_item.non_tensor_batch.get("extra_info", None)

            # --- AgentFly Support: Pass all metadata ---
            kwargs = {}
            if isinstance(extra_info, dict):
                kwargs.update(extra_info)
            # 保留原始 extra_info，便于下游兜底解析
            if extra_info is not None:
                kwargs.setdefault("extra_info", extra_info)
            # Pick top-level fields if present (scalar per sample)
            for key in ["task", "subtask", "src_smiles", "add_group", "remove_group", "ref_smiles", "reward_model"]:
                if key in data_item.non_tensor_batch and key not in kwargs:
                    kwargs[key] = _scalar(data_item.non_tensor_batch[key])
            
            # Construct pseudo-trajectory
            trajectory = [
                {"role": "user", "content": prompt_str},
                {"role": "assistant", "content": response_str}
            ]

            try:
                # Try calling with AgentFly signature
                score = self.compute_score(
                    prediction=response_str,
                    trajectory=trajectory,
                    ground_truth=ground_truth,
                    ref_smiles=kwargs.get("ref_smiles", ground_truth),
                    **kwargs
                )
            except TypeError:
                # Fallback to original verl signature
                score = self.compute_score(
                    data_source=data_source,
                    solution_str=response_str,
                    ground_truth=ground_truth,
                    extra_info=extra_info,
                )

            if isinstance(score, dict):
                reward = score["score"]
                # Store the information including original reward
                for key, value in score.items():
                    reward_extra_info[key].append(value)
            else:
                reward = score

            reward_tensor[i, valid_response_length - 1] = reward

            if data_source not in already_print_data_sources:
                already_print_data_sources[data_source] = 0

            if already_print_data_sources[data_source] < self.num_examine:
                already_print_data_sources[data_source] += 1
                print("[prompt]", prompt_str)
                print("[response]", response_str)
                print("[ground_truth]", ground_truth)
                if isinstance(score, dict):
                    for key, value in score.items():
                        print(f"[{key}]", value)
                else:
                    print("[score]", score)

        if return_dict:
            return {
                "reward_tensor": reward_tensor,
                "reward_extra_info": reward_extra_info,
            }
        else:
            return reward_tensor
