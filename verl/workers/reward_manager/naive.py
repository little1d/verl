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
import re
import json
import sys
import logging
import os

import torch

from verl import DataProto
from verl.utils.reward_score import _default_compute_score

logger = logging.getLogger(__name__)

# Debug file for reward manager
DEBUG_FILE = os.environ.get("REWARD_DEBUG_FILE", "/mnt/shared-storage-user/yangzhuo/main/projects/agentrl/AgentFly/verl/logs/reward_debug.log")

# ReAct parsing utilities (copied from agentfly to avoid dependency issues)
def parse_react_step(text: str):
    """
    Parse a single ReAct-style step into its components.
    Returns a dict with keys 'thought', 'action', and 'input', with None for missing components.
    """
    result = {"thought": None, "action": None, "input": None}
    
    # Pattern for Thought:
    thought_pattern = re.compile(r"Thought:\s*(.*?)(?=\s*(?:Action:|Input:|$))", re.IGNORECASE | re.DOTALL)
    thought_match = thought_pattern.search(text)
    if thought_match:
        result["thought"] = thought_match.group(1).strip()
    
    # Pattern for Action:
    action_pattern = re.compile(r"Action:\s*(.*?)(?=\s*(?:Thought:|Input:|$))", re.IGNORECASE | re.DOTALL)
    action_match = action_pattern.search(text)
    if action_match:
        result["action"] = action_match.group(1).strip()
    
    # Pattern for Input:
    input_pattern = re.compile(r"Input:\s*(.*?)(?=\s*(?:Thought:|Action:|$))", re.IGNORECASE | re.DOTALL)
    input_match = input_pattern.search(text)
    if input_match:
        result["input"] = input_match.group(1).strip()
    
    return result


def extract_tool_calls_from_react(response_str: str):
    """
    Extract tool calls from ReAct format response.
    Returns a list of formatted tool calls.
    """
    formatted_tool_calls = []
    
    # Parse the ReAct response
    thought_action = parse_react_step(response_str)
    action = thought_action.get("action")
    action_input = thought_action.get("input")
    
    if action and action_input:
        try:
            # Try to parse the input as JSON
            try:
                tool_call = json.loads(action_input)
            except json.JSONDecodeError:
                # If not valid JSON, try to extract JSON-like content
                # Remove potential markdown code blocks
                cleaned = action_input.strip()
                if cleaned.startswith("```"):
                    # Extract content between code blocks
                    match = re.search(r"```(?:json)?\s*(.*?)\s*```", cleaned, re.DOTALL)
                    if match:
                        cleaned = match.group(1).strip()
                try:
                    tool_call = json.loads(cleaned)
                except json.JSONDecodeError:
                    # Last resort: treat as dict-like string
                    tool_call = {"raw": action_input}
            
            # Format the tool call
            if isinstance(tool_call, dict):
                # Check if it's already in the right format
                if "name" in tool_call and "arguments" in tool_call:
                    name = tool_call["name"]
                    arguments = tool_call["arguments"]
                elif "function" in tool_call and isinstance(tool_call["function"], dict):
                    name = tool_call["function"].get("name", action)
                    arguments = tool_call["function"].get("arguments", tool_call)
                else:
                    # Use action as name, tool_call as arguments
                    name = action.strip()
                    arguments = tool_call
                
                formatted_tool_calls.append({
                    "id": None,
                    "type": "function",
                    "function": {
                        "name": name,
                        "arguments": arguments if isinstance(arguments, dict) else (json.loads(arguments) if isinstance(arguments, str) else {})
                    }
                })
        except Exception as e:
            # If parsing fails, still try to create a tool call with the action name
            if action.strip():
                # Try to parse action_input as JSON, otherwise use empty dict
                try:
                    parsed_args = json.loads(action_input) if action_input else {}
                except (json.JSONDecodeError, TypeError):
                    parsed_args = {}
                formatted_tool_calls.append({
                    "id": None,
                    "type": "function",
                    "function": {
                        "name": action.strip(),
                        "arguments": parsed_args
                    }
                })
    
    return formatted_tool_calls


class NaiveRewardManager:
    """The reward manager."""

    def __init__(self, tokenizer, num_examine, compute_score=None, reward_fn_key="data_source") -> None:
        self.tokenizer = tokenizer
        self.num_examine = num_examine  # the number of batches of decoded responses to print to the console
        self.compute_score = compute_score or _default_compute_score
        self.reward_fn_key = reward_fn_key

    def __call__(self, data: DataProto, return_dict=False):
        """We will expand this function gradually based on the available datasets"""

        # Debug: Print entry point (only log to file, not to console to reduce noise)
        try:
            with open(DEBUG_FILE, "a") as f:
                f.write(f"[RewardManager Debug] __call__ entered: return_dict={return_dict}, data_len={len(data)}\n")
                f.flush()
        except Exception:
            pass

        # If there is rm score, we directly return rm score. Otherwise, we compute via rm_score_fn
        # NOTE: For mol_edit tasks, we need to recompute even if rm_scores exists,
        # because the existing rm_scores may not have correct metadata (task, src_smiles, etc.)
        # So we skip the early return for now to ensure we can pass correct metadata
        # TODO: In the future, we could check if rm_scores contains correct data before skipping
        
        # Temporarily disable early return to debug the data flow issue
        # reward_extra_info = defaultdict(list)
        # if return_dict:
        #     for key in data.non_tensor_batch.keys():
        #         if key.startswith("rm_"):
        #             reward_extra_info[key[3:]].extend(data.non_tensor_batch[key].tolist())
        #     # Check if rm_scores exists before returning
        #     if "rm_scores" in data.batch.keys():
        #         debug_early_return = f"[RewardManager Debug] Early return: rm_scores found\n"
        #         print(debug_early_return, flush=True)
        #         logger.warning(debug_early_return)
        #         sys.stdout.flush()
        #         sys.stderr.flush()
        #         try:
        #             with open(DEBUG_FILE, "a") as f:
        #                 f.write(debug_early_return + "\n")
        #                 f.flush()
        #         except Exception as e:
        #             logger.error(f"Failed to write debug file: {e}")
        #         return {"reward_tensor": data.batch["rm_scores"], "reward_extra_info": reward_extra_info}
        # else:
        #     if "rm_scores" in data.batch.keys():
        #         debug_early_return = f"[RewardManager Debug] Early return: rm_scores found (no return_dict)\n"
        #         print(debug_early_return, flush=True)
        #         logger.warning(debug_early_return)
        #         sys.stdout.flush()
        #         sys.stderr.flush()
        #         try:
        #             with open(DEBUG_FILE, "a") as f:
        #                 f.write(debug_early_return + "\n")
        #                 f.flush()
        #         except Exception as e:
        #             logger.error(f"Failed to write debug file: {e}")
        #         return data.batch["rm_scores"]
        
        reward_extra_info = defaultdict(list)

        # Check if we have responses field - if not, we should use rm_scores if available
        if "responses" not in data.batch.keys():
            debug_no_responses = f"[RewardManager Debug] No 'responses' field found, batch keys: {list(data.batch.keys())}\n"
            print(debug_no_responses, flush=True)
            logger.warning(debug_no_responses)
            sys.stdout.flush()
            sys.stderr.flush()
            try:
                with open(DEBUG_FILE, "a") as f:
                    f.write(debug_no_responses + "\n")
                    f.flush()
            except Exception as e:
                logger.error(f"Failed to write debug file: {e}")
            
            # Debug: Print non_tensor_batch keys even when returning rm_scores
            # This helps us see what data is available
            if len(data) > 0:
                data_item = data[0]
                non_tensor_keys = list(data_item.non_tensor_batch.keys())
                debug_non_tensor = f"[RewardManager Debug] non_tensor_batch keys (when returning rm_scores): {non_tensor_keys}\n"
                print(debug_non_tensor, flush=True)
                logger.warning(debug_non_tensor)
                sys.stdout.flush()
                sys.stderr.flush()
                try:
                    with open(DEBUG_FILE, "a") as f:
                        f.write(debug_non_tensor + "\n")
                        # Print key fields
                        for key in ["task", "subtask", "src_smiles", "add_group", "remove_group", "ref_smiles", "extra_info"]:
                            if key in data_item.non_tensor_batch:
                                raw_value = data_item.non_tensor_batch[key]
                                # Try to extract scalar value
                                try:
                                    import numpy as np
                                    if isinstance(raw_value, np.ndarray):
                                        if raw_value.shape == ():
                                            scalar_value = raw_value.item()
                                        elif len(raw_value) > 0:
                                            scalar_value = raw_value[0]
                                        else:
                                            scalar_value = None
                                    elif isinstance(raw_value, (list, tuple)) and raw_value:
                                        scalar_value = raw_value[0]
                                    else:
                                        scalar_value = raw_value
                                    f.write(f"  non_tensor_batch['{key}']: {type(raw_value).__name__}={scalar_value}\n")
                                except Exception:
                                    f.write(f"  non_tensor_batch['{key}']: {type(raw_value).__name__}=<error extracting>\n")
                            else:
                                f.write(f"  non_tensor_batch['{key}']: NOT FOUND\n")
                        f.flush()
                except Exception as e:
                    logger.error(f"Failed to write debug file: {e}")
            
            # If rm_scores exists but no responses, check if we need to recompute
            if "rm_scores" in data.batch.keys():
                # Check if rm_correct exists and if it's all zeros (indicating incorrect calculation)
                need_recompute = False
                if "rm_correct" in data.non_tensor_batch:
                    try:
                        import numpy as np
                        rm_correct = data.non_tensor_batch["rm_correct"]
                        if isinstance(rm_correct, np.ndarray):
                            # Check if all correct values are 0
                            if len(rm_correct) > 0:
                                all_zeros = np.all(rm_correct == 0.0)
                                if all_zeros:
                                    debug_recompute = f"[RewardManager Debug] rm_correct is all zeros, need to recompute\n"
                                    print(debug_recompute, flush=True)
                                    logger.warning(debug_recompute)
                                    sys.stdout.flush()
                                    sys.stderr.flush()
                                    try:
                                        with open(DEBUG_FILE, "a") as f:
                                            f.write(debug_recompute + "\n")
                                            f.flush()
                                    except Exception:
                                        pass
                                    need_recompute = True
                    except Exception as e:
                        logger.warning(f"Failed to check rm_correct: {e}")
                
                # If rm_correct is all zeros, we need to recompute with correct metadata
                # Even though we have rm_scores, we need to call compute_score to calculate correct
                # We'll iterate through data and call compute_score for each item to update correct
                if need_recompute and self.compute_score is not None:
                    debug_recompute = f"[RewardManager Debug] rm_correct is all zeros, recomputing correct with metadata...\n"
                    print(debug_recompute, flush=True)
                    logger.warning(debug_recompute)
                    sys.stdout.flush()
                    sys.stderr.flush()
                    try:
                        with open(DEBUG_FILE, "a") as f:
                            f.write(debug_recompute + "\n")
                            f.flush()
                    except Exception:
                        pass
                    
                    def _scalar_helper(val):
                        """Extract scalar value from array/list, handling None and empty cases."""
                        if val is None:
                            return None
                        try:
                            import numpy as np
                            if isinstance(val, np.ndarray):
                                if val.shape == ():
                                    return val.item()
                                if len(val) > 0:
                                    result = val[0]
                                    if isinstance(result, (np.integer, np.floating)):
                                        return result.item()
                                    if result is None or (isinstance(result, str) and result.lower() == 'none'):
                                        return None
                                    return result
                                return None
                            if isinstance(val, (list, tuple)):
                                if len(val) > 0:
                                    result = val[0]
                                    if result is None or (isinstance(result, str) and result.lower() == 'none'):
                                        return None
                                    return result
                                return None
                        except Exception:
                            pass
                        if isinstance(val, str) and val.lower() in ('none', 'null', ''):
                            return None
                        return val
                    
                    # Iterate through data and recompute correct for each item
                    for i in range(len(data)):
                        data_item = data[i]
                        
                        # Extract metadata from non_tensor_batch
                        kwargs = {}
                        
                        # Field mappings: target_key -> list of possible source keys
                        # Note: subtask is removed as it's always identical to task
                        field_mappings = {
                            "task": ["task"],
                            "src_smiles": ["src_smiles"],
                            "add_group": ["add_group"],
                            "remove_group": ["remove_group"],
                            "ref_smiles": ["ref_smiles"],
                        }
                        
                        # Extract fields from non_tensor_batch
                        for target_key, source_keys in field_mappings.items():
                            for source_key in source_keys:
                                if source_key in data_item.non_tensor_batch:
                                    raw_value = data_item.non_tensor_batch[source_key]
                                    scalar_value = _scalar_helper(raw_value)
                                    if scalar_value is not None:
                                        kwargs[target_key] = scalar_value
                                        break
                        
                        # Add subtask as alias to task (for backward compatibility)
                        if "task" in kwargs:
                            kwargs["subtask"] = kwargs["task"]
                        
                        # If ref_smiles is missing, try to get it from extra_info or reward_model (backward compatibility)
                        if "ref_smiles" not in kwargs or kwargs.get("ref_smiles") is None:
                            # Try from extra_info
                            extra_info = data_item.non_tensor_batch.get("extra_info", None)
                            if extra_info and isinstance(extra_info, dict):
                                ref_smiles_from_extra = extra_info.get("ref_smiles")
                                if ref_smiles_from_extra:
                                    kwargs["ref_smiles"] = _scalar_helper(ref_smiles_from_extra)
                            # Try from reward_model.ground_truth
                            if ("ref_smiles" not in kwargs or kwargs.get("ref_smiles") is None):
                                reward_model = data_item.non_tensor_batch.get("reward_model", None)
                                if reward_model and isinstance(reward_model, dict):
                                    ground_truth = reward_model.get("ground_truth")
                                    if ground_truth:
                                        kwargs["ref_smiles"] = _scalar_helper(ground_truth)
                        
                        # Debug: Print kwargs before calling compute_score
                        if i < 3:
                            debug_kwargs = f"[RewardManager Debug] Sample {i} - kwargs before compute_score: {kwargs}\n"
                            print(debug_kwargs, flush=True)
                            logger.warning(debug_kwargs)
                            sys.stdout.flush()
                            sys.stderr.flush()
                            try:
                                with open(DEBUG_FILE, "a") as f:
                                    f.write(debug_kwargs + "\n")
                                    f.flush()
                            except Exception:
                                pass
                        
                        # Get response string for compute_score
                        # Note: When rm_scores exists, we might not have 'responses' in batch
                        # We need to extract from input_ids or use a different approach
                        response_ids = data_item.batch.get("responses", None)
                        if response_ids is None:
                            # Try to extract from input_ids using action_mask or reward_mask
                            # This happens when rm_scores was computed earlier
                            input_ids = data_item.batch.get("input_ids", None)
                            action_mask = data_item.batch.get("action_mask", None)
                            reward_mask = data_item.batch.get("reward_mask", None)
                            
                            if input_ids is not None:
                                # Try to use action_mask or reward_mask to identify response part
                                if action_mask is not None:
                                    # action_mask: 1 for response tokens, 0 for prompt tokens
                                    response_mask = action_mask.bool()
                                    if response_mask.any():
                                        response_ids = input_ids[response_mask]
                                    else:
                                        # Fallback: assume response is after prompt
                                        # Find prompt length from attention_mask
                                        attention_mask = data_item.batch.get("attention_mask", None)
                                        if attention_mask is not None:
                                            prompt_length = attention_mask.sum().item()
                                            if prompt_length < len(input_ids):
                                                response_ids = input_ids[prompt_length:]
                                            else:
                                                response_ids = None
                                        else:
                                            response_ids = None
                                elif reward_mask is not None:
                                    # reward_mask: 1 for tokens that need reward calculation
                                    response_mask = reward_mask.bool()
                                    if response_mask.any():
                                        response_ids = input_ids[response_mask]
                                    else:
                                        response_ids = None
                                else:
                                    # Fallback: try to extract from labels (labels != -100)
                                    labels = data_item.batch.get("labels", None)
                                    if labels is not None:
                                        # labels: -100 for prompt tokens, actual token ids for response
                                        response_mask = labels != -100
                                        if response_mask.any():
                                            response_ids = labels[response_mask]
                                        else:
                                            response_ids = None
                                    else:
                                        response_ids = None
                            else:
                                response_ids = None
                            
                            if response_ids is None:
                                # Log to file only (this is an expected case, not an error)
                                try:
                                    with open(DEBUG_FILE, "a") as f:
                                        f.write(f"[RewardManager Debug] Sample {i} - Cannot extract response from input_ids/labels, skipping\n")
                                        f.flush()
                                except Exception:
                                    pass
                                continue  # Skip this sample if we can't get response
                        
                        # Decode response_ids to get response string
                        try:
                            import torch
                            if isinstance(response_ids, torch.Tensor):
                                response_ids = response_ids.cpu().tolist()
                            elif hasattr(response_ids, 'tolist'):
                                response_ids = response_ids.tolist()
                        except ImportError:
                            # If torch is not available, try to convert directly
                            if hasattr(response_ids, 'tolist'):
                                response_ids = response_ids.tolist()
                        
                        # Filter out padding tokens and special tokens
                        pad_token_id = getattr(self.tokenizer, 'pad_token_id', None)
                        eos_token_id = getattr(self.tokenizer, 'eos_token_id', None)
                        if pad_token_id is not None or eos_token_id is not None:
                            exclude_tokens = []
                            if pad_token_id is not None:
                                exclude_tokens.append(pad_token_id)
                            if eos_token_id is not None:
                                exclude_tokens.append(eos_token_id)
                            response_ids_clean = [tid for tid in response_ids if tid not in exclude_tokens]
                            if not response_ids_clean:
                                response_ids_clean = response_ids
                        else:
                            response_ids_clean = response_ids
                        
                        response_str = self.tokenizer.decode(response_ids_clean, skip_special_tokens=True)
                        
                        # Use ref_smiles directly (reward_model.ground_truth is redundant)
                        ground_truth = data_item.non_tensor_batch.get("ref_smiles", None)
                        
                        # Debug: Print response_str and ground_truth
                        if i < 3:
                            debug_call = f"[RewardManager Debug] Sample {i} - Calling compute_score with:\n"
                            debug_call += f"  response_str: {response_str[:100]}...\n"
                            debug_call += f"  ground_truth: {ground_truth}\n"
                            debug_call += f"  kwargs: {kwargs}\n"
                            print(debug_call, flush=True)
                            logger.warning(debug_call)
                            sys.stdout.flush()
                            sys.stderr.flush()
                            try:
                                with open(DEBUG_FILE, "a") as f:
                                    f.write(debug_call + "\n")
                                    f.flush()
                            except Exception:
                                pass
                        
                        # Call compute_score to get correct value
                        # For molecule editing, we should use mol_edit_simple directly
                        # instead of relying on self.compute_score which might be _default_compute_score
                        kwargs_for_call = kwargs.copy()
                        ref_smiles_value = kwargs_for_call.pop("ref_smiles", ground_truth)
                        
                        # Construct trajectory for mol_edit_simple (needed for SMILES extraction)
                        # Extract tool calls from ReAct format response
                        tool_calls = extract_tool_calls_from_react(response_str)
                        
                        # Format assistant content (support both string and list format)
                        assistant_content = response_str
                        if isinstance(assistant_content, str):
                            # Convert to list format if needed (for consistency with AgentFly format)
                            assistant_content = [{"type": "text", "text": response_str}]
                        
                        # Get prompt_str from data_item (question is always present)
                        prompt_str = data_item.non_tensor_batch.get("question", "")
                        
                        trajectory = [
                            {"role": "user", "content": prompt_str},
                            {
                                "role": "assistant", 
                                "content": assistant_content,
                                "tool_calls": tool_calls if tool_calls else []
                            }
                        ]
                        
                        try:
                            # Try to import and use mol_edit_simple directly
                            try:
                                from agentfly.rewards.mol_edit_reward import mol_edit_simple
                                import asyncio
                                
                                # mol_edit_simple is async, so we need to run it
                                try:
                                    # Try to get the event loop
                                    loop = asyncio.get_event_loop()
                                except RuntimeError:
                                    # If no event loop exists, create a new one
                                    loop = asyncio.new_event_loop()
                                    asyncio.set_event_loop(loop)
                                
                                # Run the async function
                                # Make a copy of kwargs_for_call to ensure closure captures the correct values
                                kwargs_copy = kwargs_for_call.copy()
                                
                                # Create a coroutine and run it
                                # Pass all kwargs_copy items as **kwargs to ensure they're captured
                                coro = mol_edit_simple(
                                    prediction=response_str,
                                    trajectory=trajectory,
                                    ref_smiles=ref_smiles_value,
                                    **kwargs_copy  # Pass all kwargs_copy items as **kwargs
                                )
                                
                                score = loop.run_until_complete(coro)
                                
                                # Debug: Print score after calling mol_edit_simple (only for first few samples)
                                if i < 3:
                                    debug_score = f"[RewardManager Debug] Sample {i} - mol_edit_simple returned score: {score}\n"
                                    print(debug_score, flush=True)
                                    logger.warning(debug_score)
                                    sys.stdout.flush()
                                    sys.stderr.flush()
                                    try:
                                        with open(DEBUG_FILE, "a") as f:
                                            f.write(debug_score + "\n")
                                            f.flush()
                                    except Exception:
                                        pass
                            except (ImportError, AttributeError, RuntimeError) as import_err:
                                # If import fails, fall back to self.compute_score
                                debug_fallback = f"[RewardManager Debug] Sample {i} - Cannot import mol_edit_simple, falling back to self.compute_score: {import_err}\n"
                                print(debug_fallback, flush=True)
                                logger.warning(debug_fallback)
                                
                                # Try with 'prediction' parameter (for mol_edit_simple)
                                score = self.compute_score(
                                    prediction=response_str,
                                    ground_truth=ground_truth,
                                    ref_smiles=ref_smiles_value,
                                    **kwargs_for_call
                                )
                        except TypeError as e:
                            # If that fails, try with 'solution_str' parameter (for _default_compute_score)
                            if "unexpected keyword argument 'prediction'" in str(e) or "got an unexpected keyword argument 'prediction'" in str(e):
                                # Get data_source for _default_compute_score
                                data_source = data_item.non_tensor_batch.get(self.reward_fn_key, "unknown")
                                try:
                                    # extra_info is no longer needed as all fields are in kwargs
                                    score = self.compute_score(
                                        data_source=data_source,
                                        solution_str=response_str,
                                        ground_truth=ground_truth,
                                        extra_info=None  # extra_info is redundant, all fields are in kwargs
                                    )
                                except NotImplementedError:
                                    # _default_compute_score doesn't support this data_source
                                    # Skip this sample - we can't compute correct without the proper reward function
                                    debug_skip = f"[RewardManager Debug] Sample {i} - Cannot compute correct: _default_compute_score doesn't support data_source='{data_source}'. Skipping.\n"
                                    print(debug_skip, flush=True)
                                    logger.warning(debug_skip)
                                    sys.stdout.flush()
                                    sys.stderr.flush()
                                    try:
                                        with open(DEBUG_FILE, "a") as f:
                                            f.write(debug_skip + "\n")
                                            f.flush()
                                    except Exception:
                                        pass
                                    continue  # Skip this sample
                            else:
                                raise
                        except Exception as e:
                            # Catch any other errors
                            debug_error = f"[RewardManager Debug] Sample {i} - Error computing correct: {e}\n"
                            print(debug_error, flush=True)
                            logger.warning(debug_error)
                            sys.stdout.flush()
                            sys.stderr.flush()
                            try:
                                with open(DEBUG_FILE, "a") as f:
                                    f.write(debug_error + "\n")
                                    f.flush()
                            except Exception:
                                pass
                            continue  # Skip this sample
                        
                        # Update rm_correct if score contains correct
                        if isinstance(score, dict) and "correct" in score:
                            # Update rm_correct in non_tensor_batch
                            if "rm_correct" in data.non_tensor_batch:
                                import numpy as np
                                rm_correct = data.non_tensor_batch["rm_correct"]
                                if isinstance(rm_correct, np.ndarray):
                                    rm_correct[i] = score["correct"]
                                elif isinstance(rm_correct, list):
                                    rm_correct[i] = score["correct"]
                                
                                debug_update = f"[RewardManager Debug] Updated rm_correct[{i}] = {score['correct']}\n"
                                print(debug_update, flush=True)
                                logger.warning(debug_update)
                                sys.stdout.flush()
                                sys.stderr.flush()
                                try:
                                    with open(DEBUG_FILE, "a") as f:
                                        f.write(debug_update + "\n")
                                        f.flush()
                                except Exception:
                                    pass
                        else:
                            # Debug: score doesn't have correct field
                            if i < 3:
                                debug_no_correct = f"[RewardManager Debug] Sample {i} - score doesn't have 'correct' field. score type: {type(score)}, score keys: {list(score.keys()) if isinstance(score, dict) else 'N/A'}\n"
                                print(debug_no_correct, flush=True)
                                logger.warning(debug_no_correct)
                                sys.stdout.flush()
                                sys.stderr.flush()
                                try:
                                    with open(DEBUG_FILE, "a") as f:
                                        f.write(debug_no_correct + "\n")
                                        f.flush()
                                except Exception:
                                    pass
                
                if return_dict:
                    for key in data.non_tensor_batch.keys():
                        if key.startswith("rm_"):
                            reward_extra_info[key[3:]].extend(data.non_tensor_batch[key].tolist())
                    return {"reward_tensor": data.batch["rm_scores"], "reward_extra_info": reward_extra_info}
                else:
                    return data.batch["rm_scores"]
            else:
                raise KeyError(f"Neither 'responses' nor 'rm_scores' found in data.batch. Available keys: {list(data.batch.keys())}")

        reward_tensor = torch.zeros_like(data.batch["responses"], dtype=torch.float32)
        reward_extra_info = defaultdict(list)

        already_print_data_sources = {}

        debug_loop_start = f"[RewardManager Debug] Starting loop: len(data)={len(data)}\n"
        print(debug_loop_start, flush=True)
        logger.warning(debug_loop_start)
        sys.stdout.flush()
        sys.stderr.flush()
        try:
            with open(DEBUG_FILE, "a") as f:
                f.write(debug_loop_start + "\n")
                f.flush()
        except Exception as e:
            logger.error(f"Failed to write debug file: {e}")

        for i in range(len(data)):
            data_item = data[i]  # DataProtoItem

            # Debug: Print all non_tensor_batch keys for first few samples
            # IMPORTANT: This debug output MUST appear in logs to diagnose the issue
            if i < 3:
                all_keys = list(data_item.non_tensor_batch.keys())
                debug_all_keys = f"[RewardManager Debug] Sample {i} - ALL non_tensor_batch keys: {all_keys}\n"
                print(debug_all_keys, flush=True)
                logger.warning(debug_all_keys)
                sys.stdout.flush()
                sys.stderr.flush()
                try:
                    with open(DEBUG_FILE, "a") as f:
                        f.write(debug_all_keys + "\n")
                        f.flush()
                except Exception as e:
                    logger.error(f"Failed to write debug file: {e}")

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
            
            # Get prompt_str from question field (question is always present)
            # Fallback to decoded prompt_str if question is not available
            question_str = data_item.non_tensor_batch.get("question", "")
            if question_str:
                prompt_str = question_str

            # Use ref_smiles directly (reward_model.ground_truth is redundant)
            ground_truth = data_item.non_tensor_batch.get("ref_smiles", None)

            data_source = data_item.non_tensor_batch[self.reward_fn_key]

            def _scalar(val):
                """Extract scalar value from array/list, handling None and empty cases."""
                if val is None:
                    return None
                try:
                    import numpy as np
                    if isinstance(val, np.ndarray):
                        if val.shape == ():
                            return val.item()
                        if len(val) > 0:
                            result = val[0]
                            # Handle numpy scalar types
                            if isinstance(result, (np.integer, np.floating)):
                                return result.item()
                            # Handle None in array
                            if result is None or (isinstance(result, str) and result.lower() == 'none'):
                                return None
                            return result
                        return None  # Empty array
                    if isinstance(val, (list, tuple)):
                        if len(val) > 0:
                            result = val[0]
                            # Handle None in list
                            if result is None or (isinstance(result, str) and result.lower() == 'none'):
                                return None
                            return result
                        return None  # Empty list
                except Exception as e:
                    # If extraction fails, return original value
                    pass
                # If it's a string "None" or "null", return None
                if isinstance(val, str) and val.lower() in ('none', 'null', ''):
                    return None
                return val

            # --- AgentFly Support: Pass all metadata ---
            kwargs = {}
            
            # Field mappings: target_key -> list of possible source keys
            # Note: subtask is removed as it's always identical to task
            # Note: source is removed as question is always present
            # Note: extra_info is removed as all fields are in top-level
            field_mappings = {
                "task": ["task"],
                "src_smiles": ["src_smiles", "source_smiles"],
                "add_group": ["add_group", "group_a"],
                "remove_group": ["remove_group", "group_b"],
                "ref_smiles": ["ref_smiles", "reference_smiles"],
            }
            
            # Extract fields from non_tensor_batch
            for target_key, possible_keys in field_mappings.items():
                for key in possible_keys:
                    if key in data_item.non_tensor_batch:
                        value = _scalar(data_item.non_tensor_batch[key])
                        # Only set if value is not None and not empty string
                        if value is not None and value != "":
                            kwargs[target_key] = value
                            break
            
            # Add subtask as alias to task (for backward compatibility)
            if "task" in kwargs:
                kwargs["subtask"] = kwargs["task"]
            
            # If ref_smiles is missing, try to get it from extra_info or reward_model (backward compatibility)
            if "ref_smiles" not in kwargs or kwargs.get("ref_smiles") is None:
                # Try from extra_info
                extra_info = data_item.non_tensor_batch.get("extra_info", None)
                if extra_info and isinstance(extra_info, dict):
                    ref_smiles_from_extra = extra_info.get("ref_smiles")
                    if ref_smiles_from_extra:
                        kwargs["ref_smiles"] = _scalar(ref_smiles_from_extra)
                # Try from reward_model.ground_truth
                if ("ref_smiles" not in kwargs or kwargs.get("ref_smiles") is None):
                    reward_model = data_item.non_tensor_batch.get("reward_model", None)
                    if reward_model and isinstance(reward_model, dict):
                        ground_truth = reward_model.get("ground_truth")
                        if ground_truth:
                            kwargs["ref_smiles"] = _scalar(ground_truth)
            
            # Debug: Print what we found in non_tensor_batch
            if i < 3:
                debug_kwargs = f"[RewardManager Debug] Sample {i} - kwargs after non_tensor_batch lookup: {list(kwargs.keys())}\n"
                print(debug_kwargs, flush=True)
                logger.warning(debug_kwargs)
                sys.stdout.flush()
                sys.stderr.flush()
                try:
                    with open(DEBUG_FILE, "a") as f:
                        f.write(debug_kwargs + "\n")
                        f.flush()
                except Exception as e:
                    logger.error(f"Failed to write debug file: {e}")
            
            # Construct pseudo-trajectory with tool_calls parsing
            # Extract tool calls from ReAct format response
            tool_calls = extract_tool_calls_from_react(response_str)
            
            # Format assistant content (support both string and list format)
            assistant_content = response_str
            if isinstance(assistant_content, str):
                # Convert to list format if needed (for consistency with AgentFly format)
                assistant_content = [{"type": "text", "text": response_str}]
            
            trajectory = [
                {"role": "user", "content": prompt_str},
                {
                    "role": "assistant", 
                    "content": assistant_content,
                    "tool_calls": tool_calls if tool_calls else []
                }
            ]

            # Debug: Print kwargs to help diagnose missing fields (simplified, main debug is in mol_edit_reward.py)
            if i < 3:  # Print for first 3 samples to match other debug output
                debug_msg = f"[RewardManager Debug] Sample {i}:\n"
                debug_msg += f"  non_tensor_batch keys: {list(data_item.non_tensor_batch.keys())}\n"
                # Print key fields from non_tensor_batch (removed extra_info and subtask from debug output)
                for key in ["task", "src_smiles", "add_group", "remove_group", "ref_smiles"]:
                    if key in data_item.non_tensor_batch:
                        raw_value = data_item.non_tensor_batch[key]
                        scalar_value = _scalar(raw_value)
                        debug_msg += f"  non_tensor_batch['{key}']: {type(raw_value).__name__}={scalar_value}\n"
                    else:
                        debug_msg += f"  non_tensor_batch['{key}']: NOT FOUND\n"
                debug_msg += f"  kwargs passed to reward function: {list(kwargs.keys())}\n"
                if kwargs:
                    debug_msg += f"  kwargs values: {kwargs}\n"
                
                # Use both print (with flush) and logger to ensure output - same as mol_edit_reward.py
                print(debug_msg, flush=True)
                logger.warning(debug_msg)
                sys.stdout.flush()
                sys.stderr.flush()
                
                # Also write to debug file
                try:
                    with open(DEBUG_FILE, "a") as f:
                        f.write(debug_msg + "\n")
                        f.flush()
                except Exception as e:
                    logger.error(f"Failed to write debug file: {e}")
            
            try:
                # Try calling with AgentFly signature
                score = self.compute_score(
                    prediction=response_str,
                    trajectory=trajectory,
                    ground_truth=ground_truth,
                    ref_smiles=kwargs.get("ref_smiles", ground_truth),
                    **kwargs
                )
            except TypeError as e:
                # Fallback to original verl signature
                if "unexpected keyword argument 'prediction'" in str(e) or "got an unexpected keyword argument 'prediction'" in str(e):
                    try:
                        score = self.compute_score(
                            data_source=data_source,
                            solution_str=response_str,
                            ground_truth=ground_truth,
                            extra_info=extra_info,
                        )
                    except NotImplementedError:
                        # _default_compute_score doesn't support this data_source
                        # This should not happen if compute_score is properly configured
                        error_msg = f"[RewardManager Error] _default_compute_score doesn't support data_source='{data_source}'. Please ensure mol_edit_simple is configured as compute_score."
                        print(error_msg, flush=True)
                        logger.error(error_msg)
                        sys.stdout.flush()
                        sys.stderr.flush()
                        # Return a default score to avoid crashing
                        score = {"score": 0.0, "correct": 0.0}
                else:
                    raise

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
