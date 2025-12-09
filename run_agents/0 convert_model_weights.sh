hf_model_path="/mnt/shared-storage-user/yangzhuo/main/model/Qwen2.5-7B-Instruct/"
local_dir="/mnt/shared-storage-user/yangzhuo/main/projects/agentrl/AgentFly/verl/checkpoints/AgentRL/mol_edit_qwen2.5-7b/global_step_200/actor"
target_dir="/mnt/shared-storage-user/yangzhuo/main/projects/agentrl/AgentFly/verl/checkpoints/AgentRL/moledit-7b/"
python scripts/model_merger.py --backend fsdp --hf_model_path $hf_model_path --local_dir ${local_dir} --target_dir ${target_dir}