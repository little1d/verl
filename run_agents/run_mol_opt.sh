#!/usr/bin/env bash

export VLLM_USE_V1=1

set -x

swanlab login --host http://100.101.31.125:8001 --relogin -k PGXG66CPWHASFqnS6irMr

# ------------- paths (edit accordingly) -------------
model=/mnt/shared-storage-user/yangzhuo/main/model/Qwen2.5-7B-Instruct
template=qwen2.5-no-system-tool
data_dir=/mnt/shared-storage-user/yangzhuo/main/projects/agentrl/AgentFly/data/mol_opt
train_dataset=${data_dir}/train_mol_opt_train.json
val_dataset=${data_dir}/train_mol_opt_val.json

# ------------- agent / training hyperparams -------------
agent_type=react
agent_backend=async_verl
reward_name=mol_opt_reward
# Provide tools the agent can optionally call (prop + scaffold + oracle)
tools="['chem_mol_validate','chem_calc_properties','chem_calc_logp','chem_calc_solubility','chem_calc_qed','chem_murcko_scaffold','chem_scaffold_similarity','chem_tanimoto_similarity','chem_oracle_score']"
max_turns=4
batch_size=32
num_chains=4
lr=5e-7
kl_coef=0.001
kl_loss_type=mse
entropy_coeff=0.001
response_length=512
total_training_steps=400
adv_estimator=grpo
mini_batch_size=$batch_size
project_name="AgentRL"
experiment_name="mol_opt_qwen2.5-7b"

# ------------- log paths (to avoid conflicts when running multiple experiments) -------------
# Extract model name from model path for log file naming
model_name=$(basename $model)
log_dir="/mnt/shared-storage-user/yangzhuo/main/projects/agentrl/AgentFly/verl/logs"
reward_debug_log="${log_dir}/reward_debug_${model_name}.log"
mol_edit_traj_log="${log_dir}/mol_edit_traj_${model_name}.jsonl"
export REWARD_DEBUG_FILE=$reward_debug_log
export MOL_EDIT_TRAJ_FILE=$mol_edit_traj_log

# ------------- Ray (adjust to your machine) -------------
ray stop
rm -rf /tmp/ray /home/yangzhuo/tmp/ray
head_node_ip=$(hostname --ip-address)
port=6379
ray start --head --node-ip-address="$head_node_ip" --port=$port --num-cpus 16 --num-gpus 4 --include-dashboard=false

# ------------- Launch training -------------
python3 -m verl.trainer.main_ppo \
    algorithm.adv_estimator=$adv_estimator \
    data.train_files=${train_dataset} \
    data.val_files=${val_dataset} \
    data.train_batch_size=$batch_size \
    agent.agent_type=$agent_type \
    agent.tools=$tools \
    agent.template=$template \
    agent.model_name_or_path=$model \
    agent.max_turns=${max_turns} \
    agent.backend=${agent_backend} \
    agent.reward_name=$reward_name \
    agent.num_chains=$num_chains \
    agent.use_agent=True \
    actor_rollout_ref.actor.optim.lr=$lr \
    actor_rollout_ref.model.use_remove_padding=False \
    actor_rollout_ref.model.path=${model} \
    actor_rollout_ref.actor.ppo_mini_batch_size=${mini_batch_size} \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=2 \
    actor_rollout_ref.actor.use_kl_loss=True \
    actor_rollout_ref.actor.kl_loss_coef=$kl_coef \
    actor_rollout_ref.actor.kl_loss_type=$kl_loss_type \
    actor_rollout_ref.actor.entropy_coeff=$entropy_coeff \
    actor_rollout_ref.model.enable_gradient_checkpointing=False \
    actor_rollout_ref.actor.fsdp_config.param_offload=True \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=True \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=4 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=2 \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.response_length=$response_length \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.5 \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=4 \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    critic.model.path=$model \
    critic.ppo_mini_batch_size=${mini_batch_size} \
    critic.ppo_micro_batch_size_per_gpu=2 \
    algorithm.kl_ctrl.kl_coef=$kl_coef \
    trainer.critic_warmup=0 \
    trainer.logger="['console','swanlab']" \
    trainer.project_name=$project_name \
    trainer.experiment_name=${experiment_name} \
    trainer.n_gpus_per_node=4 \
    trainer.nnodes=1 \
    trainer.save_freq=50 \
    trainer.test_freq=10 \
    trainer.total_training_steps=$total_training_steps \
    trainer.val_before_train=False

# test set is not used during training; see ChemCoTBench eval script for scoring
