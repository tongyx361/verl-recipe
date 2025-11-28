#!/usr/bin/env bash
set -xeuo pipefail

CLUSTER_DATA_HOME=${CLUSTER_RO_DATA_HOME:-"${HOME}/verl"}

# Data

TRAIN_FILE=${TRAIN_FILE:-"${CLUSTER_DATA_HOME}/data/dapo_filter/train.parquet"}
TRAIN_DATA_URL=${TRAIN_DATA_URL:-"https://huggingface.co/datasets/aaabiao/dapo_filter/resolve/main/train.parquet?download=true"}
TEST_DATA_HOME=${TEST_DATA_HOME:-"${CLUSTER_DATA_HOME}/data/off-policy-correction"}
TEST_FILES=${TEST_FILES:-"[${TEST_DATA_HOME}/aime24.parquet,${TEST_DATA_HOME}/aime25.parquet]"}

# Model

MODEL_HOME=${MODEL_HOME:-"${CLUSTER_DATA_HOME}/model"}
MODEL_PATH=${MODEL_PATH:-"${MODEL_HOME}/Qwen3-14B-Base"}

# Rollout Correction

BYPASS_MODE=${BYPASS_MODE:-"false"}
ROLLOUT_IS=${ROLLOUT_IS:-"null"}
ROLLOUT_IS_THRESHOLD=${ROLLOUT_IS_THRESHOLD:-"null"}
ROLLOUT_IS_BATCH_NORMALIZE=${ROLLOUT_IS_BATCH_NORMALIZE:-"false"}
ROLLOUT_RS=${ROLLOUT_RS:-"null"}
ROLLOUT_RS_THRESHOLD=${ROLLOUT_RS_THRESHOLD:-"null"}
ROLLOUT_RS_THRESHOLD_LOWER=${ROLLOUT_RS_THRESHOLD_LOWER:-"null"}
ROLLOUT_TOKEN_VETO_THRESHOLD=${ROLLOUT_TOKEN_VETO_THRESHOLD:-"null"}
USE_POLICY_GRADIENT=${USE_POLICY_GRADIENT:-"false"}
ROLLOUT_CALCULATE_LOG_PROBS=${ROLLOUT_CALCULATE_LOG_PROBS:-"true"}


# Other Algorithm Configurations

max_prompt_length=1024
max_response_length=8192
prompt_global_bsz=512
ppo_mini_batch_size=64 # 512 / 64 = 8 mini-batch update step off-policy
lr="1e-9" # since `loss_agg_mode=seq-mean-token-sum`, which is length-independent but much larger than `token-mean`

# Implementation Efficiency Configuration

# Actor

FSDP_SIZE=${FSDP_SIZE:-"-1"}
USP_SIZE=${USP_SIZE:-"1"}
USE_TORCH_COMPILE=${USE_TORCH_COMPILE:-"true"}
FWD_BWD_MICRO_BATCH_TOKEN_NUM=${FWD_BWD_MICRO_BATCH_TOKEN_NUM:-"10000"}
FWD_MICRO_BATCH_TOKEN_NUM=${FWD_MICRO_BATCH_TOKEN_NUM:-"10000"}
BALANCE_BATCH=${BALANCE_BATCH:-"false"}

# Rollout

ROLLOUT_MODE=${ROLLOUT_MODE:-"sync"}
ROLLOUT_TP_SIZE=${ROLLOUT_TP_SIZE:-"1"}
GPU_MEMORY_UTILIZATION=${GPU_MEMORY_UTILIZATION:-"0.75"}
MAX_NUM_BATCHED_TOKENS=${MAX_NUM_BATCHED_TOKENS:-"10216"}


# Download data

if [ ! -f "${TRAIN_FILE}" ]; then
    PROXY_URL=${PROXY_URL:-""}
    [ -n "${PROXY_URL}" ] && export https_proxy="${PROXY_URL}"
    mkdir -p "$(dirname "${TRAIN_FILE}")" && wget "${TRAIN_DATA_URL}" -O "${TRAIN_FILE}"
    unset https_proxy
fi

# Entrypoint command to submit to the cluster

python3 -m verl.trainer.main_ppo \
    data.train_files="${TRAIN_FILE}" \
    data.val_files="${TEST_FILES}" \
    data.max_prompt_length="${max_prompt_length}" \
    data.max_response_length="${max_response_length}" \
    data.train_batch_size="${prompt_global_bsz}" \
    data.seed=42 \
    reward_model.reward_manager="math" \
    algorithm.adv_estimator="rloo" \
    algorithm.lam=1.0 \
    algorithm.gamma=1.0 \
    algorithm.use_kl_in_reward="false" \
    algorithm.kl_ctrl.kl_coef=0.0 \
    algorithm.rollout_correction.rollout_is="${ROLLOUT_IS}" \
    algorithm.rollout_correction.rollout_is_threshold="${ROLLOUT_IS_THRESHOLD}" \
    algorithm.rollout_correction.rollout_is_batch_normalize="${ROLLOUT_IS_BATCH_NORMALIZE}" \
    algorithm.rollout_correction.rollout_rs="${ROLLOUT_RS}" \
    algorithm.rollout_correction.rollout_rs_threshold="${ROLLOUT_RS_THRESHOLD}" \
    algorithm.rollout_correction.rollout_rs_threshold_lower="${ROLLOUT_RS_THRESHOLD_LOWER}" \
    algorithm.rollout_correction.rollout_token_veto_threshold="${ROLLOUT_TOKEN_VETO_THRESHOLD}" \
    algorithm.rollout_correction.bypass_mode="${BYPASS_MODE}" \
    algorithm.rollout_correction.use_policy_gradient="${USE_POLICY_GRADIENT}" \
    actor_rollout_ref.model.path="${MODEL_PATH}" \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.actor.use_dynamic_bsz=True \
    actor_rollout_ref.actor.use_torch_compile="${USE_TORCH_COMPILE}" \
    actor_rollout_ref.actor.fsdp_config.fsdp_size="${FSDP_SIZE}" \
    actor_rollout_ref.actor.ulysses_sequence_parallel_size="${USP_SIZE}" \
    actor_rollout_ref.actor.ppo_max_token_len_per_gpu="${FWD_BWD_MICRO_BATCH_TOKEN_NUM}" \
    actor_rollout_ref.actor.optim.lr="${lr}" \
    actor_rollout_ref.actor.optim.weight_decay=0.01 \
    actor_rollout_ref.actor.optim.lr_scheduler_type="constant" \
    actor_rollout_ref.grad_clip=1.0 \
    actor_rollout_ref.actor.ppo_epochs=1 \
    actor_rollout_ref.actor.ppo_mini_batch_size="${ppo_mini_batch_size}" \
    actor_rollout_ref.actor_clip_ratio_high=0.28 \
    actor_rollout_ref.actor.clip_ratio_low=0.2 \
    actor_rollout_ref.actor.clip_ratio_c=3.0 \
    actor_rollout_ref.actor.entropy_coeff=0.0 \
    actor_rollout_ref.actor.use_kl_loss="false" \
    actor_rollout_ref.actor.kl_loss_coef=0.0 \
    actor_rollout_ref.actor.loss_agg_mode="seq-mean-token-sum" \
    actor_rollout_ref.rollout.n=16 \
    actor_rollout_ref.rollout.temperature=1.0 \
    actor_rollout_ref.rollout.top_p=1.0 \
    actor_rollout_ref.rollout.top_k=-1 \
    actor_rollout_ref.val_kwargs.n=16 \
    actor_rollout_ref.val_kwargs.temperature=1.0 \
    actor_rollout_ref.val_kwargs.top_p=0.95 \
    actor_rollout_ref.val_kwargs.top_k=-1 \
    actor_rollout_ref.tensor_model_parallel_size="${ROLLOUT_TP_SIZE}" \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.mode="${ROLLOUT_MODE}" \
    actor_rollout_ref.rollout.calculate_log_probs="${ROLLOUT_CALCULATE_LOG_PROBS}" \
    actor_rollout_ref.rollout.gpu_memory_utilization="${GPU_MEMORY_UTILIZATION}" \
    actor_rollout_ref.rollout.micro_batch_size="${FWD_MICRO_BATCH_TOKEN_NUM}" \
    actor_rollout_ref.rollout.max_num_batched_tokens="${MAX_NUM_BATCHED_TOKENS}" \
    trainer.balance_batch="false" \
    trainer.log_val_generations=10 \
    trainer.max_actor_ckpt_to_keep=2 \
    trainer.save_freq=10 \
    trainer.test_freq=10 \
    trainer.val_before_train="false" \
    trainer.n_gpus_per_node=8 \
    trainer.nnodes=4 \
    trainer.logger='["console","wandb"]' \
    trainer.project_name="rollout_corr_rloo_example" \
    trainer.experiment_name="rloo_seq_is_pure" \
    trainer.total_epochs=1000

