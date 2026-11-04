#!/usr/bin/env bash
set -euo pipefail

# 所有部署参数集中在这里。可从任意目录执行；第一个参数覆盖 MODE。
ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd -P)"
CONDA_ROOT=/home/sunteng/miniconda3
ROBOT_ENV=panthera
POLICY_ENV=fluxvla
MODE=check                         # check / smoke / dry-run / rollout
SDK_ROOT=/home/sunteng/haoyun/Panthera-HT_SDK
ROBOT_CONFIG="$SDK_ROOT/panthera_python/robot_param/Follower.yaml"
MODEL_CONFIG="$ROOT/configs/pi05/pi05_paligemma_panthera_inference.py"
CHECKPOINT="$ROOT/checkpoints/openpi_panthera_15000/model.safetensors"
STATISTICS="$ROOT/checkpoints/dataset_statistics.json"
TOKENIZER_ROOT="$ROOT/checkpoints/openpi_panthera_15000"
CUDA_DEVICE=0
PARAM_DTYPE=checkpoint             # 保留权重原有 BF16/FP32 混合精度；也可设 fp32
SEED=7
DENOISE_STEPS=10
INSTRUCTION="Pick the white cylindrical bottle up and set it onto the pad"
STATE_SOURCE=previous-target       # 依据此前训练数据审计；可显式改 measured 做对照
HEAD_SERIAL=153122074552           # 沿用已有配置；需确认实物对应关系
WRIST_SERIAL=260522273111
CAMERA_WIDTH=640
CAMERA_HEIGHT=480
CAMERA_FPS=30
CAMERA_TIMEOUT_MS=1000
CAMERA_WARMUP_FRAMES=15
CONTROL_HZ=30
EXECUTE_STEPS=20                   # 模型每次预测 50 步，只执行前 10 步后重新观测
MAX_STEPS=600                      # 单次启动只跑一轮，不自动复位/重复
SMOKE_CHUNKS=2
POLICY_TIMEOUT_S=30
LOAD_TIMEOUT_S=180
FEEDBACK_TIMEOUT_S=0.5
MOTOR_TIMEOUT_MS=500               # 控制线程消失后由电机超时进入阻尼
COMMAND_VELOCITY=0.3               # SDK 速度上限 rad/s
MAX_JOINT_STEP=0.04                # 同时受 COMMAND_VELOCITY/CONTROL_HZ 限制
MAX_TARGET_JUMP=0.75               # 单条目标距上一命令过大则终止 rad
MAX_TRACKING_ERROR=0.35            # 已发送目标与实测位置差值上限 rad
MAX_TORQUE=(21 36 36 21 10 10)     # Follower.yaml 上限
JOINT_SIGNS=(1 1 1 1 1 1)         # policy = signs * (sdk - offsets)
JOINT_OFFSETS=(0 0 0 0 0 0)       # rad；尚未做仿真/实机零位标定
GRIPPER_CLOSED=0.0
GRIPPER_OPEN=1.6                   # SDK gripper_open 默认值；需按实物标定
GRIPPER_VELOCITY=0.5
GRIPPER_TORQUE=0.5
MAX_GRIPPER_STEP=0.05              # 每控制周期，模型归一化单位
LOG_ROOT="$ROOT/deploy/runs"
SAVE_OBSERVATIONS=1                # 1: 持续保存两路相机每帧无损 PNG/帧号/时间戳，并保留推理输入 npz
RECORDING_QUEUE_SIZE=128           # 两路共享的写盘队列帧数；满时明确报错并停止，不静默丢帧
RECORDING_PNG_COMPRESSION=0         # 0..9 均无损；0 不压缩，优先保证双路 30 FPS 写盘速度
RECORDING_FLUSH_TIMEOUT_S=30        # 退出时停止采集，再等待已接收帧写完；超时报告保存不完整

# 使用方式：bash deploy/rollout_panthera.sh [check|smoke|dry-run|rollout]
# check: 依赖、权重头、预处理检查；smoke: 真实 CUDA + 合成观测，都不打开硬件。
# dry-run: 打开两路相机和 SDK，读反馈并预测，不下发策略目标。
# rollout: 控制线程持续保持上一目标；每次执行 EXECUTE_STEPS 后重新观测。
# dry-run/rollout 都要求 SDK 串口空闲，不可与遥操作或机械臂后台服务共用。
# Ctrl+C/异常退出时 rollout 调用 SDK set_stop；SDK 析构还会执行 YAML 的
# exit_motor_brake_flag 行为。退出不等于保持承重姿态，也不会自动回零。
# 本地数据来自仿真；相机摆位、关节符号/零位、夹爪行程需按实机对齐。

export CUDA_VISIBLE_DEVICES="$CUDA_DEVICE"
export PYTHONUNBUFFERED=1
export PYTHONPATH="$ROOT${PYTHONPATH:+:$PYTHONPATH}"
cd "$ROOT"
exec "$CONDA_ROOT/envs/$ROBOT_ENV/bin/python" "$ROOT/deploy/rollout_panthera.py" \
  --mode "${1:-$MODE}" \
  --policy-python "$CONDA_ROOT/envs/$POLICY_ENV/bin/python" \
  --sdk-root "$SDK_ROOT" --robot-config "$ROBOT_CONFIG" \
  --model-config "$MODEL_CONFIG" --checkpoint "$CHECKPOINT" \
  --statistics "$STATISTICS" --tokenizer-root "$TOKENIZER_ROOT" \
  --param-dtype "$PARAM_DTYPE" --seed "$SEED" --denoise-steps "$DENOISE_STEPS" \
  --instruction "$INSTRUCTION" --state-source "$STATE_SOURCE" \
  --head-serial "$HEAD_SERIAL" --wrist-serial "$WRIST_SERIAL" \
  --camera-width "$CAMERA_WIDTH" --camera-height "$CAMERA_HEIGHT" \
  --camera-fps "$CAMERA_FPS" --camera-timeout-ms "$CAMERA_TIMEOUT_MS" \
  --camera-warmup-frames "$CAMERA_WARMUP_FRAMES" \
  --control-hz "$CONTROL_HZ" --execute-steps "$EXECUTE_STEPS" \
  --max-steps "$MAX_STEPS" --smoke-chunks "$SMOKE_CHUNKS" \
  --policy-timeout-s "$POLICY_TIMEOUT_S" --load-timeout-s "$LOAD_TIMEOUT_S" \
  --feedback-timeout-s "$FEEDBACK_TIMEOUT_S" --motor-timeout-ms "$MOTOR_TIMEOUT_MS" \
  --command-velocity "$COMMAND_VELOCITY" --max-joint-step "$MAX_JOINT_STEP" \
  --max-target-jump "$MAX_TARGET_JUMP" --max-tracking-error "$MAX_TRACKING_ERROR" \
  --max-torque "${MAX_TORQUE[@]}" --joint-signs "${JOINT_SIGNS[@]}" \
  --joint-offsets "${JOINT_OFFSETS[@]}" \
  --gripper-closed "$GRIPPER_CLOSED" --gripper-open "$GRIPPER_OPEN" \
  --gripper-velocity "$GRIPPER_VELOCITY" --gripper-torque "$GRIPPER_TORQUE" \
  --max-gripper-step "$MAX_GRIPPER_STEP" \
  --log-root "$LOG_ROOT" --save-observations "$SAVE_OBSERVATIONS" \
  --recording-queue-size "$RECORDING_QUEUE_SIZE" \
  --recording-png-compression "$RECORDING_PNG_COMPRESSION" \
  --recording-flush-timeout-s "$RECORDING_FLUSH_TIMEOUT_S"
