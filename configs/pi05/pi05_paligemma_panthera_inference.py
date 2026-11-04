# Inference config for the merged OpenPI PI0.5 checkpoint fine-tuned on
# Panthera-HT (single 6-DOF arm + gripper, task "pillbottle", step 15000).
#
#   conda activate fluxvla
#   python scripts/inference.py \
#       --config configs/pi05/pi05_paligemma_panthera_inference.py \
#       --ckpt-path checkpoints/openpi_panthera_15000/model.safetensors
#
# The runner talks to the arm through the Panthera-HT Python SDK
# (hightorque_robot) and to the cameras through pyrealsense2, so no ROS is
# required.  Both packages must be importable inside the fluxvla env.
#
# Checkpoint facts (see checkpoints/openpi_panthera_15000/):
#   * state      : 7 dims = 6 joint positions (rad) + gripper in [0, 1]
#   * action     : 7 dims = 6 joint deltas (rad) + absolute gripper in [0, 1]
#   * statistics : checkpoints/dataset_statistics.json (converted from
#                  merged_openpi/assets/local/panthera_pillbottle/norm_stats.json)
#   * tokenizer  : PaliGemma sentencepiece model shipped with the checkpoint

_base_ = ['./pi05_paligemma_ur3_full_finetune.py']

# 7 native action dims: 6 joint deltas + 1 absolute gripper.
model = dict(
    type='PI05FlowMatching',
    ori_action_dim=7,
    max_action_dim=32,
    n_action_steps=50,
)
inference_model = model.copy()

inference = dict(
    type='PantheraInferenceRunner',
    keep_params_fp32=True,
    mixed_precision_dtype='bf16',
    seed=7,
    # 30 Hz matches the Panthera-HT demonstration data.
    publish_rate=30,
    action_chunk=50,
    max_publish_step=10000,
    state_dim=7,
    # Replace this with the exact instruction used for data collection.
    task_descriptions={'1': 'pick up the pill bottle'},
    dataset=dict(
        type='PrivateInferenceDataset',
        # order matters: cam_high = external/head camera, cam_left_wrist = wrist
        img_keys=['cam_high', 'cam_left_wrist'],
        transforms=[
            dict(
                type='NormalizeStatesAndActions',
                state_dim=None,
                state_key='proprio',
                action_key='action',
                norm_type='quantile',
                output_dtype='float32'),
            dict(type='PreparePromptWithState'),
            dict(
                type='ProcessPrompts',
                max_len=200,
                # sentencepiece tokenizer shipped with the checkpoint; needs
                # <model_path>/tokenizer/tokenizer.model
                model_path='checkpoints/openpi_panthera_15000',
                tokenizer=dict(type='PaligemmaTokenizer')),
            dict(type='PadStatesAndActions', model_action_dim=32),
            # OpenPI `resize_with_pad`: keep the 4:3 aspect, center pad to
            # 224x224.  Matches the UR3/Franka PI0.5 configs in this repo.
            dict(
                type='ResizeImagesWithPad',
                height=224,
                width=224,
                backend='pil'),
            dict(type='SimpleNormalizeImages'),
        ]),
    denormalize_action=dict(
        type='DenormalizeDeltaAction',
        norm_type='quantile',
        action_dim=7,
        delta_action_mask=[True] * 6 + [False],
    ),
    operator=dict(
        type='PantheraOperator',
        robot_config='Panthera-HT_SDK/panthera_python/robot_param/'
        'Follower.yaml',
        # RealSense serial numbers on this workstation.
        cameras=dict(
            cam_high=dict(
                serial='153122074552', width=640, height=480, fps=30),
            cam_left_wrist=dict(
                serial='260522273111', width=640, height=480, fps=30),
        ),
        # rad/s used for every servo target.
        command_velocity=0.5,
        # model gripper in [0, 1] -> raw motor units (SDK limit is [0, 2]).
        gripper_scale=2.0,
        gripper_velocity=0.5,
        # joint-space safety clamp around the measured position (rad).
        max_joint_step=0.1,
    ),
)
