#!/usr/bin/env python3
"""Merge the step-4000 Panthera LoRA checkpoint (simdata-v2) into a FluxVLA bundle.

Input (OpenPI native, archived from the 5090-4 training host):
    checkpoints/openpi_panthera_4000_datav2/
        model_4000.safetensors     812 base tensors + 494 LoRA tensors on 247 layers
        model_4000.safetensors.sha256  digest recorded when the file was pulled
        metadata_4000.pt           OpenPI config, LoRA rank 16 / alpha 32, step 4000
        norm_stats_4000.json       simdata-v2 quantile statistics
        checkpoint_index.json      remote archive index (provenance only)

Output (derived files next to the untouched source, like openpi_panthera_30000_a100):
    model_fluxvla.safetensors   812 FluxVLA tensors, LoRA merged, no lora_* left
    dataset_statistics.json     {'private': {'proprio': ..., 'action': ...}}
    model_config.py             standalone FluxVLA architecture
    tokenizer/tokenizer.model   PaliGemma sentencepiece model
    paligemma_tokenizer.model   duplicate for the legacy layout
    assets/local/panthera_pillbottle/norm_stats.json
                                OpenPI-layout copy of norm_stats_4000.json
    training_config.json        source OpenPI config (read from metadata_4000.pt)
    export_manifest.json        merge provenance
    conversion_manifest.json    complete tensor name mapping
    inference_manifest.json     input/output semantics and bundle paths
    verification.json           checks that ran on this host
    README.md, SHA256SUMS

Merge arithmetic matches the earlier 8000/15000/30000 exports: compute
    W + (alpha / rank) * B @ A
in FP32, then cast back to the dtype of the source tensor, so every BF16/FP32
boundary of the base checkpoint is preserved.  The remaining 565 tensors are
copied verbatim.

Run inside the fluxvla environment from the repository root:

    python tools/convert_panthera_lora_4000_datav2.py

No CUDA device is required; the model is instantiated on the meta device and the
strict load runs on CPU.  A forward pass and any hardware rollout are out of
scope here.
"""

from __future__ import annotations

import gc
import hashlib
import json
import shutil
import sys
import types
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / 'checkpoints/openpi_panthera_4000_datav2'
STEP = 4000
SOURCE_WEIGHTS = 'model_4000.safetensors'
SOURCE_METADATA = 'metadata_4000.pt'
SOURCE_STATS = 'norm_stats_4000.json'
SOURCE_DIGEST = 'model_4000.safetensors.sha256'
ARCHIVE_INDEX = 'checkpoint_index.json'
WEIGHTS = 'model_fluxvla.safetensors'
TOTAL_TENSORS = 812
LORA_RANK = 16
LORA_ALPHA = 32.0
EXPECTED_EXPERIMENT = 'pillbottle_lora_simdata-v2'
EXPECTED_DATASET = 'local/panthera_simdata_v2_cosmos3'
EXPECTED_PROMPT = (
    'Use the robot arm to set the smooth-textured brown bottle onto the pad')
INFERENCE_CONFIG = (
    ROOT / 'configs/pi05/pi05_panthera_lora_4000_datav2_inference.py')
# The async preset is the reference rollout; the plain preset is the sync
# baseline used for A/B comparison against it.
DEPLOY_PRESET = 'deploy/config/openpi_panthera_4000_datav2_async.yaml'
DEPLOY_SYNC_PRESET = 'deploy/config/openpi_panthera_4000_datav2.yaml'
DEPLOY_PRESETS = (DEPLOY_PRESET, DEPLOY_SYNC_PRESET)
# Every Panthera bundle ships the same PaliGemma tokenizer; the 6000 bundle is a
# verified local copy and serves as the reference.
TOKENIZER = (
    ROOT / 'checkpoints/fluxvla_panthera_full-ft_6000_a100/tokenizer/tokenizer.model')
# Previous merged export of the same pi05 base lineage.  Only the 565 frozen
# tensors are compared: they must be byte-identical if both runs share a base.
PREVIOUS_EXPORT = ROOT / 'checkpoints/openpi_panthera_30000_a100/model_fluxvla.safetensors'
PREVIOUS_DELTA_MEAN = 0.02963
DERIVED_FILES = (
    'model_fluxvla.safetensors',
    'dataset_statistics.json',
    'model_config.py',
    'training_config.json',
    'export_manifest.json',
    'conversion_manifest.json',
    'inference_manifest.json',
    'verification.json',
    'README.md',
    'paligemma_tokenizer.model',
    'tokenizer/tokenizer.model',
    'assets/local/panthera_pillbottle/norm_stats.json',
)
SOURCE_FILES = (
    'model_4000.safetensors',
    'model_4000.safetensors.sha256',
    'metadata_4000.pt',
    'norm_stats_4000.json',
    'checkpoint_index.json',
)
DTYPES = {'F32': 'float32', 'BF16': 'bfloat16'}


def log(message):
    print(message, flush=True)


def read_header(path):
    """Return (tensors, metadata) from the safetensors header without reading data."""
    with open(path, 'rb') as handle:
        size = int.from_bytes(handle.read(8), 'little')
        header = json.loads(handle.read(size))
    metadata = header.pop('__metadata__', None) or {}
    return header, metadata


def sha256_file(path):
    digest = hashlib.sha256()
    with open(path, 'rb') as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def write_json(path, payload):
    path.write_text(json.dumps(payload, indent=2, sort_keys=False) + '\n')


def jsonable(value):
    """metadata.pt holds OpenPI/flax objects; keep the JSON view readable."""
    if isinstance(value, dict):
        return {str(key): jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [jsonable(item) for item in value]
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return repr(value)


def available_memory_gib():
    for line in Path('/proc/meminfo').read_text().splitlines():
        if line.startswith('MemAvailable:'):
            return int(line.split()[1]) / (1024 * 1024)
    return float('nan')


def prepare_torch():
    import torch
    if not torch.cuda.is_available():
        # fluxvla's eagle2 backbone probes the compute capability at import time;
        # the stub keeps this script usable on CPU-only hosts.
        torch.cuda.get_device_capability = lambda *args, **kwargs: (8, 0)
    torch.set_num_threads(4)
    return torch


def load_metadata(torch):
    """metadata_4000.pt pickles OpenPI/flax config objects that are not installed here.

    The stubs are removed again before anything else is imported: `inspect` walks
    sys.modules and rejects module objects whose `__file__` is not a string.
    """

    class Opaque:
        def __init__(self, *args, **kwargs):
            pass

        def __call__(self, *args, **kwargs):
            return Opaque()

        def __setstate__(self, state):
            self.__dict__['state'] = state

        def __repr__(self):
            return 'flax.nnx.filterlib.Nothing'

    class StubModule(types.ModuleType):
        def __init__(self, name):
            super().__init__(name)
            self.__file__ = f'<pickle-stub>/{name.replace(".", "_")}.py'

        def __getattr__(self, name):
            if name.startswith('__'):
                raise AttributeError(name)
            attribute = type(name, (Opaque,), {})
            setattr(self, name, attribute)
            return attribute

    stubbed = [name for name in ('flax', 'flax.nnx', 'flax.nnx.filterlib', 'openpi')
               if name not in sys.modules]
    for name in stubbed:
        sys.modules[name] = StubModule(name)
    try:
        return torch.load(SOURCE / SOURCE_METADATA, map_location='cpu',
                          weights_only=False)
    finally:
        for name in stubbed:
            sys.modules.pop(name, None)


def build_reference_model(torch):
    """FluxVLA model on the meta device: source of the 812 target names/shapes.

    Uses the architecture config of the 15000/6000/30000 lineage plus the OpenPI
    name mapping, exactly like tools/convert_panthera_lora_30000.py.
    """
    from accelerate import init_empty_weights
    from mmengine import Config
    from fluxvla.engines import build_vla_from_cfg

    config = Config.fromfile(str(ROOT / 'configs/pi05/pi05_paligemma_ur3_full_finetune.py'))
    model_config = config.model
    model_config.pretrained_name_or_path = None
    with init_empty_weights():
        model = build_vla_from_cfg(model_config)
    # The action expert's token embedding is stored under a different OpenPI name.
    model.name_mapping['llm_expert.embed_tokens'] = (
        'paligemma_with_expert.gemma_expert.lm_head')
    return model


def resolve_mapping(model, header, aliases, lora_targets):
    """Map every FluxVLA tensor name to its OpenPI weight key."""
    state = model.state_dict()
    mapping, problems = {}, []
    for name, tensor in state.items():
        candidates = [name] + [key for _, key in model._mapped_name_candidates(name)]
        candidates = list(dict.fromkeys(aliases.get(key, key) for key in candidates))
        hits = [key for key in candidates if key in header]
        if len(hits) != 1:
            problems.append((name, hits))
            continue
        source = hits[0]
        if tuple(header[source]['shape']) != tuple(tensor.shape):
            problems.append((name, source, header[source]['shape'], tuple(tensor.shape)))
            continue
        mapping[name] = source
    if problems:
        raise ValueError(f'Unresolved or mismatched tensors: {problems[:5]}')
    base_keys = {key for key in header if '.lora_' not in key}
    if base_keys - set(mapping.values()):
        raise ValueError(f'Unused source tensors: {sorted(base_keys - set(mapping.values()))}')
    merged_sources = {key[:-len('.weight')] for key in mapping.values()}
    if merged_sources & lora_targets != lora_targets:
        raise ValueError('Some LoRA targets are not reachable from the FluxVLA state dict')
    return mapping


def merge_weights(torch, header, mapping, lora_targets, scale):
    """FP32 merge of every LoRA layer, dtype-preserving copy of the rest."""
    from safetensors import safe_open

    merged, stats = {}, []
    path = SOURCE / SOURCE_WEIGHTS
    with safe_open(str(path), framework='pt', device='cpu') as handle:
        for index, name in enumerate(sorted(mapping), 1):
            source = mapping[name]
            prefix = source[:-len('.weight')]
            weight = handle.get_tensor(source)
            if prefix in lora_targets:
                low_rank = handle.get_tensor(prefix + '.lora_A').float()
                high_rank = handle.get_tensor(prefix + '.lora_B').float()
                delta = high_rank @ low_rank * scale
                value = (weight.float() + delta).to(weight.dtype)
                if not torch.isfinite(value).all():
                    raise ValueError(f'Non-finite merged tensor: {name}')
                stats.append((prefix, float(delta.norm()), float(weight.float().norm())))
            else:
                value = weight
            if value.shape != tuple(header[source]['shape']):
                raise ValueError(f'Shape drift while merging {name}')
            merged[name] = value
            if index % 64 == 0 or index == len(mapping):
                log(f'  merged {index}/{len(mapping)} tensors '
                    f'({available_memory_gib():.1f} GiB available)')
    if len(merged) != TOTAL_TENSORS:
        raise ValueError(f'Expected {TOTAL_TENSORS} FluxVLA tensors, got {len(merged)}')
    return merged, stats


def check_strict_load(torch, weights):
    """Mirror deploy/rollout_panthera.py: init_empty_weights + strict assign load."""
    from accelerate import init_empty_weights
    from mmengine import Config
    from fluxvla.engines import build_vla_from_cfg

    config = Config.fromfile(str(INFERENCE_CONFIG))
    model_config = config.inference_model.copy()
    model_config.update(pretrained_name_or_path=None, num_steps=10)
    with init_empty_weights(include_buffers=False):
        model = build_vla_from_cfg(model_config)
    model.load_state_dict(weights, strict=True, assign=True)
    expected = {name: tuple(tensor.shape) for name, tensor in model.state_dict().items()}
    for name, shape in expected.items():
        if tuple(weights[name].shape) != shape:
            raise ValueError(f'Shape mismatch after strict load: {name}')
    return len(expected)


def verify_written(torch, header, mapping, lora_targets, scale):
    """Reopen the export and re-derive every tensor from the source file."""
    from safetensors import safe_open

    path = SOURCE / WEIGHTS
    exact = finite = unchanged = recomputed = 0
    # Canonical fingerprint of the tensor payload: sorted by name, independent of
    # how safetensors happens to lay the file out.
    content = hashlib.sha256()

    def payload(tensor):
        tensor = tensor.contiguous()
        if tensor.dtype == torch.bfloat16:
            return tensor.view(torch.int16).numpy().tobytes()
        return tensor.numpy().tobytes()

    with safe_open(str(path), framework='pt', device='cpu') as handle, \
            safe_open(str(SOURCE / SOURCE_WEIGHTS), framework='pt', device='cpu') as src:
        keys = set(handle.keys())
        if keys != set(mapping):
            raise ValueError(f'Tensor set mismatch: {sorted(keys ^ set(mapping))[:5]}')
        if any('.lora_' in key for key in keys):
            raise ValueError('LoRA tensors leaked into the export')
        for name in sorted(mapping):
            source = mapping[name]
            prefix = source[:-len('.weight')]
            tensor = handle.get_tensor(name)
            expected_dtype = getattr(torch, DTYPES[header[source]['dtype']])
            if tensor.dtype != expected_dtype:
                raise ValueError(f'Dtype drift for {name}: {tensor.dtype} != {expected_dtype}')
            if not torch.isfinite(tensor).all():
                raise ValueError(f'Non-finite tensor on disk: {name}')
            finite += 1
            base = src.get_tensor(source)
            if prefix in lora_targets:
                # Re-derive the merge from the source file: any algebra, scale or
                # dtype mistake in merge_weights/verify shows up here.
                low_rank = src.get_tensor(prefix + '.lora_A').float()
                high_rank = src.get_tensor(prefix + '.lora_B').float()
                expected = (base.float() + high_rank @ low_rank * scale).to(base.dtype)
                recomputed += 1
            else:
                # The 565 untouched tensors must be byte-identical to the source.
                expected = base
                unchanged += 1
            if tensor.dtype != expected.dtype or not torch.equal(tensor, expected):
                raise ValueError(f'Tensor differs from the source-derived value: {name}')
            exact += 1
            content.update(name.encode())
            content.update(str(tensor.dtype).encode())
            content.update(str(tuple(tensor.shape)).encode())
            content.update(payload(tensor))
    return dict(verified=exact, finite=finite, unchanged=unchanged, recomputed=recomputed,
                tensor_content_sha256=content.hexdigest())


def base_lineage(torch, mapping, lora_targets, sample=48):
    """Check the frozen tensors against the 30000 export of the same pi05 base.

    LoRA training only ever updates the 494 low-rank tensors, so every other
    tensor must still equal the base checkpoint.  The 30000 bundle is a merged
    export of that base, which makes it an independent reference.
    """
    if not PREVIOUS_EXPORT.is_file():
        return None
    from safetensors import safe_open

    unchanged = sorted(name for name, source in mapping.items()
                       if source[:-len('.weight')] not in lora_targets)
    stride = max(1, len(unchanged) // sample)
    chosen = unchanged[::stride][:sample]
    compared = identical = 0
    with safe_open(str(SOURCE / SOURCE_WEIGHTS), framework='pt', device='cpu') as src, \
            safe_open(str(PREVIOUS_EXPORT), framework='pt', device='cpu') as previous:
        available = set(previous.keys())
        for name in chosen:
            if name not in available:
                continue
            compared += 1
            if torch.equal(src.get_tensor(mapping[name]), previous.get_tensor(name)):
                identical += 1
    if not compared:
        return None
    base_identical = identical == compared
    if base_identical:
        note = ('All sampled frozen tensors are byte-identical to the 30000 export, '
                'so both checkpoints share the pi05 base and the recorded delta '
                'magnitudes are comparable.')
    else:
        note = ('Some sampled frozen tensors differ from the 30000 export: the two '
                'runs do not share the same base weights, so delta magnitudes are '
                'not directly comparable.')
    return dict(
        reference=str(PREVIOUS_EXPORT.relative_to(ROOT)),
        compared=compared,
        identical=identical,
        base_identical=base_identical,
        note=note,
    )


def write_text_bundle(metadata, header, mapping, lora_targets, scale, stats,
                      verification, lineage):
    """Assets, tokenizer, statistics, configs and manifests."""
    tensor_dtypes = {}
    for source in mapping.values():
        name = DTYPES[header[source]['dtype']]
        tensor_dtypes[name] = tensor_dtypes.get(name, 0) + 1

    # Normalization assets stay exactly as the training job wrote them.
    norm_payload = json.loads((SOURCE / SOURCE_STATS).read_text())
    norm_stats = norm_payload['norm_stats']
    statistics = {'private': {'proprio': norm_stats['state'],
                              'action': norm_stats['actions']}}
    write_json(SOURCE / 'dataset_statistics.json', statistics)
    assets = SOURCE / 'assets/local/panthera_pillbottle'
    assets.mkdir(parents=True, exist_ok=True)
    shutil.copy2(SOURCE / SOURCE_STATS, assets / 'norm_stats.json')
    # The repository-level statistics file belongs to the real-robot dataset; the
    # simdata-v2 numbers must not silently replace it.
    repo_statistics = ROOT / 'checkpoints/dataset_statistics.json'
    repo_statistics_match = (
        repo_statistics.is_file()
        and json.loads(repo_statistics.read_text()) == statistics)

    # Tokenizer: identical across every Panthera bundle, verified byte for byte.
    reference = TOKENIZER
    if not reference.is_file():
        raise FileNotFoundError(reference)
    tokenizer_dir = SOURCE / 'tokenizer'
    tokenizer_dir.mkdir(exist_ok=True)
    shutil.copy2(reference, tokenizer_dir / 'tokenizer.model')
    shutil.copy2(reference, SOURCE / 'paligemma_tokenizer.model')
    tokenizer_digest = sha256_file(SOURCE / 'paligemma_tokenizer.model')
    for other in ('checkpoints/openpi_panthera_15000_a100/tokenizer/tokenizer.model',
                  'checkpoints/openpi_panthera_8000/paligemma_tokenizer.model'):
        candidate = ROOT / other
        if candidate.is_file() and sha256_file(candidate) != tokenizer_digest:
            raise ValueError(f'Tokenizer mismatch against {other}')

    # Standalone FluxVLA architecture, same shape as the 30000 bundle's file.
    from mmengine import Config
    config = Config.fromfile(str(INFERENCE_CONFIG))
    native = dict(config.inference_model)
    native.update(name_mapping=None, params_to_change_dtype=[], strict_mapping=True)
    Config(dict(model=native)).dump(str(SOURCE / 'model_config.py'))

    training_config = jsonable(metadata['config'])
    write_json(SOURCE / 'training_config.json', training_config)

    source_digest = None
    digest_file = SOURCE / SOURCE_DIGEST
    if digest_file.is_file():
        source_digest = digest_file.read_text().split()[0]
    weight_path = SOURCE / WEIGHTS
    weight_digest = sha256_file(weight_path)
    relative = [item[1] / item[2] for item in stats]
    merge = dict(
        format='openpi_lora_merged_fluxvla',
        source_checkpoint=str(SOURCE),
        source_weights=SOURCE_WEIGHTS,
        source_step=STEP,
        source_experiment=metadata['config'].get('exp_name'),
        source_config=metadata['config'].get('name'),
        source_dataset=metadata['config'].get('data', {}).get('repo_id'),
        source_global_step=metadata.get('global_step'),
        lora_rank=LORA_RANK,
        lora_alpha=LORA_ALPHA,
        merge_scale=scale,
        merge_precision='fp32_matmul_then_cast_to_source_dtype',
        merged_layers=len(lora_targets),
        unchanged_tensors=TOTAL_TENSORS - len(lora_targets),
        output_dtype='preserve',
        tensor_dtypes=tensor_dtypes,
        tensor_count=TOTAL_TENSORS,
        weight_size_bytes=weight_path.stat().st_size,
        source_sha256=source_digest,
        output_sha256=weight_digest,
        tensor_content_sha256=verification.get('tensor_content_sha256'),
        assets_copied=True,
    )
    write_json(SOURCE / 'export_manifest.json', merge)

    mapping_payload = dict(
        source=str(SOURCE),
        source_weights=SOURCE_WEIGHTS,
        tensors=TOTAL_TENSORS,
        strict_load=True,
        source_tensors_unused=[],
        lora=dict(rank=LORA_RANK, alpha=LORA_ALPHA, scale=scale,
                  layers=sorted(lora_targets)),
        aliases=read_header(SOURCE / SOURCE_WEIGHTS)[1],
        mapping=mapping,
    )
    write_json(SOURCE / 'conversion_manifest.json', mapping_payload)

    write_json(SOURCE / 'inference_manifest.json', dict(
        schema_version=1,
        framework='fluxvla',
        source_step=STEP,
        source_checkpoint=str(SOURCE),
        source_experiment=metadata['config'].get('exp_name'),
        source_dataset=metadata['config'].get('data', {}).get('repo_id'),
        source_prompt=EXPECTED_PROMPT,
        weights_file=WEIGHTS,
        norm_stats_file='assets/local/panthera_pillbottle/norm_stats.json',
        statistics_file='dataset_statistics.json',
        model_config_file='model_config.py',
        inference_config=str(INFERENCE_CONFIG.relative_to(ROOT)),
        deploy_preset=DEPLOY_PRESET,
        deploy_presets=list(DEPLOY_PRESETS),
        tokenizer_root='.',
        action_horizon=50,
        action_dim=7,
        internal_action_dim=32,
        num_denoising_steps=10,
        action_steps=10,
        rtc_prefix_len=0,
        lora_rank_at_training=LORA_RANK,
        lora_alpha=LORA_ALPHA,
        delta_action_mask=[True] * 6 + [False],
        action_space='six joint deltas (rad) plus an absolute gripper in [0, 1]',
        normalization='quantile q01/q99 of the simdata-v2 export, no clipping',
        extra_delta_transform=True,
        max_token_len=200,
        # Panthera episodes carry two cameras (head + wrist); pi0_flowmatching
        # derives the number of image slots from img_masks, so there is no
        # third masked-black slot.
        image_order=['head', 'wrist'],
        state_dim=7,
        publish_rate_hz=30,
    ))

    verification = dict(
        source_checkpoint=str(SOURCE),
        source_weights=SOURCE_WEIGHTS,
        source_sha256=source_digest,
        output_sha256=weight_digest,
        source_tensors=len(header),
        output_tensors=TOTAL_TENSORS,
        lora_rank=LORA_RANK,
        lora_alpha=LORA_ALPHA,
        merge_scale=scale,
        merged_layers=len(lora_targets),
        unchanged_tensors=TOTAL_TENSORS - len(lora_targets),
        lora_tensors_remaining=0,
        tensor_dtypes=tensor_dtypes,
        lora_delta_relative_norm=dict(
            mean=round(sum(relative) / len(relative), 6),
            max=round(max(relative), 6),
            min=round(min(relative), 6)),
        previous_export_delta_relative_norm_mean=PREVIOUS_DELTA_MEAN,
        base_lineage=lineage,
        repo_statistics_match=repo_statistics_match,
        repo_statistics_path='checkpoints/dataset_statistics.json',
        statistics_note=(
            'The bundle statistics come from the simdata-v2 export; the '
            'repository-level checkpoints/dataset_statistics.json belongs to the '
            'real-robot pillbottle dataset and was intentionally left untouched.'),
        tokenizer_sha256=tokenizer_digest,
        **verification,
        reproducibility=(
            'model_fluxvla.safetensors writes a single-key header metadata and a stable '
            'tensor order, so the file SHA256 is reproducible for a fixed environment; '
            'tensor_content_sha256 is the layout-independent fingerprint of the payload. '
            'The merge itself is FP32; a different BLAS or thread count can shift values '
            'by one ULP before the cast back to BF16/FP32.'),
        limitation=(
            'No CUDA forward pass, no OpenPI-versus-FluxVLA numerical parity and no '
            'simulation/hardware rollout were run: this host has no NVIDIA driver.'),
    )
    write_json(SOURCE / 'verification.json', verification)
    return merge, verification


def write_readme(merge):
    text = f"""# Panthera PI0.5 LoRA step {STEP}（simdata-v2）— FluxVLA 推理包

来源：`5090-4:/data/panghaoyun/project/openpi/checkpoints/pi05_panthera_lora/{EXPECTED_EXPERIMENT}/`
中归档的 `model_4000.safetensors`（OpenPI 原生训练产物，`safetensors.torch.save_model`
直存的全量权重）。`metadata_4000.pt` 记录的训练配置为 `{merge['source_config']}`，实验
`{merge['source_experiment']}`，数据集 `{merge['source_dataset']}`
（Cosmos3 仿真导出：300 条 episode、52153 帧），训练 35000 步、batch 48，
这里是第 {STEP} 步。原始 `{SOURCE_WEIGHTS}`、`{SOURCE_METADATA}`、
`{SOURCE_STATS}` 未做任何修改。

LoRA rank {LORA_RANK} / alpha {LORA_ALPHA}，共 {merge['merged_layers']} 个线性层
（PaliGemma LM 121 + action expert 126）。合并按 FP32 计算
`W + ({LORA_ALPHA:.0f}/{LORA_RANK}) * B @ A`，再恢复每个张量原有的 BF16/FP32 精度；
其余 {merge['unchanged_tensors']} 个张量与源权重逐 bit 相同。导出 {TOTAL_TENSORS}
个张量，无 `lora_*` 残留。

## 文件

- `model_fluxvla.safetensors`：{TOTAL_TENSORS} 个 FluxVLA 原生命名的推理权重。
- `model_config.py`：独立的 FluxVLA 模型结构（无配置继承依赖）。
- `dataset_statistics.json`：FluxVLA 统计量（`private.proprio` / `private.action`），
  取自 `{SOURCE_STATS}`。
- `assets/local/panthera_pillbottle/norm_stats.json`：OpenPI 布局的同一份统计量。
- `tokenizer/tokenizer.model`、`paligemma_tokenizer.model`：PaliGemma 分词器（两种布局）。
- `training_config.json`：源的 OpenPI 训练配置（从 `{SOURCE_METADATA}` 抽出）。
- `export_manifest.json`：合并来源与配方。
- `conversion_manifest.json`：{TOTAL_TENSORS} 条张量名映射（含源文件别名）。
- `inference_manifest.json`：输入输出语义、训练指令与配套配置路径。
- `verification.json`：离线校验结果。
- `SHA256SUMS`：导出文件与源文件校验和。
- `checkpoint_index.json`：远端归档目录的分步索引（provenance，未改动）。

## 与 15000/30000 包的区别

统计量不同：本包使用 simdata-v2 仿真数据集的 q01/q99，仓库级
`checkpoints/dataset_statistics.json`（实机 pillbottle 数据）没有改动，二者不可混用。
其余结构、动作语义与 30000 包一致：7 维 state、7 维 action，训练时
`extra_delta_transform=true`，因此前 6 维是关节增量、第 7 维是绝对夹爪。

## 可复现性

`model_fluxvla.safetensors` 表头只写一个 `format` 键、张量顺序固定，避免 safetensors
元数据哈希顺序造成的文件级哈希漂移；与文件布局无关的张量指纹记录在
`export_manifest.json` 的 `tensor_content_sha256`。合并是 FP32 计算，换 BLAS 或线程数
可能在回写 BF16/FP32 前出现 1 ULP 级差异。

## 动作语义

6 个关节增量（rad）+ 1 个绝对夹爪。训练时 `extra_delta_transform=true`，因此必须使用
`delta_action_mask=[True]*6+[False]` 的 delta 解码器，即
`configs/pi05/pi05_panthera_lora_4000_datav2_inference.py`。**不要**复用 6000 全参微调包
的 all-False 配置，那个 checkpoint 的训练约定不同。

训练 prompt 为 `{EXPECTED_PROMPT}`。

## 用法

```bash
# smoke 不打开硬件与相机（async 参考预设，RTC prefix + 加速度平滑）
/home/sunteng/miniconda3/envs/fluxvla/bin/python deploy/launch_panthera.py smoke \\
  --config {DEPLOY_PRESET}

# 同步基线预设（每块执行 25 步、关闭 RTC），用于与 async 预设做 A/B 对照
/home/sunteng/miniconda3/envs/fluxvla/bin/python deploy/launch_panthera.py smoke \\
  --config {DEPLOY_SYNC_PRESET}
```

## 未做的验证

本机没有 NVIDIA 驱动（`nvidia-smi` 失败），只做了离线校验：按
`deploy/rollout_panthera.py` 的调用序列做 strict 加载、逐张量数值一致、有限性、
未改动张量与源文件逐 bit 相同、tokenizer 与统计量一致性。CUDA 前向、OpenPI 与
FluxVLA 的数值对照、仿真/实机成功率均未评估。
"""
    (SOURCE / 'README.md').write_text(text)


def write_checksums(source_digest):
    lines = []
    for relative in DERIVED_FILES:
        path = SOURCE / relative
        if path.is_file():
            lines.append(f'{sha256_file(path)}  {relative}')
    for relative in SOURCE_FILES:
        path = SOURCE / relative
        if not path.is_file():
            continue
        digest = source_digest if relative == SOURCE_WEIGHTS else sha256_file(path)
        lines.append(f'{digest}  {relative}')
    (SOURCE / 'SHA256SUMS').write_text('\n'.join(lines) + '\n')


def main():
    torch = prepare_torch()
    if not (SOURCE / SOURCE_WEIGHTS).is_file():
        raise FileNotFoundError(SOURCE / SOURCE_WEIGHTS)
    if (SOURCE / WEIGHTS).exists():
        raise FileExistsError(f'{SOURCE / WEIGHTS} already exists; remove it first')
    if not INFERENCE_CONFIG.is_file():
        raise FileNotFoundError(INFERENCE_CONFIG)
    for preset in DEPLOY_PRESETS:
        if not (ROOT / preset).is_file():
            raise FileNotFoundError(ROOT / preset)
    if available_memory_gib() < 10:
        log(f'WARNING: only {available_memory_gib():.1f} GiB available; '
            'the merge holds the full 7.5 GB export in memory')

    log(f'Reading {SOURCE / SOURCE_METADATA}')
    metadata = load_metadata(torch)
    config = metadata['config']
    model_config = config['model']
    if metadata['global_step'] != STEP:
        raise ValueError(f'Unexpected step {metadata["global_step"]}')
    if config.get('exp_name') != EXPECTED_EXPERIMENT:
        raise ValueError(f'Unexpected experiment {config.get("exp_name")}')
    if config['data'].get('repo_id') != EXPECTED_DATASET:
        raise ValueError(f'Unexpected dataset {config["data"].get("repo_id")}')
    if model_config['pytorch_lora_rank'] != LORA_RANK:
        raise ValueError(f'Unexpected LoRA rank: {model_config}')
    if model_config['pytorch_lora_alpha'] != LORA_ALPHA:
        raise ValueError(f'Unexpected LoRA alpha: {model_config}')
    if model_config['action_dim'] != 32 or model_config['action_horizon'] != 50:
        raise ValueError(f'Unexpected action configuration: {model_config}')
    if not model_config['pi05'] or not model_config['discrete_state_input']:
        raise ValueError('Expected a pi05 model with discrete state input')
    if config['data']['extra_delta_transform'] is not True:
        raise ValueError('Expected extra_delta_transform=true (delta joint actions)')
    scale = model_config['pytorch_lora_alpha'] / model_config['pytorch_lora_rank']
    log(f'step={metadata["global_step"]} config={config["name"]} '
        f'exp={config["exp_name"]} dataset={config["data"]["repo_id"]} '
        f'lora_scale={scale}')

    log(f'Verifying SHA256 of {SOURCE / SOURCE_WEIGHTS}')
    digest_file = SOURCE / SOURCE_DIGEST
    if not digest_file.is_file():
        raise FileNotFoundError(digest_file)
    expected_digest = digest_file.read_text().split()[0]
    source_digest = sha256_file(SOURCE / SOURCE_WEIGHTS)
    if source_digest != expected_digest:
        raise ValueError(f'Source SHA256 mismatch: {source_digest} != {expected_digest}')
    log(f'  source sha256 OK: {source_digest}')

    header, aliases = read_header(SOURCE / SOURCE_WEIGHTS)
    lora_targets = {key.rsplit('.lora_', 1)[0] for key in header if '.lora_' in key}
    if len(header) != TOTAL_TENSORS + 2 * len(lora_targets):
        raise ValueError(f'Unexpected source tensor count: {len(header)}')
    for target in lora_targets:
        if tuple(header[target + '.lora_A']['shape'])[0] != LORA_RANK:
            raise ValueError(f'LoRA rank drift in {target}')
        if tuple(header[target + '.lora_B']['shape'])[1] != LORA_RANK:
            raise ValueError(f'LoRA rank drift in {target}')
    log(f'Source: {len(header)} tensors, {len(lora_targets)} LoRA layers, '
        f'{len(aliases)} alias(es)')

    model = build_reference_model(torch)
    mapping = resolve_mapping(model, header, aliases, lora_targets)
    log(f'Resolved {len(mapping)} FluxVLA tensors to OpenPI keys')
    del model
    gc.collect()

    merged, stats = merge_weights(torch, header, mapping, lora_targets, scale)
    log('Checking strict load against the inference config')
    loaded = check_strict_load(torch, merged)

    log(f'Writing {SOURCE / WEIGHTS}')
    from safetensors.torch import save_file
    # A single metadata key keeps the safetensors header byte-stable: the Rust
    # serializer walks its metadata map in random order, so more keys would make
    # the file hash differ between otherwise identical runs.  Full provenance
    # lives in export_manifest.json, next to the weights.
    save_file(merged, str(SOURCE / WEIGHTS), metadata={'format': 'fluxvla'})
    # Free the 7.5 GB in-memory export before the verification pass; the disk
    # payload is re-derived from the source file tensor by tensor.
    del merged
    gc.collect()

    checks = verify_written(torch, header, mapping, lora_targets, scale)
    log(f'Verified on disk: {checks}')

    # Re-load through the deploy sequence when the host has room for a second
    # full copy; otherwise the tensor-by-tensor verification above already
    # covers the same payload.
    if available_memory_gib() >= 9:
        log('Re-loading the export through the deploy path (CPU)')
        from safetensors.torch import load_file
        weights = load_file(str(SOURCE / WEIGHTS), device='cpu')
        reloaded = check_strict_load(torch, weights)
        del weights
        gc.collect()
        reload_note = f'passed on CPU ({reloaded} tensors, deploy sequence)'
    else:
        reload_note = (
            'skipped the reload: only '
            f'{available_memory_gib():.1f} GiB available; the export was still '
            'verified tensor by tensor against the source')
        log(reload_note)

    lineage = base_lineage(torch, mapping, lora_targets)
    log(f'Base lineage check: {lineage}')

    merge_manifest, verification = write_text_bundle(
        metadata, header, mapping, lora_targets, scale, stats,
        dict(strict_model_load=reload_note,
             strict_load_tensors=loaded, **checks),
        lineage)
    write_readme(merge_manifest)
    write_checksums(source_digest)

    summary = dict(output=str(SOURCE / WEIGHTS),
                   tensors=TOTAL_TENSORS,
                   merged_layers=len(lora_targets),
                   merge_scale=scale,
                   weight_bytes=merge_manifest['weight_size_bytes'],
                   weight_sha256=merge_manifest['output_sha256'],
                   strict_load=reload_note,
                   statistics=dict(
                       lora_delta_relative_norm_mean=verification[
                           'lora_delta_relative_norm']['mean'],
                       repo_statistics_match=verification['repo_statistics_match']),
                   files=[name for name in DERIVED_FILES if (SOURCE / name).is_file()])
    log(json.dumps(summary, indent=2))


if __name__ == '__main__':
    main()
