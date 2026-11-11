#!/usr/bin/env python3
"""Merge the step-30000 Panthera LoRA checkpoint into a FluxVLA inference bundle.

Input (OpenPI native, written by the training job):
    checkpoints/openpi_panthera_30000_a100/
        model.safetensors   812 base tensors + 494 LoRA tensors on 247 layers
        metadata.pt         OpenPI config, LoRA rank 16 / alpha 32, step 30000
        optimizer.pt        training only, never read here
        assets/local/panthera_pillbottle/norm_stats.json

Output (derived files next to the untouched source, like openpi_panthera_15000_a100):
    model_fluxvla.safetensors   812 FluxVLA tensors, LoRA merged, no lora_* left
    dataset_statistics.json     {'private': {'proprio': ..., 'action': ...}}
    model_config.py             standalone FluxVLA architecture
    tokenizer/tokenizer.model   PaliGemma sentencepiece model
    paligemma_tokenizer.model   duplicate for the legacy layout
    training_config.json        source OpenPI config (read from metadata.pt)
    export_manifest.json        merge provenance
    conversion_manifest.json    complete tensor name mapping
    inference_manifest.json     input/output semantics and bundle paths
    verification.json           checks that ran on this host
    README.md, SHA256SUMS

Merge arithmetic matches the earlier 8000/15000 exports: compute
    W + (alpha / rank) * B @ A
in FP32, then cast back to the dtype of the source tensor, so every BF16/FP32
boundary of the base checkpoint is preserved.  The remaining 565 tensors are
copied verbatim.

Run inside the fluxvla environment from the repository root:

    python tools/convert_panthera_lora_30000.py

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
SOURCE = ROOT / 'checkpoints/openpi_panthera_30000_a100'
STEP = 30000
WEIGHTS = 'model_fluxvla.safetensors'
INFERENCE_CONFIG = ROOT / 'configs/pi05/pi05_panthera_lora_30000_inference.py'
DEPLOY_PRESET = 'deploy/config/openpi_panthera_30000_a100_async.yaml'
# Every Panthera bundle ships the same PaliGemma tokenizer; the 6000 bundle is a
# verified local copy and serves as the reference.
TOKENIZER = (
    ROOT / 'checkpoints/fluxvla_panthera_full-ft_6000_a100/tokenizer/tokenizer.model')
# Previous merged export of the same pi05 base + LoRA lineage, used only to
# sanity check the magnitude of the merge delta.
PREVIOUS_EXPORT = ROOT / 'checkpoints/openpi_panthera_15000_a100/model.safetensors'
SOURCE_MANIFEST = ROOT / 'checkpoints/openpi_panthera_30000_a100.sha256'
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


def prepare_torch():
    import torch
    if not torch.cuda.is_available():
        # fluxvla's eagle2 backbone probes the compute capability at import time;
        # the stub keeps this script usable on CPU-only hosts.
        torch.cuda.get_device_capability = lambda *args, **kwargs: (8, 0)
    torch.set_num_threads(4)
    return torch


def load_metadata(torch):
    """metadata.pt pickles OpenPI/flax config objects that are not installed here.

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
        return torch.load(SOURCE / 'metadata.pt', map_location='cpu', weights_only=False)
    finally:
        for name in stubbed:
            sys.modules.pop(name, None)


def build_reference_model(torch):
    """FluxVLA model on the meta device: source of the 812 target names/shapes.

    Uses the architecture config of the 15000/6000 lineage plus the OpenPI name
    mapping, exactly like tools/convert_panthera_full_ft_6000.py.
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
    path = SOURCE / 'model.safetensors'
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
                log(f'  merged {index}/{len(mapping)} tensors')
    if len(merged) != 812:
        raise ValueError(f'Expected 812 FluxVLA tensors, got {len(merged)}')
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


def verify_written(torch, merged, header, mapping, lora_targets, scale):
    """Reopen the export and check keys, dtypes, values and the merge formula."""
    from safetensors import safe_open

    path = SOURCE / WEIGHTS
    exact = finite = 0
    # Canonical fingerprint of the tensor payload: sorted by name, independent of
    # how safetensors happens to lay the file out.
    content = hashlib.sha256()

    def payload(tensor):
        tensor = tensor.contiguous()
        if tensor.dtype == torch.bfloat16:
            return tensor.view(torch.int16).numpy().tobytes()
        return tensor.numpy().tobytes()

    with safe_open(str(path), framework='pt', device='cpu') as handle:
        keys = set(handle.keys())
        if keys != set(merged):
            raise ValueError(f'Tensor set mismatch: {sorted(keys ^ set(merged))[:5]}')
        if any('.lora_' in key for key in keys):
            raise ValueError('LoRA tensors leaked into the export')
        for name in sorted(merged):
            reference = merged[name]
            tensor = handle.get_tensor(name)
            expected_dtype = getattr(torch, DTYPES[header[mapping[name]]['dtype']])
            if tensor.dtype != expected_dtype:
                raise ValueError(f'Dtype drift for {name}: {tensor.dtype} != {expected_dtype}')
            if not torch.isfinite(tensor).all():
                raise ValueError(f'Non-finite tensor on disk: {name}')
            finite += 1
            if tensor.dtype != reference.dtype or not torch.equal(tensor, reference):
                raise ValueError(f'Tensor differs from the in-memory merge: {name}')
            exact += 1
            content.update(name.encode())
            content.update(str(tensor.dtype).encode())
            content.update(str(tuple(tensor.shape)).encode())
            content.update(payload(tensor))
        reversed_mapping = {value: key for key, value in mapping.items()}
        # The 565 untouched tensors must be byte-identical to the OpenPI source.
        unchanged = 0
        with safe_open(str(SOURCE / 'model.safetensors'), framework='pt', device='cpu') as src:
            for name in sorted(merged):
                prefix = mapping[name][:-len('.weight')]
                if prefix in lora_targets:
                    continue
                if not torch.equal(handle.get_tensor(name), src.get_tensor(mapping[name])):
                    raise ValueError(f'Unchanged tensor was altered: {name}')
                unchanged += 1
            # Re-derive every merged layer from the source file, independently of
            # the in-memory result, and require bit equality with the export.
            recomputed = 0
            for prefix in sorted(lora_targets):
                name = reversed_mapping[prefix + '.weight']
                low_rank = src.get_tensor(prefix + '.lora_A').float()
                high_rank = src.get_tensor(prefix + '.lora_B').float()
                base = src.get_tensor(prefix + '.weight')
                expected = (base.float() + high_rank @ low_rank * scale).to(base.dtype)
                if not torch.equal(handle.get_tensor(name), expected):
                    raise ValueError(f'Merge formula mismatch for {name}')
                recomputed += 1
    return dict(verified=exact, finite=finite, unchanged=unchanged, recomputed=recomputed,
                tensor_content_sha256=content.hexdigest())


def delta_magnitudes(mapping, lora_targets, scale, sample=24):
    """Compare our merge delta with the 15000 export's, if that file is available.

    Both checkpoints descend from the same pi05 base, so the per-layer relative
    magnitude of the two LoRA deltas should be in the same ballpark; a wrong
    scaling factor would show up as a factor-of-two (or worse) gap.
    """
    if not PREVIOUS_EXPORT.is_file():
        return None
    from safetensors import safe_open

    reversed_mapping = {value: key for key, value in mapping.items()}
    layers = []
    with safe_open(str(SOURCE / 'model.safetensors'), framework='pt', device='cpu') as src, \
            safe_open(str(PREVIOUS_EXPORT), framework='pt', device='cpu') as merged_15000:
        available = set(merged_15000.keys())
        for prefix in sorted(lora_targets)[:sample]:
            key = prefix + '.weight'
            if key not in available or key not in reversed_mapping:
                continue
            base = src.get_tensor(key).float()
            previous = (merged_15000.get_tensor(key).float() - base).norm().item()
            low_rank = src.get_tensor(prefix + '.lora_A').float()
            high_rank = src.get_tensor(prefix + '.lora_B').float()
            current = (high_rank @ low_rank * scale).norm().item()
            norm = base.norm().item()
            layers.append((previous / norm, current / norm))
    if not layers:
        return None
    return dict(
        layer_15000_relative_delta_mean=round(
            sum(item[0] for item in layers) / len(layers), 6),
        layer_30000_relative_delta_mean=round(
            sum(item[1] for item in layers) / len(layers), 6),
        sampled_layers=len(layers))


def write_text_bundle(metadata, header, mapping, lora_targets, scale, stats,
                      verification, cross_check):
    """Assets, tokenizer, statistics, configs and manifests."""
    tensor_dtypes = {}
    for source in mapping.values():
        name = DTYPES[header[source]['dtype']]
        tensor_dtypes[name] = tensor_dtypes.get(name, 0) + 1

    # Normalization assets stay exactly as the training job wrote them.
    norm_path = SOURCE / 'assets/local/panthera_pillbottle/norm_stats.json'
    norm_stats = json.loads(norm_path.read_text())['norm_stats']
    statistics = {'private': {'proprio': norm_stats['state'],
                              'action': norm_stats['actions']}}
    write_json(SOURCE / 'dataset_statistics.json', statistics)
    # The repository-level statistics file must stay interchangeable with this bundle.
    repo_statistics = ROOT / 'checkpoints/dataset_statistics.json'
    statistics_exact = (not repo_statistics.is_file()
                        or json.loads(repo_statistics.read_text()) == statistics)

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

    # Standalone FluxVLA architecture, same shape as the 6000 bundle's file.
    from mmengine import Config
    config = Config.fromfile(str(INFERENCE_CONFIG))
    native = dict(config.inference_model)
    native.update(name_mapping=None, params_to_change_dtype=[], strict_mapping=True)
    Config(dict(model=native)).dump(str(SOURCE / 'model_config.py'))

    training_config = jsonable(metadata['config'])
    write_json(SOURCE / 'training_config.json', training_config)

    source_digest = None
    if SOURCE_MANIFEST.is_file():
        for line in SOURCE_MANIFEST.read_text().splitlines():
            parts = line.split()
            if len(parts) == 2 and parts[1] == 'model.safetensors':
                source_digest = parts[0]
    weight_path = SOURCE / WEIGHTS
    weight_digest = sha256_file(weight_path)
    relative = [item[1] / item[2] for item in stats]
    merge = dict(
        format='openpi_lora_merged_fluxvla',
        source_checkpoint=str(SOURCE),
        source_step=STEP,
        source_experiment=metadata['config'].get('exp_name'),
        source_config=metadata['config'].get('name'),
        source_global_step=metadata.get('global_step'),
        lora_rank=16,
        lora_alpha=32.0,
        merge_scale=scale,
        merge_precision='fp32_matmul_then_cast_to_source_dtype',
        merged_layers=len(lora_targets),
        unchanged_tensors=812 - len(lora_targets),
        output_dtype='preserve',
        tensor_dtypes=tensor_dtypes,
        tensor_count=812,
        weight_size_bytes=weight_path.stat().st_size,
        source_sha256=source_digest,
        output_sha256=weight_digest,
        tensor_content_sha256=verification.get('tensor_content_sha256'),
        assets_copied=True,
    )
    write_json(SOURCE / 'export_manifest.json', merge)

    mapping_payload = dict(
        source=str(SOURCE),
        tensors=812,
        strict_load=True,
        source_tensors_unused=[],
        lora=dict(rank=16, alpha=32.0, scale=scale,
                  layers=sorted(lora_targets)),
        mapping=mapping,
    )
    write_json(SOURCE / 'conversion_manifest.json', mapping_payload)

    write_json(SOURCE / 'inference_manifest.json', dict(
        schema_version=1,
        framework='fluxvla',
        source_step=STEP,
        source_checkpoint=str(SOURCE),
        weights_file=WEIGHTS,
        norm_stats_file='assets/local/panthera_pillbottle/norm_stats.json',
        statistics_file='dataset_statistics.json',
        model_config_file='model_config.py',
        inference_config='configs/pi05/pi05_panthera_lora_30000_inference.py',
        deploy_preset=DEPLOY_PRESET,
        tokenizer_root='.',
        action_horizon=50,
        action_dim=7,
        internal_action_dim=32,
        num_denoising_steps=10,
        action_steps=10,
        rtc_prefix_len=0,
        lora_rank_at_training=16,
        lora_alpha=32.0,
        delta_action_mask=[True] * 6 + [False],
        action_space='six joint deltas (rad) plus an absolute gripper in [0, 1]',
        normalization='q01/q99, no clipping',
        extra_delta_transform=True,
        max_token_len=200,
        image_order=['head', 'wrist', 'masked_black'],
        state_dim=7,
        publish_rate_hz=30,
    ))

    verification = dict(
        source_checkpoint=str(SOURCE),
        source_sha256=source_digest,
        output_sha256=weight_digest,
        source_tensors=len(header),
        output_tensors=812,
        lora_rank=16,
        lora_alpha=32.0,
        merge_scale=scale,
        merged_layers=len(lora_targets),
        unchanged_tensors=812 - len(lora_targets),
        lora_tensors_remaining=0,
        tensor_dtypes=tensor_dtypes,
        lora_delta_relative_norm=dict(
            mean=round(sum(relative) / len(relative), 6),
            max=round(max(relative), 6),
            min=round(min(relative), 6)),
        statistics_exact=True,
        tokenizer_sha256=tokenizer_digest,
        previous_export_cross_check=cross_check,
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
    text = f"""# Panthera PI0.5 LoRA step {STEP} — FluxVLA 推理包

来源：`{SOURCE.name}`（OpenPI 原生训练产物）。`metadata.pt` 记录的训练配置为
`{merge['source_config']}`，实验 `{merge['source_experiment']}`，从 step 15000 续训到
step {STEP}（4 卡 A100，batch 24）。原始 `model.safetensors`、`metadata.pt`、
`optimizer.pt` 未做任何修改。

LoRA rank 16 / alpha 32，共 247 个线性层（PaliGemma LM 121 + action expert 126）。
合并按 FP32 计算 `W + (32/16) * B @ A`，再恢复每个张量原有的 BF16/FP32 精度；
其余 565 个张量与源权重逐 bit 相同。导出 812 个张量，无 `lora_*` 残留。

## 文件

- `model_fluxvla.safetensors`：812 个 FluxVLA 原生命名的推理权重。
- `model_config.py`：独立的 FluxVLA 模型结构（无配置继承依赖）。
- `dataset_statistics.json`：FluxVLA 统计量（`private.proprio` / `private.action`）。
- `assets/local/panthera_pillbottle/norm_stats.json`：OpenPI 原始统计量，两者数值一致。
- `tokenizer/tokenizer.model`、`paligemma_tokenizer.model`：PaliGemma 分词器（两种布局）。
- `training_config.json`：源的 OpenPI 训练配置（从 `metadata.pt` 抽出）。
- `export_manifest.json`：合并来源与配方。
- `conversion_manifest.json`：812 条张量名映射。
- `inference_manifest.json`：输入输出语义与配套配置路径。
- `verification.json`：离线校验结果。
- `SHA256SUMS`：导出文件校验和（源文件见 `../{SOURCE.name}.sha256`）。

## 可复现性

`model_fluxvla.safetensors` 表头只写一个 `format` 键、张量顺序固定，避免 safetensors
元数据哈希顺序造成的文件级哈希漂移；与文件布局无关的张量指纹记录在
`export_manifest.json` 的 `tensor_content_sha256`。合并是 FP32 计算，换 BLAS 或线程数
可能在回写 BF16/FP32 前出现 1 ULP 级差异。

## 动作语义

6 个关节增量（rad）+ 1 个绝对夹爪。训练时 `extra_delta_transform=true`，因此必须使用
`delta_action_mask=[True]*6+[False]` 的 delta 解码器，即
`configs/pi05/pi05_panthera_lora_30000_inference.py`。**不要**复用 6000 全参微调包的
all-False 配置，那个 checkpoint 的训练约定不同。

## 用法

```bash
# smoke 不打开硬件与相机
/home/sunteng/miniconda3/envs/fluxvla/bin/python deploy/launch_panthera.py smoke \\
  --config {DEPLOY_PRESET}
```

## 未做的验证

本机没有 NVIDIA 驱动（`nvidia-smi` 失败），只做了离线校验：按 `deploy/rollout_panthera.py`
的调用序列做 strict 加载、逐张量数值一致、有限性、未改动张量与源文件逐 bit 相同、tokenizer
与统计量一致性。CUDA 前向、OpenPI 与 FluxVLA 的数值对照、仿真/实机成功率均未评估。
"""
    (SOURCE / 'README.md').write_text(text)


def write_checksums():
    lines = []
    for relative in DERIVED_FILES:
        path = SOURCE / relative
        if path.is_file():
            lines.append(f'{sha256_file(path)}  {relative}')
    (SOURCE / 'SHA256SUMS').write_text('\n'.join(lines) + '\n')


def main():
    torch = prepare_torch()
    if not (SOURCE / 'model.safetensors').is_file():
        raise FileNotFoundError(SOURCE / 'model.safetensors')
    if (SOURCE / WEIGHTS).exists():
        raise FileExistsError(f'{SOURCE / WEIGHTS} already exists; remove it first')
    if not INFERENCE_CONFIG.is_file():
        raise FileNotFoundError(INFERENCE_CONFIG)

    log(f'Reading {SOURCE / "metadata.pt"}')
    metadata = load_metadata(torch)
    config = metadata['config']
    model_config = config['model']
    if metadata['global_step'] != STEP:
        raise ValueError(f'Unexpected step {metadata["global_step"]}')
    if model_config['pytorch_lora_rank'] != 16 or model_config['pytorch_lora_alpha'] != 32.0:
        raise ValueError(f'Unexpected LoRA configuration: {model_config}')
    if model_config['action_dim'] != 32 or model_config['action_horizon'] != 50:
        raise ValueError(f'Unexpected action configuration: {model_config}')
    if not model_config['pi05'] or not model_config['discrete_state_input']:
        raise ValueError('Expected a pi05 model with discrete state input')
    if config['data']['extra_delta_transform'] is not True:
        raise ValueError('Expected extra_delta_transform=true (delta joint actions)')
    scale = model_config['pytorch_lora_alpha'] / model_config['pytorch_lora_rank']
    log(f'step={metadata["global_step"]} config={config["name"]} '
        f'exp={config["exp_name"]} lora_scale={scale}')

    header, aliases = read_header(SOURCE / 'model.safetensors')
    lora_targets = {key.rsplit('.lora_', 1)[0] for key in header if '.lora_' in key}
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

    checks = verify_written(torch, merged, header, mapping, lora_targets, scale)
    log(f'Verified on disk: {checks}')
    del merged
    gc.collect()

    log('Re-loading the export through the deploy path (CPU)')
    from safetensors.torch import load_file
    weights = load_file(str(SOURCE / WEIGHTS), device='cpu')
    reloaded = check_strict_load(torch, weights)
    del weights
    gc.collect()

    merge_manifest, verification = write_text_bundle(
        metadata, header, mapping, lora_targets, scale, stats,
        dict(strict_model_load=f'passed on CPU ({reloaded} tensors, deploy sequence)',
             strict_load_tensors=loaded, **checks),
        delta_magnitudes(mapping, lora_targets, scale))
    write_readme(merge_manifest)
    write_checksums()

    summary = dict(output=str(SOURCE / WEIGHTS),
                   tensors=812,
                   merged_layers=len(lora_targets),
                   merge_scale=scale,
                   weight_bytes=merge_manifest['weight_size_bytes'],
                   weight_sha256=merge_manifest['output_sha256'],
                   strict_load=f'passed ({reloaded} tensors)',
                   files=[name for name in DERIVED_FILES if (SOURCE / name).is_file()])
    log(json.dumps(summary, indent=2))


if __name__ == '__main__':
    main()
