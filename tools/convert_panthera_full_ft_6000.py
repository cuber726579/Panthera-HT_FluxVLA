"""Convert the retained OpenPI full FT export to a separate FluxVLA bundle."""
import hashlib
import json
from pathlib import Path
import shutil
import torch
from accelerate import init_empty_weights
from mmengine import Config
from safetensors import safe_open
from safetensors.torch import save_file
from fluxvla.engines import build_vla_from_cfg

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / 'checkpoints/openpi_panthera_full-ft_6000_a100'
OUTPUT = ROOT / 'checkpoints/fluxvla_panthera_full-ft_6000_a100'

def main():
    torch.set_num_threads(4)
    if OUTPUT.exists():
        raise FileExistsError(OUTPUT)
    saved = json.loads((SOURCE / 'training_config.json').read_text())
    assert saved['model']['pytorch_lora_rank'] == 0
    assert saved['data']['extra_delta_transform'] is False
    cfg = Config.fromfile(str(ROOT / 'configs/pi05/pi05_paligemma_ur3_full_finetune.py')).model
    cfg.pretrained_name_or_path = None
    with init_empty_weights():
        model = build_vla_from_cfg(cfg)
    model.name_mapping['llm_expert.embed_tokens'] = 'paligemma_with_expert.gemma_expert.lm_head'
    mapped, mapping, used = {}, {}, set()
    with safe_open(SOURCE / 'model.safetensors', framework='pt', device='cpu') as h:
        names, aliases = set(h.keys()), h.metadata() or {}
        if any('lora_A' in key or 'lora_B' in key for key in names):
            raise ValueError('Unexpected LoRA parameters in full FT checkpoint')
        for name, target in model.state_dict().items():
            candidates = [name] + [key for _, key in model._mapped_name_candidates(name)]
            candidates = list(dict.fromkeys(aliases.get(key, key) for key in candidates if aliases.get(key, key) in names))
            if len(candidates) != 1:
                raise ValueError((name, candidates))
            source = candidates[0]
            tensor = h.get_tensor(source)
            if tensor.shape != target.shape or not torch.isfinite(tensor).all():
                raise ValueError((source, name, tensor.shape, target.shape))
            mapped[name], mapping[name] = tensor, source
            used.add(source)
        if names != used:
            raise ValueError(f'Unused source tensors: {names - used}')
    model.load_state_dict(mapped, strict=True, assign=True)
    OUTPUT.mkdir()
    save_file(mapped, str(OUTPUT / 'model.safetensors'), metadata={'format': 'fluxvla', 'source': str(SOURCE)})
    native_cfg = dict(cfg)
    native_cfg.update(name_mapping=None, params_to_change_dtype=[], strict_mapping=True)
    Config(dict(model=native_cfg)).dump(str(OUTPUT / 'model_config.py'))
    shutil.copytree(SOURCE / 'assets', OUTPUT / 'assets')
    shutil.copy2(SOURCE / 'training_config.json', OUTPUT / 'training_config.json')
    tokenizer = ROOT / 'checkpoints/openpi_panthera_8000/paligemma_tokenizer.model'
    shutil.copy2(tokenizer, OUTPUT / 'paligemma_tokenizer.model')
    (OUTPUT / 'tokenizer').mkdir()
    shutil.copy2(tokenizer, OUTPUT / 'tokenizer/tokenizer.model')
    stats_path = OUTPUT / 'assets/local/panthera_pillbottle/norm_stats.json'
    stats = json.loads(stats_path.read_text())['norm_stats']
    (OUTPUT / 'dataset_statistics.json').write_text(json.dumps({'private': {'proprio': stats['state'], 'action': stats['actions']}}, indent=2) + '\n')
    manifest = dict(schema_version=1, framework='fluxvla', source_step=6000,
        source_checkpoint=str(SOURCE), norm_stats_file='assets/local/panthera_pillbottle/norm_stats.json',
        model_config_file='model_config.py', action_horizon=50, action_dim=7, internal_action_dim=32,
        num_denoising_steps=10, action_steps=10, rtc_prefix_len=0, lora_rank_at_training=0,
        delta_action_mask=[False]*7, action_space='absolute joint and gripper targets',
        normalization='q01/q99, no clipping', max_token_len=200,
        image_order=['head', 'wrist', 'masked_black'])
    (OUTPUT / 'inference_manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
    (OUTPUT / 'conversion_manifest.json').write_text(json.dumps(dict(source=str(SOURCE), tensors=len(mapped),
        strict_load=True, source_tensors_unused=[], mapping=mapping), indent=2) + '\n')
    # Reopen output and compare every mapped tensor exactly, including dtype.
    with safe_open(OUTPUT / 'model.safetensors', framework='pt', device='cpu') as h:
        assert set(h.keys()) == set(mapped)
        for name, original in mapped.items():
            actual = h.get_tensor(name)
            assert actual.dtype == original.dtype and torch.equal(actual, original), name
    print(json.dumps(dict(output=str(OUTPUT), target_tensors=len(mapped), source_tensors=len(used),
        strict_load=True, exact_tensor_equality=True, delta_action_mask=[False]*7), indent=2), flush=True)

if __name__ == '__main__':
    main()
