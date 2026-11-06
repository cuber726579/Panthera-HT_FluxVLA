"""Load deployment YAML and replace this process with the robot interpreter."""

import argparse
import os
from pathlib import Path
import re
import shlex
from string import Template

import yaml


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = ROOT / 'deploy/config/openpi_panthera_15000_async.yaml'
PATH_OPTIONS = {'robot_config', 'model_config', 'checkpoint', 'statistics',
                'tokenizer_root', 'log_root', 'policy_python'}
ENV_OVERRIDES = ('INFERENCE_MODE', 'RTC_METHOD', 'RTC_DELAY_STEPS', 'RTC_USE_VJP')


def resolve_path(value, variables):
    path = Path(Template(str(value)).substitute(variables)).expanduser()
    return str(path if path.is_absolute() else ROOT / path)


def build_command(config_path, mode=None, environ=None):
    env = dict(os.environ if environ is None else environ)
    with Path(config_path).open(encoding='utf-8') as source:
        config = yaml.safe_load(source)
    if not isinstance(config, dict) or set(config) != {'runtime', 'rollout'}:
        raise ValueError('Config must contain exactly runtime and rollout mappings')
    runtime, rollout = config['runtime'], config['rollout']
    if not isinstance(runtime, dict) or not isinstance(rollout, dict):
        raise ValueError('runtime and rollout must be mappings')
    if set(runtime) != {'conda_root', 'robot_env', 'policy_env', 'cuda_device'}:
        raise ValueError('runtime requires conda_root, robot_env, policy_env, cuda_device')
    if any(not isinstance(key, str) or not re.fullmatch(r'[a-z][a-z0-9_]*', key)
           for key in rollout):
        raise ValueError('rollout keys must be snake_case command-line option names')
    rollout = dict(rollout)
    variables = {'ROOT': str(ROOT)}
    conda_root = Path(resolve_path(runtime['conda_root'], variables))
    rollout['sdk_root'] = resolve_path(rollout['sdk_root'], variables)
    variables['SDK_ROOT'] = rollout['sdk_root']
    for key in PATH_OPTIONS & rollout.keys():
        rollout[key] = resolve_path(rollout[key], variables)
    rollout.setdefault('policy_python', str(conda_root / 'envs' / runtime['policy_env'] / 'bin/python'))
    if mode is not None:
        rollout['mode'] = mode
    for name in ENV_OVERRIDES:
        if env.get(name):
            rollout[name.lower()] = env[name]

    command = [str(conda_root / 'envs' / runtime['robot_env'] / 'bin/python'),
               str(ROOT / 'deploy/rollout_panthera.py')]
    for key, value in rollout.items():
        values = value if isinstance(value, list) else [value]
        if not values or any(not isinstance(item, (str, int, float, bool)) for item in values):
            raise ValueError(f'Invalid scalar or list for rollout.{key}')
        command.append('--' + key.replace('_', '-'))
        command.extend(str(int(item)) if isinstance(item, bool) else str(item) for item in values)

    env['CUDA_VISIBLE_DEVICES'] = str(runtime['cuda_device'])
    env['PYTHONUNBUFFERED'] = '1'
    env['PYTHONPATH'] = str(ROOT) + (os.pathsep + env['PYTHONPATH'] if env.get('PYTHONPATH') else '')
    return command, env


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode', nargs='?', choices=['check', 'smoke', 'dry-run', 'rollout'])
    parser.add_argument('--config', type=Path, default=DEFAULT_CONFIG,
                        help='deployment YAML (default: %(default)s)')
    parser.add_argument('--print-command', action='store_true',
                        help='print the resolved command without loading models or opening hardware')
    args = parser.parse_args()
    try:
        command, env = build_command(args.config, args.mode)
    except (OSError, ValueError, TypeError, KeyError, yaml.YAMLError) as exc:
        parser.error(f'{args.config}: {exc}')
    if args.print_command:
        print(shlex.join(['env', f'CUDA_VISIBLE_DEVICES={env["CUDA_VISIBLE_DEVICES"]}',
                          f'PYTHONUNBUFFERED={env["PYTHONUNBUFFERED"]}',
                          f'PYTHONPATH={env["PYTHONPATH"]}', *command]))
        return
    os.chdir(ROOT)
    try:
        os.execve(command[0], command, env)
    except OSError as exc:
        parser.error(f'Cannot start robot interpreter {command[0]}: {exc}')


if __name__ == '__main__':
    main()
