"""The checkpoint contract (guide 19) -- sim-to-real is auditable or it is
folklore.

A bundle carries: weights (actor + critics + targets), optimizer and noise
state (resumability), the FULL observation spec + its hash, the action map
(index -> axis, sign convention, scale), control rate, reward version, the
resolved run config + its hash, git SHA, seed, env-step count, library
versions, and RNG states. Filenames are versioned
(`{arm}_s{seed}_step{n}_tv{score}.pt`), a best.json manifest records WHICH
checkpoint was selected and WHY (eval metric, not recency), and loads pin
map_location -- each rule is a counterexample from the reference codebases
(offline_training_recipe.md section 5, stage 4).

Export: the ACTOR ALONE to ONNX (opset >= 17, deterministic), regenerated at
promotion; the torch bundle stays the source of truth. Normalisation and
action unscaling are NOT in the graph -- they live in the shared modules the
inference node imports (recipe 6.5).
"""
import json
import os
import random
import subprocess
import time
from typing import Optional

import numpy as np
import torch

from .networks import Actor


def _git_sha() -> str:
    try:
        return subprocess.run(
            ['git', 'rev-parse', 'HEAD'], capture_output=True, text=True,
            timeout=5, cwd=os.path.dirname(os.path.abspath(__file__))
        ).stdout.strip() or 'unknown'
    except Exception:
        return 'unknown'


def library_versions() -> dict:
    import gymnasium
    return {'torch': torch.__version__, 'numpy': np.__version__,
            'gymnasium': gymnasium.__version__}


def save_bundle(path: str, *, trainer, noise, cfg, obs_spec, action_map: dict,
                env_config: dict, env_step: int, eval_snapshot: dict,
                curriculum_stage: int = 0) -> str:
    bundle = {
        'format_version': 1,
        'arm': cfg.arm,
        'seed': cfg.seed,
        'env_step': int(env_step),
        'trainer': trainer.state_dict(),
        'noise': noise.state_dict(),
        'obs_spec': obs_spec.to_dict(),
        'obs_spec_hash': obs_spec.spec_hash(),
        'action_map': action_map,
        'control_rate_hz': 1.0 / obs_spec.control_dt_s,
        'reward_version': env_config.get('reward_version', 'faithful'),
        'config': cfg.to_dict(),
        'config_hash': cfg.config_hash(),
        'env_config': env_config,
        'git_sha': _git_sha(),
        'library_versions': library_versions(),
        'eval': eval_snapshot,
        'curriculum_stage': int(curriculum_stage),
        'rng': {
            'python': random.getstate(),
            'numpy_legacy': np.random.get_state(),
            'torch': torch.get_rng_state(),
        },
        'saved_at': time.strftime('%Y-%m-%dT%H:%M:%S'),
    }
    torch.save(bundle, path)
    return path


def load_bundle(path: str, expect_obs_spec=None) -> dict:
    bundle = torch.load(path, map_location='cpu', weights_only=False)
    if expect_obs_spec is not None:
        got = bundle.get('obs_spec_hash')
        want = expect_obs_spec.spec_hash()
        if got != want:
            # The refusal is the contract (guide 19): an actor is
            # meaningless without the exact input spec it trained under.
            raise ValueError(
                f'checkpoint observation spec {got} != expected {want}; '
                f'bundle spec: {bundle.get("obs_spec")}')
    return bundle


def actor_from_bundle(bundle: dict) -> Actor:
    cfg = bundle['config']
    obs_dim = bundle['obs_spec']['dim']
    act_dim = bundle['obs_spec'].get('action_dim', 2)
    actor = Actor(obs_dim, act_dim, cfg['nets']['hidden_actor'],
                  action_scale=cfg['train']['action_scale'],
                  final_init=cfg['nets']['final_init'],
                  faithful_init=cfg['nets']['faithful_init'])
    actor.load_state_dict(bundle['trainer']['actor'])
    actor.eval()
    return actor


def versioned_name(arm: str, seed: int, step: int, tv: float) -> str:
    return f'{arm}_s{seed}_step{step}_tv{int(round(100 * tv)):03d}.pt'


def update_best_manifest(run_dir: str, entry: dict) -> None:
    manifest_path = os.path.join(run_dir, 'best.json')
    entry = dict(entry, selected_by='eval time-in-view, then return',
                 updated_at=time.strftime('%Y-%m-%dT%H:%M:%S'))
    with open(manifest_path, 'w') as f:
        json.dump(entry, f, indent=2)


def export_onnx(bundle: dict, out_path: str,
                atol: float = 1e-5) -> Optional[str]:
    """Actor -> ONNX (opset 17, dynamic batch) with a torch-vs-onnxruntime
    parity check. Returns the path, or None if onnx is unavailable."""
    try:
        import onnx  # noqa: F401
        import onnxruntime as ort
    except ImportError:
        return None
    actor = actor_from_bundle(bundle)
    obs_dim = bundle['obs_spec']['dim']
    dummy = torch.zeros(1, obs_dim, dtype=torch.float32)
    torch.onnx.export(
        actor, dummy, out_path, opset_version=17, input_names=['obs'],
        output_names=['action'], dynamic_axes={'obs': {0: 'batch'},
                                               'action': {0: 'batch'}})
    sess = ort.InferenceSession(out_path, providers=['CPUExecutionProvider'])
    rng = np.random.default_rng(0)
    x = rng.uniform(-1.0, 1.0, size=(32, obs_dim)).astype(np.float32)
    with torch.no_grad():
        ref = actor(torch.from_numpy(x)).numpy()
    got = sess.run(None, {'obs': x})[0]
    err = float(np.abs(ref - got).max())
    if err > atol:
        raise RuntimeError(f'ONNX parity check failed: max err {err}')
    return out_path
