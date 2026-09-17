"""Run configuration: the guide section 15 config block, versioned per run.

One YAML per arm under chase_train/config/. Every run writes its resolved
config + hash into the run directory and the checkpoint bundle -- the
Henderson rule (guide 24): every number that moved is recorded.
"""
import copy
import hashlib
import json
from dataclasses import asdict, dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import yaml

ARMS = ('td3', 'ddpg_repaired', 'ddpg_faithful')


@dataclass
class NetsCfg:
    hidden_actor: List[int] = field(default_factory=lambda: [256, 256])
    hidden_critic: List[int] = field(default_factory=lambda: [256, 256])
    final_init: float = 3.0e-3
    faithful_init: bool = False


@dataclass
class NoiseCfg:
    type: str = 'gaussian'            # 'gaussian' | 'ou'
    sigma0: float = 0.3
    sigma_min: float = 0.05
    decay_steps: int = 0              # 0 -> total_steps
    theta: float = 0.15               # OU
    ou_sigma: float = 0.3
    ou_dt: float = 1.0e-2             # keras-rl's silent default, EXPLICIT
    post_scale: bool = False          # faithful: noise added after x60


@dataclass
class TrainCfg:
    total_steps: int = 200_000
    gamma: float = 0.99
    tau: float = 0.005
    batch: int = 64
    buffer: int = 100_000
    actor_lr: float = 3.0e-4
    critic_lr: float = 3.0e-4
    clipnorm: float = 0.0
    start_steps: int = 1000           # uniform warm-start (guide 11.1)
    update_after: int = 1000
    policy_delay: int = 2
    target_noise: float = 0.2
    noise_clip: float = 0.5
    action_scale: float = 1.0
    truncation_as_terminal: bool = False   # ONLY the faithful arm [D-impl 7]
    noise: NoiseCfg = field(default_factory=NoiseCfg)
    demo_prefill_episodes: int = 20   # P-controller prefill (guide 17)
    demo_dir: str = ''                # optional .npz episodes to load too


@dataclass
class EvalCfg:
    every: int = 5000
    episodes_per_scenario: int = 10
    eval_seed0: int = 10_000          # disjoint from training seeds
    plateau_evals: int = 5            # stopping rule (recipe stage 3)
    plateau_epsilon: float = 0.005


@dataclass
class CurriculumStage:
    families: List[str]
    speed_cap: float
    advance_tv: float = 0.90          # guide 20
    demote_tv: float = 0.70           # rolling gate can step back down


@dataclass
class CurriculumCfg:
    enabled: bool = True
    stages: List[CurriculumStage] = field(default_factory=lambda: [
        CurriculumStage(['static'], 0.0),
        CurriculumStage(['static', 'constant_velocity'], 0.4),
        CurriculumStage(['constant_velocity', 'vertical_oscillation',
                         'horizontal_oscillation'], 0.6),
        CurriculumStage(['static', 'constant_velocity',
                         'vertical_oscillation', 'horizontal_oscillation',
                         'aggressive'], 1.0),
    ])


@dataclass
class RunConfig:
    arm: str = 'td3'
    seed: int = 0
    device: str = 'auto'
    out_root: str = 'runs'
    env: Dict = field(default_factory=dict)      # chase_gym.EnvConfig kwargs
    nets: NetsCfg = field(default_factory=NetsCfg)
    train: TrainCfg = field(default_factory=TrainCfg)
    eval: EvalCfg = field(default_factory=EvalCfg)
    curriculum: CurriculumCfg = field(default_factory=CurriculumCfg)

    def __post_init__(self):
        if self.arm not in ARMS:
            raise ValueError(f'arm must be one of {ARMS}, got {self.arm!r}')

    @property
    def n_critics(self) -> int:
        return 2 if self.arm == 'td3' else 1

    def to_dict(self) -> dict:
        return asdict(self)

    def config_hash(self) -> str:
        blob = json.dumps(self.to_dict(), sort_keys=True).encode()
        return hashlib.sha256(blob).hexdigest()[:16]


def _merge(base: dict, override: dict) -> dict:
    out = copy.deepcopy(base)
    for k, v in override.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = _merge(out[k], v)
        else:
            out[k] = v
    return out


def _build(d: dict) -> RunConfig:
    d = copy.deepcopy(d)
    nets = NetsCfg(**d.pop('nets', {}))
    train_d = d.pop('train', {})
    noise = NoiseCfg(**train_d.pop('noise', {}))
    train = TrainCfg(noise=noise, **train_d)
    ev = EvalCfg(**d.pop('eval', {}))
    cur_d = d.pop('curriculum', {})
    stages = [CurriculumStage(**s) for s in cur_d.pop('stages', [])] or None
    cur = CurriculumCfg(**cur_d) if stages is None else \
        CurriculumCfg(stages=stages, **cur_d)
    return RunConfig(nets=nets, train=train, eval=ev, curriculum=cur, **d)


def load(path: str, overrides: Optional[Sequence[str]] = None,
         seed: Optional[int] = None) -> RunConfig:
    """YAML -> RunConfig, with 'a.b.c=value' dot-overrides (YAML-parsed)."""
    with open(path) as f:
        raw = yaml.safe_load(f) or {}
    for ov in overrides or []:
        if '=' not in ov:
            raise ValueError(f'override {ov!r} is not key=value')
        key, val = ov.split('=', 1)
        node: dict = {}
        leaf = node
        parts = key.split('.')
        for p in parts[:-1]:
            leaf[p] = {}
            leaf = leaf[p]
        leaf[parts[-1]] = yaml.safe_load(val)
        raw = _merge(raw, node)
    cfg = _build(raw)
    if seed is not None:
        cfg.seed = seed
    return cfg
