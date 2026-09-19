"""Dress a running Project AirSim scene as an enclosed warehouse interior, so
the Tier C follower flies down an aisle with realistic indoor backgrounds
(shelving racks loaded with crates, boxes and barrels) instead of the
packaged outdoor level.

Why this exists: the real deployment is a Tello flying INDOORS in a
warehouse, and the detector inherits the renderer's domain -- an RGB drone
detector trained on outdoor Blocks frames will not transfer to a warehouse.
This builds a warehouse the drone actually flies inside of.

Pieces (verified live on Blocks 1.0.1 / UE 5.7, 2026-09-19):
  * Shell: floor, ceiling and four walls from the packaged `1M_Cube` asset
    (spawn_object; scale is in metres since the cube is 1 m).
  * Racks: uprights + three shelf levels down both sides of a central aisle.
  * Props: CC0 glTF models from wh_assets/ (crates, boxes, a barrel), spawned
    with spawn_object_from_file and arranged on the shelves and floor.
  * Light: the packaged outdoor sun is switched off and the interior is lit
    by ceiling PointLightActors whose intensity is set from the lighting
    class -- so the dataset factory's `--lighting low|medium|high` now
    physically drives the scene, it is no longer just an attestation.

The aisle runs along +x (the follower's forward axis); the flight volume is
y in [-1.5, 1.5], z in [0, -4] (NED), x in [0, 19].
"""
import math
import os
import random

ASSETS = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'wh_assets')
PROPS = ['wooden_crate_01', 'wooden_crate_02', 'cardboard_box_01',
         'plastic_crate_01', 'Barrel_01']

# lighting class -> ceiling point-light intensity (unitless UE point-light
# units; tuned live so the drone and shelves are well exposed without the
# ceiling blowing out).
LIGHTING_INTENSITY = {'low': 3500.0, 'medium': 8000.0, 'high': 18000.0}


def _pose(x, y, z, yaw=0.0):
    return {'translation': {'x': float(x), 'y': float(y), 'z': float(z)},
            'rotation': {'w': math.cos(yaw / 2.0), 'x': 0.0, 'y': 0.0,
                         'z': math.sin(yaw / 2.0)},
            'frame_id': 'DEFAULT_FRAME'}


def build(eng, seed=0, ceiling=True, cube='1M_Cube'):
    """Spawn the shell, racks and props. Returns the number of glTF props."""
    w = eng._world
    rng = random.Random(seed)

    def box(name, cx, cy, cz, sx, sy, sz):
        w.spawn_object(name, cube, _pose(cx, cy, cz), [sx, sy, sz], False)

    # -- room shell: x 0..19 fwd, y -4.6..4.6, z 0 (floor) .. -4 (ceiling) --
    box('wh_floor', 9, 0, 0.05, 22, 10, 0.1)
    if ceiling:
        box('wh_ceil', 9, 0, -4.0, 22, 10, 0.1)
    box('wh_wall_L', 9, -4.6, -2, 22, 0.2, 4)
    box('wh_wall_R', 9, 4.6, -2, 22, 0.2, 4)
    box('wh_wall_B', -1.5, 0, -2, 0.2, 10, 4)
    box('wh_wall_F', 19.5, 0, -2, 0.2, 10, 4)

    # -- shelving racks down both sides of the aisle, loaded with props --
    placements = []
    for side in (-1, 1):
        ry = side * 3.0
        for post_x in range(1, 19, 3):
            box(f'post_{side}_{post_x}', post_x, ry, -1.5, 0.1, 1.0, 3.0)
        for lvl, sz in enumerate((-0.5, -1.7, -2.9)):
            box(f'shelf_{side}_{lvl}', 9.5, ry, sz, 17, 1.0, 0.06)
            for px in range(2, 18, 2):
                placements.append((rng.choice(PROPS),
                                   px + rng.uniform(-0.3, 0.3),
                                   ry + rng.uniform(-0.2, 0.2),
                                   sz - 0.35, rng.uniform(0, math.pi)))
    for _ in range(10):
        placements.append((rng.choice(PROPS), rng.uniform(2, 17),
                           rng.choice([-1.7, 1.7]) + rng.uniform(-0.2, 0.2),
                           -0.3, rng.uniform(0, math.pi)))

    cache = {}
    for i, (asset, x, y, z, yaw) in enumerate(placements):
        if asset not in cache:
            with open(os.path.join(ASSETS, asset + '.glb'), 'rb') as f:
                cache[asset] = f.read()
        w.spawn_object_from_file(f'wh_prop_{i}', 'gltf', cache[asset], True,
                                 _pose(x, y, z, yaw), [1, 1, 1], False)
    return len(placements)


def light(eng, intensity=8000.0):
    """Switch off the outdoor sun and light the interior from the ceiling."""
    w = eng._world
    try:
        w.set_sunlight_intensity(0)
    except Exception:
        pass
    n = 0
    for lx in range(2, 19, 3):
        name = w.spawn_object(f'wh_light_{lx}', 'PointLightActor',
                              _pose(lx, 0, -3.7), [1, 1, 1], False)
        w.set_light_object_intensity(name, float(intensity))
        n += 1
    return n


def build_and_light(eng, seed=0, lighting='medium'):
    """Full warehouse: geometry + interior lighting from the lighting class."""
    n_props = build(eng, seed=seed)
    n_lights = light(eng, LIGHTING_INTENSITY.get(lighting, 8000.0))
    return n_props, n_lights
