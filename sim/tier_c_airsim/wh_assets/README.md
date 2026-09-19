# Warehouse props (CC0, Poly Haven)

Self-contained GLB models used by `warehouse.py` to dress the Tier C scene.
All are **CC0** from Poly Haven (polyhaven.com) — no attribution required,
free for any use. Each was downloaded as glTF (1k textures) and packed into a
single binary `.glb` with `fetch_assets.py` (trimesh).

Regenerate/refresh them with:

    pip install trimesh pillow
    python3 fetch_assets.py wooden_crate_01 .
    python3 fetch_assets.py cardboard_box_01 .
    # ... etc for each prop below

Props: wooden_crate_01, wooden_crate_02, cardboard_box_01, plastic_crate_01,
Barrel_01.
