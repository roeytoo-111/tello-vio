import sys, os, json, subprocess, urllib.request, trimesh
def curl(url, path):
    subprocess.run(['curl','-sL','-A','Mozilla/5.0','-o',path,url], check=True)
    if os.path.getsize(path) < 50: raise RuntimeError(f'tiny file {path}')
def pack(asset, outdir, res='1k'):
    subprocess.run(["curl","-sL","-A","Mozilla/5.0","-o",f"_{asset}.json",f"https://api.polyhaven.com/files/{asset}"],check=True); files = json.load(open(f"_{asset}.json"))
    inc = files['gltf'][res]['gltf']
    d = os.path.join(outdir, '_'+asset); os.makedirs(os.path.join(d,'textures'), exist_ok=True)
    curl(inc['url'], os.path.join(d, f'{asset}_{res}.gltf'))
    for rel, meta in inc['include'].items():
        os.makedirs(os.path.dirname(os.path.join(d, rel)), exist_ok=True)
        curl(meta['url'], os.path.join(d, rel))
    scene = trimesh.load(os.path.join(d, f'{asset}_{res}.gltf'), force='scene')
    glb = os.path.join(outdir, asset + '.glb'); scene.export(glb)
    print(f'{asset}: {os.path.getsize(glb)//1024} KB, extents(m) {scene.extents.round(2).tolist()}')
    return glb
if __name__ == '__main__':
    pack(sys.argv[1], sys.argv[2])
