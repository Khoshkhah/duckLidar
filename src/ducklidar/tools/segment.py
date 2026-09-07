"""TOOL — SAM 3 over a photo folder: occluder masks, the building silhouette, facade elements.

**Not part of a build.** `ducklidar.building3d` never calls this; it reads the folders this
writes. Needs a GPU and `transformers` with SAM 3 — which is exactly why it is a tool:

    python -m ducklidar.tools.segment --photos out/pilot_netloft_photos
    python -m ducklidar.tools.segment --photos out/photos --only 3 7   # redo two photos

**One pass per photo, both prompt sets.** The pilot ran two scripts over the same images and
paid the image load and the processor twice; the two passes are 60-70 % of a building's
wall-clock and the second one re-read every photo for nothing. Same prompts, same
thresholds, same outputs — one loop.

Writes, under `--photos`:

    masks/NN.png            occluders, 255 = do not paint this pixel on a wall
    masks/NN_building.png   the building silhouette the camera pose is fitted to
    elements/NN.json        instances: label, score, bbox [x0,y0,x1,y1] px, area px
    elements/NN.png         uint16 instance-id map (0 = none, id = index in the json + 1)
    masks_sheet.png         the occluders drawn over the photos, to check

A photo with almost no "sky" is an indoor photosphere: it is flagged `unused` in the
manifest and skipped, which is also what the build then skips.

Element scores are lower than occluder scores (0.25 vs 0.3) on purpose: a missed window
costs more than a spurious one, and the wall grammar (rows, repeats) can reject a spurious
one later.
"""
import argparse
import json
import sys
import time
from collections import Counter
from pathlib import Path

import numpy as np

OCCLUDERS = ["car", "truck", "van", "bus", "bicycle", "motorcycle", "person", "tree", "bush",
             "plant", "pole", "street lamp", "sign", "flower pot", "bench", "bollard",
             "umbrella", "sky"]
BUILDING = ["building", "house", "facade"]
ELEMENTS = ["window", "door", "garage door", "awning", "storefront", "sign", "balcony", "downspout"]
OCC_SCORE, EL_SCORE, DILATE = 0.3, 0.25, 5
MIN_AREA = 60          # px: a smaller instance is noise, not an element
SKY_FRAC = 0.02        # less sky than this and the camera was indoors


def load_model(model="facebook/sam3"):
    """SAM 3 on the GPU. Imported here so `ducklidar.tools` costs nothing to import."""
    import torch
    from transformers import Sam3Model, Sam3Processor
    t0 = time.time()
    proc = Sam3Processor.from_pretrained(model)
    net = Sam3Model.from_pretrained(model, dtype=torch.float16).to("cuda").eval()
    print(f"SAM 3 ready in {time.time()-t0:.0f}s")
    return proc, net


def prompt(proc, net, im, text, score):
    """One text prompt on one image -> (masks (k,h,w) bool, scores (k,))."""
    import torch
    inputs = proc(images=im, text=text, return_tensors="pt").to("cuda", torch.float16)
    with torch.no_grad():
        out = net(**inputs)
    res = proc.post_process_instance_segmentation(
        out, threshold=score, mask_threshold=0.5,
        target_sizes=inputs.get("original_sizes").tolist())[0]
    if not len(res["masks"]):
        return np.zeros((0, im.height, im.width), bool), np.zeros(0)
    return res["masks"].cpu().numpy().astype(bool), res["scores"].cpu().numpy()


def segment_photo(proc, net, path, occluders=OCCLUDERS, building=BUILDING, elements=ELEMENTS):
    """Both prompt sets over ONE image load -> dict(occ, bld, instances, idmap, found, indoor)."""
    import cv2
    from PIL import Image
    im = Image.open(path).convert("RGB")
    occ = np.zeros((im.height, im.width), bool); found = {}
    for text in occluders:
        m, _ = prompt(proc, net, im, text, OCC_SCORE)
        if len(m):
            m = m.any(0); occ |= m; found[text] = int(m.sum())
    indoor = found.get("sky", 0) < SKY_FRAC * occ.size
    occ = cv2.dilate(occ.astype(np.uint8),
                     np.ones((2 * DILATE + 1, 2 * DILATE + 1), np.uint8)).astype(bool)
    if indoor:
        return dict(im=im, occ=occ, bld=None, instances=[], idmap=None, found=found, indoor=True)

    bld = np.zeros_like(occ)
    for text in building:
        m, _ = prompt(proc, net, im, text, OCC_SCORE)
        if len(m): bld |= m.any(0)
    found["building"] = int(bld.sum())

    inst = []; idmap = np.zeros((im.height, im.width), np.uint16)
    for text in elements:
        masks, scores = prompt(proc, net, im, text, EL_SCORE)
        for m, sc in zip(masks, scores.tolist()):
            if m.sum() < MIN_AREA: continue
            ys, xs = np.nonzero(m); k = len(inst) + 1
            inst.append(dict(id=k, label=text, score=round(float(sc), 3),
                             bbox=[int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1],
                             area=int(m.sum())))
            idmap[m & (idmap == 0)] = k
    return dict(im=im, occ=occ, bld=bld & ~occ, instances=inst, idmap=idmap, found=found, indoor=False)


def main(argv=None):
    import cv2
    from PIL import Image
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--photos", required=True, help="the folder a fetcher wrote (holds the manifest)")
    ap.add_argument("--only", type=int, nargs="*", default=[], help="photo indices to (re)do")
    ap.add_argument("--indoor", type=int, nargs="*", default=[], help="indices to skip outright")
    ap.add_argument("--model", default="facebook/sam3")
    ap.add_argument("--force", action="store_true", help="redo photos already segmented")
    a = ap.parse_args(argv)

    photos = Path(a.photos)
    masks = photos / "masks"; els = photos / "elements"
    masks.mkdir(parents=True, exist_ok=True); els.mkdir(parents=True, exist_ok=True)
    mf = photos / "manifest_all.json"
    if not mf.exists(): mf = photos / "streetview/manifest.json"
    if not mf.exists(): mf = photos / "manifest.json"
    man = [p for p in json.load(open(mf)) if "file" in p]
    only = set(a.only); indoor = set(a.indoor)

    proc, net = load_model(a.model)
    tiles, flagged = [], []
    for i, p in enumerate(man):
        if i in indoor or p.get("unused") or (only and i not in only): continue
        if not only and not a.force and (masks / f"{i:02d}_building.png").exists(): continue
        r = segment_photo(proc, net, p["file"])
        cv2.imwrite(str(masks / f"{i:02d}.png"), r["occ"].astype(np.uint8) * 255)
        if r["indoor"]:
            p["unused"] = "indoor photosphere (no sky)"; flagged.append(i)
            print(f"  #{i:2d} INDOOR (no sky) — flagged unused"); continue
        cv2.imwrite(str(masks / f"{i:02d}_building.png"), r["bld"].astype(np.uint8) * 255)
        json.dump(r["instances"], open(els / f"{i:02d}.json", "w"))
        cv2.imwrite(str(els / f"{i:02d}.png"), r["idmap"])
        c = Counter(x["label"] for x in r["instances"])
        print(f"  #{i:2d} occluded {r['occ'].mean():5.1%}  "
              + ", ".join(f"{k} {v//1000}k" for k, v in sorted(r["found"].items(), key=lambda kv: -kv[1])[:5])
              + f"  |  {len(r['instances']):3d} elements  " + ", ".join(f"{k} {v}" for k, v in c.most_common()))
        arr = np.asarray(r["im"]).copy()
        arr[r["occ"]] = (0.35 * arr[r["occ"]] + 0.65 * np.array([216, 27, 96])).astype(np.uint8)
        t = Image.fromarray(arr); t.thumbnail((320, 320)); tiles.append(t)

    if tiles:
        cols = 6; rows = (len(tiles) + cols - 1) // cols
        sheet = Image.new("RGB", (cols * 324, rows * 324), "white")
        for k, t in enumerate(tiles): sheet.paste(t, ((k % cols) * 324, (k // cols) * 324))
        sheet.save(photos / "masks_sheet.png"); print("wrote", photos / "masks_sheet.png")
    if flagged:
        json.dump(man, open(mf, "w"), indent=1)
        print(f"manifest updated: {len(flagged)} indoor photos flagged unused")
    return 0


if __name__ == "__main__":
    sys.exit(main())
