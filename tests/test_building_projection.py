"""The camera model and the facade homography must be each other's inverse.

A checkerboard painted on a synthetic wall, rendered into a 640x640 fov-90 photo by
`camera.project`, must come back out of `walls.warp_facade`. If this fails, textures on walls
are wrong by construction, not by pose. (Ported from shadowCity2's tools/pilot/test_projection.py.)
"""
import math
from pathlib import Path

import numpy as np
import pytest

cv2 = pytest.importorskip("cv2")
from ducklidar import facade
from ducklidar.building import camera, geometry, walls


def _facade(x0, y0, heading_deg, width, height, z0):
    a = math.radians(heading_deg); n = np.array([math.cos(a), math.sin(a), 0.0])
    u = np.array([n[1], -n[0], 0.0]); v = np.array([0, 0, 1.0])
    return dict(n=n, u=u, v=v, p0=np.array([x0, y0, z0]), s0=0.0, s1=width, t0=0.0, t1=height,
                tris=np.array([]), area=width * height)


def test_project_warp_round_trip(tmp_path):
    F = _facade(1000.0, 2000.0, 90.0, 20.0, 8.0, 5.0)                    # 20 x 8 m wall facing north
    cam = F["p0"] + F["u"] * 10 + F["n"] * 14 + np.array([0, 0, -3.3])   # 14 m away, 1.7 m above the base
    heading = math.degrees(math.atan2(-F["n"][0], -F["n"][1])) % 360; pitch = 6.0
    w, h = walls.tex_size(F)
    board = np.zeros((h, w, 3), np.uint8); sq = 25
    for r in range(h):
        for c in range(w):
            board[r, c] = (230, 230, 230) if ((r // sq) + (c // sq)) % 2 == 0 else (30, 90, 160)
    board[:sq, :sq] = (220, 40, 40)                                      # marks the TOP-LEFT (s0, t1)

    facade.CTX.use(MANIFEST=[], MASKS=tmp_path / "no-masks", SOLID={}, W=640)
    uv, _ = camera.project(geometry.facade_corners(F), cam, heading, pitch, 90.0)
    dst = np.array([[0, 0], [w, 0], [w, h], [0, h]], np.float32)
    M = cv2.getPerspectiveTransform(uv.astype(np.float32), dst)
    photo = cv2.warpPerspective(cv2.cvtColor(board, cv2.COLOR_RGB2BGR), np.linalg.inv(M), (640, 640),
                                borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0))
    f = tmp_path / "synthetic.jpg"; cv2.imwrite(str(f), photo, [cv2.IMWRITE_JPEG_QUALITY, 97])

    man = [dict(file=str(f))]; facade.CTX.use(MANIFEST=man)
    tex, valid = walls.warp_facade(F, dict(cam=cam, heading=heading, pitch=pitch, fov=90.0), 0, man)
    inner = np.zeros((h, w), bool); inner[4:-4, 4:-4] = True
    agree = (np.abs(tex.astype(int) - board.astype(int)).mean(axis=2)[inner] < 40).mean()
    assert agree > 0.9, f"checkerboard recovered on only {agree:.1%} of the wall"
    assert np.all(np.abs(tex[8, 8].astype(int) - [220, 40, 40]) < 40), "top-left marker moved"
