#!/usr/bin/env python3
"""Send one synthetic observation to the FG-WVAM argmax server."""
import base64
import io
import json
import urllib.request

import numpy as np
from PIL import Image


def img_b64(seed: int) -> str:
    rng = np.random.default_rng(seed)
    arr = rng.integers(0, 255, (256, 256, 3), dtype=np.uint8)
    buf = io.BytesIO()
    Image.fromarray(arr).save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("ascii")


frames = [[img_b64(1000 * t + v) for v in range(4)] for t in range(2)]
payload = {
    "prompt": "navigate to the barrel",
    "frames": frames,
    "pose": [0.0] * 7,
    "poses": [[0.0] * 7, [0.0] * 7],
    "pose_steps": [0.0, 1.0],
    "step": 1,
    "seed": 42,
}
req = urllib.request.Request(
    "http://127.0.0.1:18099/act",
    data=json.dumps(payload).encode("utf-8"),
    headers={"Content-Type": "application/json"},
)
with urllib.request.urlopen(req, timeout=600) as resp:
    print("status", resp.status)
    print(resp.read().decode("utf-8")[:4000])