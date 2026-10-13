"""Immutable, hash-verified float MiniLM assets for real CPU adaptation."""
from __future__ import annotations

import hashlib
import json
import logging
import os
from pathlib import Path
import tempfile
import urllib.request

from .neural import MODEL_ID, MODEL_REVISION

LOGGER = logging.getLogger(__name__)
FLOAT_ASSETS = {
    "config.json": ("git-sha1", "72b987fd805cfa2b58c4c8c952b274a11bfd5a00", 612),
    "model.safetensors": ("sha256", "53aa51172d142c89d9012cce15ae4d6cc0ca6895895114379cacb4fab128d9db", 90_868_376),
    "tokenizer.json": ("git-sha1", "cb202bfe2e3c98645018a6d12f182a434c9d3e02", 466_247),
    "tokenizer_config.json": ("git-sha1", "c79f2b6a0cea6f4b564fed1938984bace9d30ff0", 350),
    "special_tokens_map.json": ("git-sha1", "e7b0375001f109a6b8873d756ad4f7bbb15fbaa5", 112),
}


def verify_float_asset(path: Path, name: str) -> str:
    kind, expected, size = FLOAT_ASSETS[name]
    if path.stat().st_size != size:
        raise ValueError(f"Unexpected asset size: {name}")
    content = path.read_bytes()
    sha256 = hashlib.sha256(content).hexdigest()
    digest = sha256 if kind == "sha256" else hashlib.sha1(f"blob {size}\0".encode() + content).hexdigest()
    if digest != expected:
        raise ValueError(f"Asset hash mismatch: {name}")
    return sha256


def download_float_minilm(cache_dir: Path) -> Path:
    """Download only safetensors/JSON; never remote Python or pickle weights."""
    target_dir = Path(cache_dir) / "models" / f"{MODEL_REVISION}-float"
    target_dir.mkdir(parents=True, exist_ok=True)
    hashes = {}
    for name, (_, _, size) in FLOAT_ASSETS.items():
        target = target_dir / name
        if not target.exists():
            LOGGER.info("Downloading pinned float asset %s", name)
            url = f"https://huggingface.co/{MODEL_ID}/resolve/{MODEL_REVISION}/{name}"
            request = urllib.request.Request(url, headers={"User-Agent": "EvidenceForge/0.2"})
            fd, temp_name = tempfile.mkstemp(prefix=".download-", dir=target_dir)
            temporary = Path(temp_name)
            try:
                with os.fdopen(fd, "wb") as output, urllib.request.urlopen(request, timeout=90) as response:
                    while block := response.read(1024 * 1024):
                        output.write(block)
                        if output.tell() > size:
                            raise ValueError(f"Oversized asset: {name}")
                verify_float_asset(temporary, name)
                os.replace(temporary, target)
            except BaseException:
                temporary.unlink(missing_ok=True)
                raise
        hashes[name] = verify_float_asset(target, name)
    manifest = {"model_id": MODEL_ID, "revision": MODEL_REVISION, "sha256": hashes,
                "source": f"https://huggingface.co/{MODEL_ID}/tree/{MODEL_REVISION}", "license": "Apache-2.0"}
    (target_dir / "assets_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return target_dir
