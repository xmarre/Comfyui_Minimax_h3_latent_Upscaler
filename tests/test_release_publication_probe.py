from __future__ import annotations

import json
from urllib.request import Request, urlopen


REPO = "xmarre/Comfyui_Minimax_h3_latent_Upscaler"
RELEASE_SHA = "2c707492084962f7ed665e8817a05a11b14dab27"
TAG = "v0.1.1"
NODE_ID = "comfyui-minimax-h3-latent-upscaler"
VERSION = "0.1.1"
EXPECTED_ASSETS = {
    "Comfyui_Minimax_h3_latent_Upscaler-v0.1.1.zip",
    "SHA256SUMS",
}


def _json(url: str):
    request = Request(url, headers={"User-Agent": "release-publication-probe"})
    with urlopen(request, timeout=30) as response:
        return json.loads(response.read().decode("utf-8"))


def test_github_release_is_published_from_exact_release_commit():
    release = _json(f"https://api.github.com/repos/{REPO}/releases/tags/{TAG}")
    assert release["tag_name"] == TAG
    assert release["draft"] is False
    assert release["prerelease"] is False
    assert release["target_commitish"] == RELEASE_SHA
    assert {asset["name"] for asset in release["assets"]} == EXPECTED_ASSETS


def test_comfy_registry_contains_v011():
    versions = _json(f"https://api.comfy.org/nodes/{NODE_ID}/versions?include_status_reason=true")
    matching = [item for item in versions if item.get("version") == VERSION]
    assert matching, f"Comfy Registry does not contain {NODE_ID} {VERSION}; versions={versions!r}"
    assert matching[0].get("status") in {"NodeVersionStatusActive", "NodeVersionStatusPending"}, matching[0]
