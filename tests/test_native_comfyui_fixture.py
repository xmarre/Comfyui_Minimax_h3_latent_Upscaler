from __future__ import annotations

import ast
import os
from pathlib import Path

import pytest

pytestmark = pytest.mark.skipif(
    not os.environ.get("COMFYUI_PATH"),
    reason="native ComfyUI source fixture is not configured",
)


def _comfyui_root() -> Path:
    return Path(os.environ["COMFYUI_PATH"]).resolve()


def _source_tree(relative_path: str) -> ast.Module:
    path = _comfyui_root() / relative_path
    assert path.is_file(), f"reviewed ComfyUI source is missing {relative_path}"
    return ast.parse(path.read_text(encoding="utf-8"), filename=str(path))


def _class(tree: ast.AST, name: str) -> ast.ClassDef:
    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef) and node.name == name:
            return node
    raise AssertionError(f"reviewed ComfyUI source is missing class {name}")


def _function(tree: ast.AST, name: str) -> ast.FunctionDef | ast.AsyncFunctionDef:
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == name:
            return node
    raise AssertionError(f"reviewed ComfyUI source is missing function {name}")


def _argument_names(function: ast.FunctionDef | ast.AsyncFunctionDef) -> set[str]:
    args = function.args
    return {
        *(item.arg for item in args.posonlyargs),
        *(item.arg for item in args.args),
        *(item.arg for item in args.kwonlyargs),
    }


def _assigned_names(tree: ast.AST) -> set[str]:
    names: set[str] = set()
    for node in ast.walk(tree):
        targets = []
        if isinstance(node, ast.Assign):
            targets = node.targets
        elif isinstance(node, ast.AnnAssign):
            targets = [node.target]
        for target in targets:
            if isinstance(target, ast.Name):
                names.add(target.id)
    return names


def test_custom_node_imports_against_reviewed_comfyui_source():
    import folder_paths
    import nodes

    key = "MinimaxH3LatentUpscaler3DRefineHandoff"
    assert key in nodes.NODE_CLASS_MAPPINGS
    node_cls = nodes.NODE_CLASS_MAPPINGS[key]
    assert node_cls.INPUT_IS_LIST is True
    assert node_cls.OUTPUT_IS_LIST == (True,)
    assert node_cls.RETURN_TYPES == ("LATENT",)

    schema = node_cls.INPUT_TYPES()
    assert {"latent", "noise", "sampler", "sigmas"}.issubset(schema["required"])
    assert {"audio_latent", "refine_state", "model", "positive", "negative"}.issubset(
        schema["optional"]
    )
    assert "latent_upscale_models" in folder_paths.folder_names_and_paths


def test_refinement_runtime_matches_reviewed_comfyui_source_contract():
    samplers_tree = _source_tree("comfy/samplers.py")
    guider = _class(samplers_tree, "CFGGuider")
    methods = {
        node.name: node
        for node in guider.body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
    }
    assert {"inner_set_conds", "set_conds", "set_cfg", "sample"}.issubset(methods)
    assert {
        "noise",
        "latent_image",
        "sampler",
        "sigmas",
        "denoise_mask",
        "callback",
        "disable_pbar",
        "seed",
    }.issubset(_argument_names(methods["sample"]))

    assert _function(_source_tree("comfy/sample.py"), "fix_empty_latent_channels")
    assert _function(_source_tree("comfy/model_management.py"), "intermediate_device")
    assert _function(_source_tree("latent_preview.py"), "prepare_callback")
    assert _class(_source_tree("comfy/nested_tensor.py"), "NestedTensor")
    assert "PROGRESS_BAR_ENABLED" in _assigned_names(_source_tree("comfy/utils.py"))
