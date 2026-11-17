"""Run with ComfyUI's Python and its root on PYTHONPATH:

    python -m unittest discover -s custom_nodes/ComfyUI-SaveCheckpointWithMetadata/tests -v
"""

import importlib.util
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import comfy.cli_args

comfy.cli_args.args.cpu = True

import torch
from safetensors import safe_open
from safetensors.torch import load_file

import comfy.model_base
import comfy.model_patcher
import comfy.ops
import comfy.sd
import comfy.supported_models


spec = importlib.util.spec_from_file_location("checkpoint_metadata", Path(__file__).resolve().parents[1] / "__init__.py")
nodes = importlib.util.module_from_spec(spec)
spec.loader.exec_module(nodes)


class TinyCheckpoint(torch.nn.Module):
    state_dict_for_saving = comfy.model_base.BaseModel.state_dict_for_saving

    def __init__(self):
        super().__init__()
        self.diffusion_model = torch.nn.Sequential(comfy.ops.manual_cast.Linear(2, 2))
        self.model_config = comfy.supported_models.SDXL({})
        self.model_type = comfy.model_base.ModelType.EPS
        self.diffusion_model[0].weight = torch.nn.Parameter(torch.ones(2, 2))
        self.diffusion_model[0].bias = torch.nn.Parameter(torch.zeros(2))


def tiny_clip(lazy):
    model = torch.nn.Module()
    for i, name in enumerate(("q_proj", "k_proj", "v_proj", "text_projection"), 1):
        path = "clip_g.transformer."
        if name != "text_projection":
            path += "text_model.encoder.layers.0.self_attn."
        parent = model
        for component in path.rstrip(".").split("."):
            if component not in parent._modules:
                parent.add_module(component, torch.nn.Module())
            parent = parent._modules[component]
        layer = comfy.ops.manual_cast.Linear(2, 2)
        layer.weight = torch.nn.Parameter(torch.full((2, 2), float(i)))
        if name == "text_projection":
            layer.weight = torch.nn.Parameter(torch.arange(4, 8, dtype=torch.float32).reshape(2, 2))
        layer.bias = torch.nn.Parameter(torch.full((2,), float(i)))
        layer.comfy_patched_weights = not lazy
        parent.add_module(name, layer)

    clip = comfy.sd.CLIP(no_init=True)
    clip.cond_stage_model = model
    clip.tokenizer = SimpleNamespace(state_dict=lambda: {})
    clip.patcher = comfy.model_patcher.ModelPatcher(model, torch.device("cpu"), torch.device("cpu"))
    if lazy:
        clip.patcher.add_patches({k: (torch.full_like(v, 0.25),) for k, v in model.state_dict().items()})
    return clip


class CheckpointSaveTests(unittest.TestCase):
    def test_sdxl_clip_export_preserves_patches_and_metadata(self):
        for lazy in (True, False):
            with self.subTest(lazy=lazy), tempfile.TemporaryDirectory() as tmp:
                clip = tiny_clip(lazy)
                original = {k: v.clone() for k, v in clip.get_sd().items()}
                exported = clip.state_dict_for_saving()
                if lazy:
                    self.assertEqual(next(iter(exported.values())).device.type, "comfy-lazy-caster")

                model = comfy.model_patcher.ModelPatcher(TinyCheckpoint(), torch.device("cpu"), torch.device("cpu"))
                model.add_patches({"diffusion_model.0.weight": (torch.full((2, 2), 0.5),)})
                node = nodes.SaveCheckpointWithMetadata()
                with patch("folder_paths.get_output_directory", return_value=tmp), patch("comfy.model_management.load_models_gpu"):
                    result = node.save(model, "checkpoint", "smart_counter", '{"author":"test","steps":20}', "replace", True, "", clip=clip)

                saved = load_file(result[0])
                prefix = "conditioner.embedders.1.model."
                delta = 0.25 if lazy else 0
                for suffix, shape in (("weight", (2, 2)), ("bias", (2,))):
                    expected = torch.cat([torch.full(shape, float(i) + delta) for i in (1, 2, 3)])
                    torch.testing.assert_close(saved[prefix + "transformer.resblocks.0.attn.in_proj_" + suffix], expected)
                projection = torch.arange(4, 8, dtype=torch.float32).reshape(2, 2) + delta
                torch.testing.assert_close(saved[prefix + "text_projection"], projection.T)
                torch.testing.assert_close(saved["model.diffusion_model.0.weight"], torch.full((2, 2), 1.5))
                with safe_open(result[0], framework="pt") as checkpoint:
                    self.assertEqual(checkpoint.metadata(), {"author": "test", "steps": "20"})
                self.assertEqual(json.loads(result[1]), {"author": "test", "steps": "20"})
                self.assertEqual(result[2:], ("", "{}"))
                for key, value in clip.get_sd().items():
                    torch.testing.assert_close(value, original[key])

    def test_model_only_and_merge_metadata(self):
        for node_type in (nodes.SaveCheckpointWithMetadata, nodes.SaveDiffusionModelWithMetadata):
            with self.subTest(node=node_type.__name__), tempfile.TemporaryDirectory() as tmp:
                model = comfy.model_patcher.ModelPatcher(TinyCheckpoint(), torch.device("cpu"), torch.device("cpu"))
                with patch("folder_paths.get_output_directory", return_value=tmp), patch("comfy.model_management.load_models_gpu"):
                    result = node_type().save(model, "model", "smart_counter", '{"author":"override"}', "merge_minimal", True, "", prompt={"test": 1}, extra_pnginfo={"author": "base", "workflow": {"version": 1}})
                with safe_open(result[0], framework="pt") as checkpoint:
                    self.assertEqual(checkpoint.metadata(), {"author": "override", "prompt": '{"test": 1}', "workflow": '{"version": 1}'})
                self.assertEqual(set(load_file(result[0])), {"model.diffusion_model.0.weight", "model.diffusion_model.0.bias"})


if __name__ == "__main__":
    unittest.main()
