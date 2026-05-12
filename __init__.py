import json
import os

import folder_paths
import comfy.sd


class SaveCheckpointWithMetadata:
    """
    Save a checkpoint with explicit control over safetensors metadata.

    Modes:
      - replace: write ONLY metadata_json. Ignore prompt_override and EXTRA_PNGINFO.
      - merge_minimal: base = prompt (override or hidden PROMPT) plus EXTRA_PNGINFO
        if include_extra_pnginfo=True, then overlay metadata_json.

    File naming modes:
      - smart_counter: if unsuffixed file does not exist, use it; otherwise continue
        from next free index based on existing files.
      - no_counter_overwrite: always use unsuffixed file name; overwrite if it exists.
    """

    SEARCH_ALIASES = ["save checkpoint metadata", "checkpoint metadata", "safetensors metadata"]

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("MODEL", {
                    "tooltip": "MODEL to serialize into a .safetensors checkpoint. Wire your merged/trained model here."
                }),
                "filename_prefix": ("STRING", {
                    "default": "checkpoints/CustomMeta",
                    "tooltip": "Subfolder/prefix under output directory. Example: checkpoints/MyModel."
                }),
                "filename_mode": (["smart_counter", "no_counter_overwrite"], {
                    "default": "smart_counter",
                    "tooltip": "smart_counter: first save uses prefix.safetensors if free, later saves use prefix_00001_.safetensors. no_counter_overwrite: always write to prefix.safetensors and overwrite if it exists."
                }),
                "metadata_json": ("STRING", {
                    "default": "{}",
                    "multiline": True,
                    "tooltip": "JSON object of header keys to write. Values must be strings; non-strings are JSON-encoded for you. Example: {\"author\":\"Alex\",\"training\":{\"epochs\":80}}"
                }),
                "metadata_mode": (["replace", "merge_minimal"], {
                    "default": "replace",
                    "tooltip": "replace: write only metadata_json. merge_minimal: start with prompt and optional EXTRA_PNGINFO, then apply metadata_json."
                }),
                "include_extra_pnginfo": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "Merge mode only. When ON, copy keys from hidden EXTRA_PNGINFO into the header before applying metadata_json."
                }),
                "prompt_override": ("STRING", {
                    "default": "",
                    "multiline": True,
                    "tooltip": "Optional override for the hidden PROMPT. Ignored in replace mode. In merge_minimal mode, becomes the base prompt unless metadata_json also contains prompt."
                }),
            },
            "optional": {
                "clip": ("CLIP", {
                    "tooltip": "Optional CLIP to embed into the saved checkpoint."
                }),
                "vae": ("VAE", {
                    "tooltip": "Optional VAE to embed into the saved checkpoint."
                }),
                "clip_vision": ("CLIP_VISION", {
                    "tooltip": "Optional CLIP_VISION to embed into the saved checkpoint."
                }),
            },
            "hidden": {
                "prompt": "PROMPT",
                "extra_pnginfo": "EXTRA_PNGINFO",
            },
        }

    RETURN_TYPES = ("STRING", "STRING", "STRING", "STRING")
    RETURN_NAMES = ("ckpt_path", "saved_metadata", "saved_prompt", "saved_extra_pnginfo")
    FUNCTION = "save"
    OUTPUT_NODE = True
    CATEGORY = "advanced/model_merging"

    def _coerce_metadata(self, dct):
        if dct is None:
            return {}

        if not isinstance(dct, dict):
            raise ValueError("metadata_json must be a JSON object.")

        out = {}
        for k, v in dct.items():
            key = str(k)
            out[key] = v if isinstance(v, str) else json.dumps(v, ensure_ascii=False)

        return out

    def _serialize_prompt(self, prompt_obj_or_str):
        if prompt_obj_or_str is None:
            return None

        if isinstance(prompt_obj_or_str, str):
            s = prompt_obj_or_str.strip()
            return s if s else None

        return json.dumps(prompt_obj_or_str, ensure_ascii=False)

    def _build_unsuffixed_path(self, full_dir, base):
        return os.path.join(full_dir, f"{base}.safetensors")

    def _build_suffixed_path(self, full_dir, base, n):
        return os.path.join(full_dir, f"{base}_{n:05}_.safetensors")

    def _next_suffix_from_dir(self, full_dir, base):
        """
        Inspect existing files and return the next numeric suffix.
        If no suffixed files exist, return 1.

        Looks for:
          base_00001_.safetensors
        """
        next_n = 1

        try:
            for name in os.listdir(full_dir):
                if not name.startswith(base + "_") or not name.endswith("_.safetensors"):
                    continue

                middle = name[len(base) + 1:-len("_.safetensors")]
                if middle.isdigit():
                    n = int(middle)
                    if n >= next_n:
                        next_n = n + 1

        except FileNotFoundError:
            next_n = 1

        return next_n

    def _choose_save_path(self, full_dir, base, filename_mode):
        """
        Filename policy:
          - smart_counter:
              If base.safetensors does not exist, use it.
              Else compute the next suffix from existing files and use that.
          - no_counter_overwrite:
              Always use base.safetensors.
        """
        unsuffixed = self._build_unsuffixed_path(full_dir, base)

        if filename_mode == "no_counter_overwrite":
            return unsuffixed

        if not os.path.exists(unsuffixed):
            return unsuffixed

        suffix_n = self._next_suffix_from_dir(full_dir, base)
        save_path = self._build_suffixed_path(full_dir, base, suffix_n)

        while os.path.exists(save_path):
            suffix_n += 1
            save_path = self._build_suffixed_path(full_dir, base, suffix_n)

        return save_path

    def _save_file(self, save_path, model, clip, vae, clip_vision, final_meta):
        """
        Base checkpoint save implementation.

        This is used by SaveCheckpointWithMetadata and can be overridden by
        subclasses that need different save behavior.
        """
        comfy.sd.save_checkpoint(
            save_path,
            model,
            clip=clip,
            vae=vae,
            clip_vision=clip_vision,
            metadata=final_meta,
            extra_keys={}
        )

    def save(
        self,
        model,
        filename_prefix,
        filename_mode,
        metadata_json,
        metadata_mode,
        include_extra_pnginfo,
        prompt_override,
        clip=None,
        vae=None,
        clip_vision=None,
        prompt=None,
        extra_pnginfo=None,
    ):
        try:
            user_meta_obj = json.loads(metadata_json) if metadata_json.strip() else {}
        except Exception as e:
            raise ValueError(f"Invalid metadata_json: {e}")

        user_meta = self._coerce_metadata(user_meta_obj)

        if prompt_override and prompt_override.strip():
            try:
                parsed_prompt_override = json.loads(prompt_override)
                effective_prompt = self._serialize_prompt(parsed_prompt_override)
            except Exception:
                effective_prompt = self._serialize_prompt(prompt_override)
        else:
            effective_prompt = self._serialize_prompt(prompt)

        base_metadata = {}
        saved_extra_subset = {}

        if metadata_mode == "merge_minimal":
            if effective_prompt is not None:
                base_metadata["prompt"] = effective_prompt

            if include_extra_pnginfo and isinstance(extra_pnginfo, dict):
                for k, v in extra_pnginfo.items():
                    key = str(k)

                    try:
                        val = v if isinstance(v, str) else json.dumps(v, ensure_ascii=False)
                    except Exception:
                        val = str(v)

                    base_metadata[key] = val
                    saved_extra_subset[key] = val

        final_meta = base_metadata.copy()
        final_meta.update(user_meta)

        out_dir = folder_paths.get_output_directory()
        full_dir, base, _counter_hint, _subfolder, _filename = folder_paths.get_save_image_path(
            filename_prefix,
            out_dir
        )
        os.makedirs(full_dir, exist_ok=True)

        save_path = self._choose_save_path(full_dir, base, filename_mode)

        self._save_file(
            save_path=save_path,
            model=model,
            clip=clip,
            vae=vae,
            clip_vision=clip_vision,
            final_meta=final_meta,
        )

        saved_metadata_str = json.dumps(final_meta, ensure_ascii=False, indent=2)
        saved_prompt_str = final_meta.get("prompt", "") or ""
        saved_extra_str = json.dumps(saved_extra_subset, ensure_ascii=False, indent=2)

        return (save_path, saved_metadata_str, saved_prompt_str, saved_extra_str)


class SaveDiffusionModelWithMetadata(SaveCheckpointWithMetadata):
    """
    Save only the MODEL/diffusion model with explicit safetensors metadata control.

    This shares metadata, filename replacement, and counter behavior with
    SaveCheckpointWithMetadata, but exposes no CLIP/VAE inputs.
    """

    SEARCH_ALIASES = ["export model", "save diffusion model", "model save metadata"]

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("MODEL", {
                    "tooltip": "MODEL to serialize into a .safetensors diffusion model file."
                }),
                "filename_prefix": ("STRING", {
                    "default": "diffusion_models/CustomMeta",
                    "tooltip": "Subfolder/prefix under output directory. Example: diffusion_models/MyModel."
                }),
                "filename_mode": (["smart_counter", "no_counter_overwrite"], {
                    "default": "smart_counter",
                    "tooltip": "smart_counter: first save uses prefix.safetensors if free, later saves use prefix_00001_.safetensors. no_counter_overwrite: overwrite prefix.safetensors."
                }),
                "metadata_json": ("STRING", {
                    "default": "{}",
                    "multiline": True,
                    "tooltip": "JSON object of safetensors header keys to write."
                }),
                "metadata_mode": (["replace", "merge_minimal"], {
                    "default": "replace",
                    "tooltip": "replace: write only metadata_json. merge_minimal: include prompt/extra_pnginfo, then apply metadata_json."
                }),
                "include_extra_pnginfo": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "Merge mode only. Copy hidden EXTRA_PNGINFO keys before applying metadata_json."
                }),
                "prompt_override": ("STRING", {
                    "default": "",
                    "multiline": True,
                    "tooltip": "Optional replacement for hidden PROMPT in merge_minimal mode."
                }),
            },
            "hidden": {
                "prompt": "PROMPT",
                "extra_pnginfo": "EXTRA_PNGINFO",
            },
        }

    RETURN_TYPES = ("STRING", "STRING", "STRING", "STRING")
    RETURN_NAMES = ("model_path", "saved_metadata", "saved_prompt", "saved_extra_pnginfo")
    FUNCTION = "save"
    OUTPUT_NODE = True
    CATEGORY = "advanced/model_merging"

    def _save_file(self, save_path, model, clip, vae, clip_vision, final_meta):
        """
        Diffusion-model-only save.

        This intentionally ignores CLIP/VAE/CLIP_VISION inputs because this node
        only exposes and saves the MODEL object.
        """
        comfy.sd.save_checkpoint(
            save_path,
            model,
            clip=None,
            vae=None,
            clip_vision=None,
            metadata=final_meta,
            extra_keys={}
        )

    def save(
        self,
        model,
        filename_prefix,
        filename_mode,
        metadata_json,
        metadata_mode,
        include_extra_pnginfo,
        prompt_override,
        prompt=None,
        extra_pnginfo=None,
    ):
        return super().save(
            model=model,
            filename_prefix=filename_prefix,
            filename_mode=filename_mode,
            metadata_json=metadata_json,
            metadata_mode=metadata_mode,
            include_extra_pnginfo=include_extra_pnginfo,
            prompt_override=prompt_override,
            clip=None,
            vae=None,
            clip_vision=None,
            prompt=prompt,
            extra_pnginfo=extra_pnginfo,
        )


NODE_CLASS_MAPPINGS = {
    "SaveCheckpointWithMetadata": SaveCheckpointWithMetadata,
    "SaveDiffusionModelWithMetadata": SaveDiffusionModelWithMetadata,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "SaveCheckpointWithMetadata": "Save Checkpoint with Metadata",
    "SaveDiffusionModelWithMetadata": "Save Diffusion Model with Metadata",
}
