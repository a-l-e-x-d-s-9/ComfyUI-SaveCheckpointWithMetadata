import json
import os
import folder_paths
import comfy.sd

class SaveCheckpointWithMetadata:
    """
    Save a checkpoint with explicit control over safetensors metadata.

    Modes:
      - replace: write ONLY metadata_json. Ignore prompt_override and EXTRA_PNGINFO.
      - merge_minimal: base = prompt (override or hidden PROMPT) plus EXTRA_PNGINFO (if include_extra_pnginfo=True),
                       then overlay metadata_json (your keys win).

    File naming modes:
      - smart_counter: if unsuffixed file does not exist, use it; otherwise continue from next free index
        based on existing files (never backwards).
      - no_counter_overwrite: always use unsuffixed file name; overwrite if it exists.
    """

    @classmethod
    def INPUT_TYPES(s):
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
                    "tooltip": "smart_counter: first save uses prefix.safetensors if free, later saves use prefix_00001_.safetensors, ... no_counter_overwrite: always write to prefix.safetensors (overwrite if exists)."
                }),
                "metadata_json": ("STRING", {
                    "default": "{}",
                    "multiline": True,
                    "tooltip": "JSON object of header keys to write. Values must be strings; non-strings are JSON-encoded for you. Example: {\"author\":\"Alex\",\"training\":{\"epochs\":80}}"
                }),
                "metadata_mode": (["replace", "merge_minimal"], {
                    "default": "replace",
                    "tooltip": "replace: write only metadata_json. merge_minimal: start with prompt and optional EXTRA_PNGINFO, then apply metadata_json (your keys win)."
                }),
                "include_extra_pnginfo": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "Merge mode only. When ON, copy keys from hidden EXTRA_PNGINFO into the header before applying metadata_json."
                }),
                "prompt_override": ("STRING", {
                    "default": "",
                    "multiline": True,
                    "tooltip": "Optional override for the hidden PROMPT. Ignored in replace. In merge_minimal, becomes the base prompt unless you also set prompt in metadata_json."
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

    # Text outputs to inspect what was written
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
            ks = str(k)
            out[ks] = v if isinstance(v, str) else json.dumps(v, ensure_ascii=False)
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
        Looks for: base_00001_.safetensors style names only.
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
            # Directory does not exist yet; caller ensures it will be created
            next_n = 1
        return next_n

    def _choose_ckpt_path(self, full_dir, base, filename_mode):
        """
        Filename policy:
        - smart_counter:
            If base.safetensors does not exist, use it.
            Else compute the next suffix from existing files and use that.
        - no_counter_overwrite:
            Always use base.safetensors (overwrite if exists).
        """
        unsuffixed = self._build_unsuffixed_path(full_dir, base)

        if filename_mode == "no_counter_overwrite":
            return unsuffixed

        # smart_counter
        if not os.path.exists(unsuffixed):
            return unsuffixed

        start_n = self._next_suffix_from_dir(full_dir, base)
        path = self._build_suffixed_path(full_dir, base, start_n)
        while os.path.exists(path):
            start_n += 1
            path = self._build_suffixed_path(full_dir, base, start_n)
        return path

    def save(self,
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
             extra_pnginfo=None):

        # Parse user metadata
        try:
            user_meta_obj = json.loads(metadata_json) if metadata_json.strip() else {}
        except Exception as e:
            raise ValueError(f"Invalid metadata_json: {e}")
        user_meta = self._coerce_metadata(user_meta_obj)

        # Effective prompt
        if prompt_override and prompt_override.strip():
            try:
                parsed = json.loads(prompt_override)
                effective_prompt = self._serialize_prompt(parsed)
            except Exception:
                effective_prompt = self._serialize_prompt(prompt_override)
        else:
            effective_prompt = self._serialize_prompt(prompt)

        # Base metadata by mode, track included EXTRA_PNGINFO keys
        base_metadata = {}
        saved_extra_subset = {}
        if metadata_mode == "merge_minimal":
            if effective_prompt is not None:
                base_metadata["prompt"] = effective_prompt
            if include_extra_pnginfo and isinstance(extra_pnginfo, dict):
                for k, v in extra_pnginfo.items():
                    try:
                        val = v if isinstance(v, str) else json.dumps(v, ensure_ascii=False)
                    except Exception:
                        val = str(v)
                    key = str(k)
                    base_metadata[key] = val
                    saved_extra_subset[key] = val

        # Final metadata
        final_meta = base_metadata.copy()
        final_meta.update(user_meta)

        # Resolve output path and ensure directory exists
        out_dir = folder_paths.get_output_directory()
        full_dir, base, _counter_hint, _subfolder, _ = folder_paths.get_save_image_path(filename_prefix, out_dir)
        os.makedirs(full_dir, exist_ok=True)

        # Choose final path per selected filename_mode
        ckpt_path = self._choose_ckpt_path(full_dir, base, filename_mode)

        # Save checkpoint
        comfy.sd.save_checkpoint(
            ckpt_path,
            model,
            clip=clip,
            vae=vae,
            clip_vision=clip_vision,
            metadata=final_meta,
            extra_keys={}
        )

        # Prepare text outputs
        saved_metadata_str = json.dumps(final_meta, ensure_ascii=False, indent=2)
        saved_prompt_str = final_meta.get("prompt", "") or ""
        saved_extra_str = json.dumps(saved_extra_subset, ensure_ascii=False, indent=2)

        return (ckpt_path, saved_metadata_str, saved_prompt_str, saved_extra_str)

NODE_CLASS_MAPPINGS = {
    "SaveCheckpointWithMetadata": SaveCheckpointWithMetadata,
}
NODE_DISPLAY_NAME_MAPPINGS = {
    "Save Checkpoint with Metadata": "Save Checkpoint with Metadata",
}
