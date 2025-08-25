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
      - smart_counter: if unsuffixed file does not exist, use it; otherwise continue from next free index,
        never counting backwards even if counter_hint is large.
      - no_counter_overwrite: always use unsuffixed file name; if it exists, overwrite it.
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
                    "tooltip": "Subfolder/prefix under output directory. Example: 'checkpoints/MyModel'."
                }),
                "filename_mode": (["smart_counter", "no_counter_overwrite"], {
                    "default": "smart_counter",
                    "tooltip": "smart_counter: first save uses 'prefix.safetensors' if free, later saves use 'prefix_00001_.safetensors', '..._00002_...', etc. no_counter_overwrite: always save to 'prefix.safetensors' and overwrite if it exists."
                }),
                "metadata_json": ("STRING", {
                    "default": "{}",
                    "multiline": True,
                    "tooltip": "JSON object of header keys to write. Values must be strings; non-strings are JSON-encoded for you. Example: {\"author\":\"Alex\",\"training\":{\"epochs\":80}}"
                }),
                # Dropdown: list-of-choices renders as a combo box
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
                    "tooltip": "Optional override for the hidden PROMPT. In replace mode this is ignored. In merge_minimal, becomes the base 'prompt' unless you also set 'prompt' in metadata_json."
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
                # Hidden inputs provided by ComfyUI runtime
                "prompt": "PROMPT",
                "extra_pnginfo": "EXTRA_PNGINFO",
            },
        }

    RETURN_TYPES = ()
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

    def _choose_ckpt_path(self, full_dir, base, counter_hint, filename_mode):
        """
        Filename policy:
        - smart_counter:
            If 'base.safetensors' does not exist, use it.
            Else, continue from next free index starting at max(counter_hint, 1) + 1, scanning upward.
        - no_counter_overwrite:
            Always use 'base.safetensors' (overwrite if it exists).
        """
        unsuffixed = self._build_unsuffixed_path(full_dir, base)

        if filename_mode == "no_counter_overwrite":
            return unsuffixed

        # smart_counter
        if not os.path.exists(unsuffixed):
            return unsuffixed

        n = max(counter_hint, 1) + 1
        path = self._build_suffixed_path(full_dir, base, n)
        while os.path.exists(path):
            n += 1
            path = self._build_suffixed_path(full_dir, base, n)
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

        # Base metadata by mode
        base_metadata = {}
        if metadata_mode == "merge_minimal":
            if effective_prompt is not None:
                base_metadata["prompt"] = effective_prompt
            if include_extra_pnginfo and isinstance(extra_pnginfo, dict):
                for k, v in extra_pnginfo.items():
                    try:
                        base_metadata[str(k)] = v if isinstance(v, str) else json.dumps(v, ensure_ascii=False)
                    except Exception:
                        base_metadata[str(k)] = str(v)

        # Final metadata
        final_meta = base_metadata.copy()
        final_meta.update(user_meta)

        # Resolve output path
        out_dir = folder_paths.get_output_directory()
        full_dir, base, counter_hint, subfolder, _ = folder_paths.get_save_image_path(filename_prefix, out_dir)

        # Ensure directory exists before we choose the name
        os.makedirs(full_dir, exist_ok=True)

        # Choose final path per selected filename_mode
        ckpt_path = self._choose_ckpt_path(full_dir, base, counter_hint, filename_mode)

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
        return {}

NODE_CLASS_MAPPINGS = {
    "SaveCheckpointWithMetadata": SaveCheckpointWithMetadata,
}
NODE_DISPLAY_NAME_MAPPINGS = {
    "Save Checkpoint with Metadata": "Save Checkpoint with Metadata",
}
