# ComfyUI-SaveCheckpointWithMetadata

Save .safetensors checkpoints with custom metadata and explicit filename behavior in ComfyUI.

- Write exactly the header you want via `metadata_json`.
- Optional `prompt_override` to replace the hidden PROMPT.
- Merge mode to include hidden `EXTRA_PNGINFO`.
- Filename modes:
  - `smart_counter`: first save uses unsuffixed `prefix.safetensors` if free, next saves continue from the next free counter without going backwards.
  - `no_counter_overwrite`: always write to `prefix.safetensors` and overwrite if it exists.

## Install

### ComfyUI Manager
- Manager -> Install via Git URL -> paste:
https://github.com/a-l-e-x-d-s-9/ComfyUI-SaveCheckpointWithMetadata.git

- Restart ComfyUI.

### Manual
```bash
cd /path/to/ComfyUI/custom_nodes
git clone https://github.com/<your-account>/ComfyUI-SaveCheckpointWithMetadata.git
# optional deps
python3 -m pip install -r ComfyUI-SaveCheckpointWithMetadata/requirements.txt
# optional post-install hook
python3 ComfyUI-SaveCheckpointWithMetadata/install.py || true
