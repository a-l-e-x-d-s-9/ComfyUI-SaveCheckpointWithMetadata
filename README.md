# ComfyUI-SaveCheckpointWithMetadata

Save `.safetensors` files with custom metadata and explicit filename behavior in ComfyUI.

This extension provides two nodes:

- `Save Checkpoint with Metadata` - saves a checkpoint-style `.safetensors` file from `MODEL`, with optional `CLIP`, `VAE`, and `CLIP_VISION` inputs.
- `Save Diffusion Model with Metadata` - saves only a `MODEL` / diffusion-model-style `.safetensors` file, without `CLIP`, `VAE`, or `CLIP_VISION` inputs.

Both nodes share the same metadata control and filename behavior:

- Write exactly the header you want via `metadata_json`.
- Optional `prompt_override` to replace the hidden `PROMPT`.
- Merge mode to include hidden `EXTRA_PNGINFO`.
- Filename modes:
  - `smart_counter`: first save uses unsuffixed `prefix.safetensors` if free, next saves continue from the next free counter without going backwards.
  - `no_counter_overwrite`: always write to `prefix.safetensors` and overwrite if it exists.

## Install

### ComfyUI Manager

1) Manager -> Install via Git URL -> paste:

```text
https://github.com/a-l-e-x-d-s-9/ComfyUI-SaveCheckpointWithMetadata.git
```

2) Restart ComfyUI.

If Manager shows a security level error, either install manually or add the repo to the Manager catalog. See Troubleshooting below.

### Manual

```bash
cd /path/to/ComfyUI/custom_nodes
git clone https://github.com/a-l-e-x-d-s-9/ComfyUI-SaveCheckpointWithMetadata.git

# optional deps
python3 -m pip install -r ComfyUI-SaveCheckpointWithMetadata/requirements.txt

# optional post-install hook
python3 ComfyUI-SaveCheckpointWithMetadata/install.py || true
```

Restart ComfyUI.

## Node location

- Category: `advanced/model_merging`
- Display names:
  - `Save Checkpoint with Metadata`
  - `Save Diffusion Model with Metadata`

## Which node should I use?

### Use `Save Checkpoint with Metadata` when:

- You want to save a checkpoint-style file.
- You want optional `CLIP`, `VAE`, or `CLIP_VISION` saved into the file.
- You want the default prefix to point to `checkpoints/CustomMeta`.

### Use `Save Diffusion Model with Metadata` when:

- You want a model-only file similar in purpose to ComfyUI's built-in `ModelSave` node.
- You only want to provide a single `MODEL` input.
- You do not want `CLIP`, `VAE`, or `CLIP_VISION` inputs on the node.
- You want the default prefix to point to `diffusion_models/CustomMeta`.

## Quick start - checkpoint save

1) Add `Save Checkpoint with Metadata`.
2) Wire your `MODEL` into the node.
3) Optionally wire `CLIP`, `VAE`, and/or `CLIP_VISION`.
4) Choose `filename_prefix`, for example `checkpoints/MyModel`.
5) Pick `filename_mode`:
   - `smart_counter` for automatic unique names.
   - `no_counter_overwrite` to always write `prefix.safetensors`.
6) Pick `metadata_mode`:
   - `replace` to write only your `metadata_json`.
   - `merge_minimal` to start with prompt plus optional `EXTRA_PNGINFO`, then apply `metadata_json`.
7) Fill `metadata_json` with a JSON object of header keys.
8) Optionally set `prompt_override` to replace the hidden `PROMPT` in merge mode.
9) Queue. Your checkpoint is saved in ComfyUI's output directory under the resolved subfolder.

## Quick start - diffusion model save

1) Add `Save Diffusion Model with Metadata`.
2) Wire your `MODEL` into the node.
3) Choose `filename_prefix`, for example `diffusion_models/MyModel`.
4) Pick `filename_mode`:
   - `smart_counter` for automatic unique names.
   - `no_counter_overwrite` to always write `prefix.safetensors`.
5) Pick `metadata_mode`:
   - `replace` to write only your `metadata_json`.
   - `merge_minimal` to start with prompt plus optional `EXTRA_PNGINFO`, then apply `metadata_json`.
6) Fill `metadata_json` with a JSON object of header keys.
7) Optionally set `prompt_override` to replace the hidden `PROMPT` in merge mode.
8) Queue. Your model file is saved in ComfyUI's output directory under the resolved subfolder.

## Inputs

### Shared inputs

Both nodes have these inputs:

| Input | Type | Description |
|------|------|-------------|
| model | MODEL | The model to serialize into a `.safetensors` file. |
| filename_prefix | STRING | Subfolder and base name under ComfyUI output directory, for example `checkpoints/MyModel` or `diffusion_models/MyModel`. |
| filename_mode | DROPDOWN | `smart_counter` or `no_counter_overwrite`. See Filename behavior. |
| metadata_json | STRING (multiline) | JSON object of header keys to write. Values must be strings; non-strings are JSON-encoded for you. Example: `{"author":"Alex","training":{"epochs":80}}`. |
| metadata_mode | DROPDOWN | `replace`: write only `metadata_json`. `merge_minimal`: base header includes prompt and, optionally, `EXTRA_PNGINFO`, then your `metadata_json` overwrites or adds keys. |
| include_extra_pnginfo | BOOLEAN | Used only in `merge_minimal`. When on, copy keys from hidden `EXTRA_PNGINFO` into the base header. |
| prompt_override | STRING (multiline) | Optional override for the hidden `PROMPT`. Ignored in `replace`. In `merge_minimal`, becomes the base `prompt` unless you also set `prompt` in `metadata_json`. |

### Extra inputs on `Save Checkpoint with Metadata`

Only `Save Checkpoint with Metadata` has these optional inputs:

| Input | Type | Description |
|------|------|-------------|
| clip | CLIP (optional) | Embed CLIP in the checkpoint. |
| vae | VAE (optional) | Embed VAE in the checkpoint. |
| clip_vision | CLIP_VISION (optional) | Embed CLIP_VISION in the checkpoint. |

`Save Diffusion Model with Metadata` intentionally does not expose these inputs.

Tip: To see tooltips, enable ComfyUI Settings -> Nodes -> Enable Tooltips.

## Outputs

Both nodes output text values so you can inspect what was written:

| Output | Description |
|--------|-------------|
| saved path | Full path to the saved `.safetensors` file. Named `ckpt_path` on the checkpoint node and `model_path` on the diffusion model node. |
| saved_metadata | Final metadata dictionary written to the safetensors header, formatted as JSON. |
| saved_prompt | The final `prompt` metadata value, if present. |
| saved_extra_pnginfo | The subset of `EXTRA_PNGINFO` copied into metadata when using `merge_minimal`. |

## Filename behavior

### `smart_counter`

- If `prefix.safetensors` does not exist, write there.
- Otherwise continue with `prefix_00001_.safetensors`, `prefix_00002_.safetensors`, etc.
- The node scans existing files and uses the next free suffix.

### `no_counter_overwrite`

- Always write to `prefix.safetensors`.
- If the file exists, it is overwritten.

## Metadata behavior

### `replace`

The final safetensors header contains only the keys from `metadata_json`.

`prompt_override`, hidden `PROMPT`, and hidden `EXTRA_PNGINFO` are ignored.

### `merge_minimal`

The node builds a small base metadata dictionary first:

1) Adds `prompt` from `prompt_override`, if provided.
2) Otherwise adds `prompt` from hidden ComfyUI `PROMPT`, if available.
3) Adds hidden `EXTRA_PNGINFO` keys if `include_extra_pnginfo` is enabled.
4) Applies `metadata_json` last.

Keys in `metadata_json` win over keys from the base metadata.

## Examples

### Replace-only header

- metadata_mode: `replace`
- metadata_json:

```json
{"author":"author","project":"Flux-Kontext","modelspec.architecture":"sdxl"}
```

Result: header contains exactly those keys.

### Merge prompt and extra info, then override

- metadata_mode: `merge_minimal`
- prompt_override: paste your workflow JSON or plain text
- include_extra_pnginfo: on
- metadata_json:

```json
{"prompt":"OVERRIDE","author":"Alex","training":{"epochs":80}}
```

Result: the final `prompt` value is `OVERRIDE`, because `metadata_json` is applied last.

### Save a model-only file

Use `Save Diffusion Model with Metadata` with:

- filename_prefix: `diffusion_models/MyModel`
- metadata_mode: `replace`
- metadata_json:

```json
{"author":"Alex","model_type":"diffusion_model","modelspec.architecture":"flux"}
```

## Notes

- Safetensors metadata requires `dict[str, str]`. Non-string values are JSON-encoded by the node.
- These nodes do not follow the stock `--disable-metadata` flag. They write the metadata you specify.
- Hidden inputs `PROMPT` and `EXTRA_PNGINFO` are provided by ComfyUI runtime. The nodes only read them when `merge_minimal` is selected.
- `Save Diffusion Model with Metadata` saves only the `MODEL` object and ignores CLIP/VAE/CLIP_VISION by design.
- For maximum compatibility, test the output of `Save Diffusion Model with Metadata` by loading it through the same ComfyUI loader you normally use for diffusion model files.

## Troubleshooting

### Manager shows: This action is not allowed with this security level configuration.

- Option A: Manual install by `git clone` into `ComfyUI/custom_nodes/`.
- Option B: Lower Manager `security_level` to `middle` or `weak` in its `config.ini`, then restart ComfyUI.
- Option C: Submit your repo to the ComfyUI-Manager catalog (`custom-node-list.json`) so it can be installed at normal security. Until the PR is merged, users can use Install via Git URL or manual install.

### I see `prefix_00001_.safetensors` even on the first save

In `smart_counter`, the node prefers `prefix.safetensors` if it does not exist. If you still get a numbered file, check that a file named `prefix.safetensors` is not already present in the resolved output folder.

### I want modelspec.* keys like the stock node adds

Add them to `metadata_json` explicitly. Example:

```json
{"modelspec.architecture":"sdxl","modelspec.vision":"clip"}
```

### The diffusion model file does not load where I expected

`Save Diffusion Model with Metadata` is intended for model-only saves, but exact compatibility can depend on the model type and the loader you use later. Try loading the output with the same ComfyUI model-only loader you normally use. If you need full checkpoint behavior with CLIP/VAE included, use `Save Checkpoint with Metadata` instead.

## License

No.

## Credits

Author: alexds9

Inspired by ComfyUI stock saving logic, with explicit metadata and naming control.
