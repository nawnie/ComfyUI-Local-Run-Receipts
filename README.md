# Local Run Receipts

Local Run Receipts gives a ComfyUI image run a stable key and a small receipt saved beside its images. It is for retries, batch runs, and parameter sweeps where overwriting a result is worse than stopping with a clear conflict.

The package has no network code, no model downloads, and no dependencies beyond the Pillow and NumPy packages already used by ComfyUI.

## What it writes

`Build Run Key` turns declared values into a canonical JSON record and an `lrr1_...` SHA-256 key. Feed both outputs into `Commit Image Run` with the image batch you want to keep.

The commit node writes this shape under ComfyUI's normal output directory:

```text
output/
  local-run-receipts/
    lrr1_<sha256>/
      run-001.png
      run-002.png
      receipt.json
```

The key directory is reserved before any image is written. Image files are written first; `receipt.json` is the commit marker and is written last.

## Results

- `CREATED` means the images and receipt were written for the first time.
- `ALREADY_IDENTICAL` means the same key, prompt snapshot, and image bytes were already present. No file is changed.
- A conflict stops the workflow. The node never overwrites a receipt, a finished image, or an incomplete run directory.

`receipt.json` records the declared key inputs, each saved image's SHA-256 and dimensions, and a SHA-256 snapshot of the ComfyUI prompt. The prompt itself is not copied into the PNG or receipt.

## What the key does not prove

A run key is an identity for the values you declared. It does not prove that the same model weights, custom nodes, CUDA kernels, or machine settings were used later. Use it to detect a changed run or an accidental retry, not as a claim of bit-for-bit reproducibility.

## Install

Until the Registry listing is live, clone this repository into ComfyUI's `custom_nodes` folder and restart ComfyUI:

```text
ComfyUI/custom_nodes/ComfyUI-Local-Run-Receipts
```

After the Registry listing is published, install `local-run-receipts` from ComfyUI Manager or with the Comfy CLI.

## Use it in a workflow

1. Add `Local Run Receipts: Build Run Key` before the output node.
2. Set a namespace, a label, a seed, and `parameters_json` that describe the run you intend to save.
3. Connect `run_key` and `canonical_record` to `Local Run Receipts: Commit Image Run`.
4. Connect your final `IMAGE` output to the commit node.
5. Leave `output_subfolder` as `local-run-receipts` unless you want a separate safe subfolder.

The node accepts only safe relative output paths. It cannot write outside ComfyUI's output directory.

## Development checks

Run the built-in tests with a ComfyUI Python environment:

```powershell
& C:\DemoAPI-ComfyUI\.venv\Scripts\python.exe -m unittest discover -s tests -v
```

The tests use a temporary output directory. They do not start ComfyUI, load a model, or use a GPU.

## License

MIT. See [LICENSE](LICENSE).
