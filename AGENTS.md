# Local Run Receipts contributor notes

- Keep this package local-only. Do not add HTTP clients, telemetry, analytics, update checks, model downloads, or remote execution.
- The output node may write only inside `folder_paths.get_output_directory()` using validated relative paths.
- A completed receipt is immutable. A repeat must return `ALREADY_IDENTICAL` only when the receipt and all image bytes match.
- Keep dependencies empty unless ComfyUI itself requires the dependency for every supported install.
- Run the unit suite before committing. Do not use the managed DemoAPI ComfyUI profile to test custom nodes.
