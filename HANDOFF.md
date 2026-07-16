# Local Run Receipts handoff

- Package root: this repository.
- Test command: `C:\DemoAPI-ComfyUI\.venv\Scripts\python.exe -m unittest discover -s tests -v`
- The package records declared run values, artifact hashes, and a prompt hash. It does not claim bit-for-bit reproducibility.
- Registry publication requires the owner-controlled Comfy Registry publisher identity and publishing API key. Never put that key in a file or GitHub workflow.
