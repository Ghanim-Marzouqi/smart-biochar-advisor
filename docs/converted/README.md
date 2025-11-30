# Converted files

This folder contains converted files generated from `.docx` sources in the `docs/` folder.

What was produced for each `EDX,S*.docx`:

- `EDX,S*.docx` — original Word document.
- `EDX,S*.html` — HTML representation (images are embedded as data URIs where applicable).
- `EDX,S*.md` — Markdown generated from the HTML. This may include inline images encoded as base64 data URIs.
- `EDX,S*_local-images.md` — Markdown that references extracted image files from the original `.docx` (no base64 blobs). This is recommended for Claude or other assistants that accept markdown + separate image files.
- `EDX,S*_images/` — directory containing the images extracted from each `.docx` (PNG/JPEG).

Notes and tips:

- Use the `_local-images.md` variants if you want to keep the converted markdown readable and keep image data as files rather than embedded base64.
- If you want plain text only, I can also generate `text` versions that remove image references and keep only textual content.
- If you prefer a single `md` file with external links instead of relative paths, tell me and I can change the image paths to absolute URLs (images must be hosted).

How to convert new `.docx` files locally (node-based script):

1. Install dependencies (run once in repo root):

```bash
npm install
```

2. Convert all `.docx` files in `docs/`:

```bash
node scripts/convert-docx-to-md.js
```

This script:
- Produces both `*.md` (with possible inline data URIs) and `*_local-images.md` (with local image references).
- Saves images in `docs/converted/<filename>_images/` and HTML versions in `docs/converted/`.

If you want, I can: generate text-only files, compress/optimize extracted images, or produce a zip of each converted set for uploading.
