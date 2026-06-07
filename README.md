# Every frame an INR

A browser-only WebGPU experiment that trains a small Fourier-feature INR on a 256x256 image.

Run locally with any static server:

```powershell
python -m http.server 8000 --bind 127.0.0.1
```

Then open http://127.0.0.1:8000 in Chrome or Edge with WebGPU enabled.

