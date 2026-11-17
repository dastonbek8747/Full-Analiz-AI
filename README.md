# LLaVA Image & Video Analyzer

> A Streamlit app that analyses images and videos with the LLaVA-NeXT vision model running through Transformers.

![Python](https://img.shields.io/badge/Python-3.x-3776AB?logo=python&logoColor=white)
![License](https://img.shields.io/badge/License-MIT-green)


## Features

- Describe images in detail.
- Analyse video clips frame by frame.
- Graceful fallbacks when optional dependencies are missing.
- Runs locally with PyTorch.

## Tech Stack

| Layer | Technology |
|---|---|
| UI | Streamlit |
| Vision model | LLaVA-NeXT |
| Runtime | PyTorch, Transformers |
| Video | OpenCV |

## Quick Start

```bash
git clone https://github.com/dastonbek8747/llava-image-video-analyzer.git
cd llava-image-video-analyzer
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
streamlit run main.py
```

## Secrets

Copy `.streamlit/secrets.toml.example` to `.streamlit/secrets.toml` if you need Google API keys. The real secrets file is not committed.

## License

MIT © 2026 Doston Sultonov.
