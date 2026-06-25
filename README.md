# arXiv paper fetcher

A tiny local script for downloading arXiv PDFs, extracting text, and then zipping it.

## Use

```bash
pip install -r requirements.txt
```

Edit `arxiv_manifest.yaml`:

```yaml
papers:
  - key: name
    arxiv: "5555.55555"
```

Run:

```bash
python fetch_arxiv.py
```

PDFs go in `papers/pdf/`, extracted text goes in `papers/text/`. Existing files are reused unless you pass `--force`.
