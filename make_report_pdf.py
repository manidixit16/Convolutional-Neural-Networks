"""Render REPORT.md to a styled REPORT.pdf, with graphs embedded.

Converts the Markdown report to HTML (tables + images + typographic styling),
inlines every local image as a base64 data URI (so the sandboxed browser can
render them), and prints to PDF using the pre-installed Chromium via Playwright.
"""
import os
import re
import glob
import base64
import mimetypes
import markdown
from playwright.sync_api import sync_playwright

SRC = "REPORT.md"
OUT = "REPORT.pdf"

CSS = """
@page { size: A4; margin: 18mm 16mm; }
* { box-sizing: border-box; }
body { font-family: -apple-system, 'Segoe UI', Helvetica, Arial, sans-serif;
       font-size: 12px; line-height: 1.55; color: #1a1a1a; }
h1 { font-size: 22px; border-bottom: 3px solid #2b6cb0; padding-bottom: 6px; color: #1a365d; }
h2 { font-size: 16px; margin-top: 22px; color: #2b6cb0; border-bottom: 1px solid #e2e8f0; padding-bottom: 4px; }
h3 { font-size: 13px; margin-top: 16px; color: #2c5282; }
table { border-collapse: collapse; width: 100%; margin: 12px 0; font-size: 11px; }
th, td { border: 1px solid #cbd5e0; padding: 6px 9px; text-align: left; }
th { background: #ebf4ff; color: #1a365d; }
tr:nth-child(even) td { background: #f7fafc; }
code { background: #edf2f7; padding: 1px 4px; border-radius: 3px; font-size: 11px; }
strong { color: #1a202c; }
em { color: #4a5568; }
ul { margin: 6px 0; }
img { max-width: 100%; height: auto; display: block; margin: 12px auto;
      border: 1px solid #e2e8f0; border-radius: 4px; }
figure { margin: 14px 0; }
figcaption { font-size: 10.5px; color: #4a5568; font-style: italic; text-align: center; margin-top: 4px; }
"""


def inline_images(html):
    """Replace <img src="local/path"> with base64 data URIs."""
    def repl(m):
        src = m.group(1)
        if src.startswith("data:") or src.startswith("http"):
            return m.group(0)
        if not os.path.exists(src):
            return m.group(0)
        mime = mimetypes.guess_type(src)[0] or "image/png"
        data = base64.b64encode(open(src, "rb").read()).decode()
        return m.group(0).replace(src, f"data:{mime};base64,{data}")
    return re.sub(r'<img[^>]*\ssrc="([^"]+)"', repl, html)


def find_chrome():
    for pat in ("/opt/pw-browsers/chromium-*/chrome-linux/chrome",
                "/opt/pw-browsers/chromium-*/chrome-linux/headless_shell"):
        hits = sorted(glob.glob(pat))
        if hits:
            return hits[-1]
    return None


def main():
    with open(SRC) as f:
        md_text = f.read()
    html_body = markdown.markdown(md_text, extensions=["tables", "fenced_code", "sane_lists", "attr_list"])
    html_body = inline_images(html_body)
    html = f"<!doctype html><html><head><meta charset='utf-8'><style>{CSS}</style></head><body>{html_body}</body></html>"

    chrome = find_chrome()
    with sync_playwright() as p:
        launch_kwargs = {"args": ["--no-sandbox"]}
        if chrome:
            launch_kwargs["executable_path"] = chrome
        browser = p.chromium.launch(**launch_kwargs)
        page = browser.new_page()
        page.set_content(html, wait_until="networkidle")
        page.pdf(path=OUT, format="A4", print_background=True)
        browser.close()
    print(f"Wrote {OUT} ({os.path.getsize(OUT)} bytes)")


if __name__ == "__main__":
    main()
