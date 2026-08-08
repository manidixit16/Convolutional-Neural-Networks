"""Render REPORT.md to a styled REPORT.pdf.

Converts the Markdown report to HTML (with tables + basic typographic styling)
and prints it to PDF using the pre-installed Chromium via Playwright.
"""
import os
import glob
import markdown
from playwright.sync_api import sync_playwright

SRC = "REPORT.md"
OUT = "REPORT.pdf"

CSS = """
@page { size: A4; margin: 20mm 18mm; }
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
"""


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
    html_body = markdown.markdown(md_text, extensions=["tables", "fenced_code", "sane_lists"])
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
