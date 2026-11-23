"""Markdown -> PDF (A4) przez HTML i Chrome/Edge w trybie headless.

    .venv/Scripts/python.exe scripts/md2pdf.py docs/raport_koncowy.md

Obrazki: ścieżki względne liczone od katalogu pliku .md. Wymaga pakietu `markdown`.
"""

from __future__ import annotations

import re
import subprocess
import sys
import tempfile
from pathlib import Path

import markdown

BROWSERS = [
    r"C:\Program Files\Google\Chrome\Application\chrome.exe",
    r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe",
    r"C:\Program Files\Microsoft\Edge\Application\msedge.exe",
]

CSS = """
@page { size: A4; margin: 2cm 2.2cm; }
body { font-family: Cambria, Georgia, 'Times New Roman', serif; font-size: 10.5pt; line-height: 1.38; color: #111; }
h1 { font-size: 17pt; text-align: center; margin: 0 0 4pt; }
h1 + p { text-align: center; font-size: 10pt; margin: 0 0 2pt; }
h1 + p + p { text-align: center; font-size: 9.5pt; color: #333; margin: 0 0 14pt; padding-bottom: 8pt; border-bottom: 0.6pt solid #999; }
h2 { font-size: 13pt; margin: 16pt 0 5pt; break-after: avoid; }
h3 { font-size: 11pt; margin: 11pt 0 4pt; break-after: avoid; }
p { margin: 4pt 0; text-align: justify; }
ul, ol { margin: 3pt 0; padding-left: 18pt; } li { margin: 2pt 0; text-align: justify; }
table { border-collapse: collapse; margin: 6pt auto 8pt; font-size: 8.6pt; break-inside: avoid; }
th, td { padding: 2.2pt 5pt; border-bottom: 0.4pt solid #bbb; text-align: left; vertical-align: top; }
thead th { border-top: 1pt solid #111; border-bottom: 0.7pt solid #111; }
tbody tr:last-child td { border-bottom: 1pt solid #111; }
table:has(th:nth-child(5)) td:not(:first-child):not(:nth-child(2)),
table:has(th:nth-child(5)) th:not(:first-child):not(:nth-child(2)) { text-align: center; }
div.tbl { break-inside: avoid; }
code { font-family: Consolas, monospace; font-size: 8.8pt; }
img { display: block; max-width: 100%; margin: 6pt auto 2pt; break-inside: avoid; }
p:has(> img) { text-align: center; break-inside: avoid; }
p:has(> img) + p > em:only-child { display: block; text-align: center; font-size: 9pt; }
"""


def main(md_path: str, pdf_path: str | None = None) -> None:
    md = Path(md_path).resolve()
    pdf = Path(pdf_path).resolve() if pdf_path else md.with_suffix(".pdf")
    body = markdown.markdown(md.read_text(encoding="utf-8"),
                             extensions=["tables", "sane_lists"])
    # podpis tabeli ("Tabela N.") i tabela w jednym bloku, żeby nie rozdzielał ich koniec strony
    body = re.sub(r"(<p><strong>Tabela.*?</p>\s*<table>.*?</table>)",
                  r'<div class="tbl">\1</div>', body, flags=re.S)
    base = md.parent.as_uri() + "/"
    html = (f'<!doctype html><html lang="pl"><head><meta charset="utf-8">'
            f'<base href="{base}"><title>{md.stem}</title><style>{CSS}</style></head>'
            f'<body>{body}</body></html>')
    browser = next((b for b in BROWSERS if Path(b).exists()), None)
    if browser is None:
        sys.exit("Nie znaleziono Chrome ani Edge")
    with tempfile.TemporaryDirectory() as tmp:
        h = Path(tmp) / "doc.html"
        h.write_text(html, encoding="utf-8")
        subprocess.run([browser, "--headless=new", "--disable-gpu", "--no-pdf-header-footer",
                        "--allow-file-access-from-files", f"--print-to-pdf={pdf}", h.as_uri()],
                       check=True, capture_output=True)
    print(f"Zapisano {pdf}")


if __name__ == "__main__":
    main(*sys.argv[1:])
