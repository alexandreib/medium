#!/usr/bin/env python3
"""
Convert a Markdown article to Medium-importable HTML.

Medium's importer (https://medium.com/p/import) works best with:
- Clean semantic HTML (h1, h2, h3, p, pre>code, table, img)
- Absolute image URLs (GitHub raw URLs)
- No custom CSS classes — Medium strips them anyway

Usage:
    python md_to_html.py <input.md> <output.html> [--branch main] [--repo alexandreib/medium]
"""

import argparse
import re
from pathlib import Path

import markdown
from markdown.extensions.tables import TableExtension
from markdown.extensions.fenced_code import FencedCodeExtension
from markdown.extensions.codehilite import CodeHiliteExtension


GITHUB_RAW_BASE = "https://raw.githubusercontent.com/{repo}/{branch}/articles"

# Minimal inline CSS that Medium's importer respects for a clean look
TEMPLATE = """\
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="utf-8">
    <meta name="viewport" content="width=device-width, initial-scale=1">
    <title>{title}</title>
    <style>
        body {{ font-family: Georgia, serif; max-width: 740px; margin: 2rem auto; padding: 0 1rem; line-height: 1.7; color: #222; }}
        h1 {{ font-size: 2rem; }}
        h2 {{ font-size: 1.5rem; margin-top: 2rem; }}
        h3 {{ font-size: 1.25rem; }}
        pre {{ background: #f5f5f5; padding: 1rem; overflow-x: auto; border-radius: 4px; }}
        code {{ font-family: 'SFMono-Regular', Consolas, 'Liberation Mono', Menlo, monospace; font-size: 0.9em; }}
        table {{ border-collapse: collapse; width: 100%; margin: 1rem 0; }}
        th, td {{ border: 1px solid #ddd; padding: 0.5rem 0.75rem; text-align: left; }}
        th {{ background: #f5f5f5; }}
        img {{ max-width: 100%; height: auto; display: block; margin: 1.5rem auto; }}
        blockquote {{ border-left: 3px solid #ccc; margin: 1rem 0; padding-left: 1rem; color: #555; }}
        hr {{ border: none; border-top: 1px solid #ddd; margin: 2rem 0; }}
    </style>
</head>
<body>
{body}
</body>
</html>
"""


def rewrite_image_urls(md_text: str, raw_base: str) -> str:
    """Convert relative image paths to absolute GitHub raw URLs."""
    def replace_img(match):
        alt = match.group(1)
        path = match.group(2)
        # Skip already-absolute URLs
        if path.startswith("http://") or path.startswith("https://"):
            return match.group(0)
        # Strip leading ./ or ../
        clean = re.sub(r"^(\.\./|\./)+" , "", path)
        return f"![{alt}]({raw_base}/{clean})"

    return re.sub(r"!\[([^\]]*)\]\(([^)]+)\)", replace_img, md_text)


def extract_title(md_text: str) -> str:
    """Extract the first H1 as the page title."""
    m = re.search(r"^#\s+(.+)$", md_text, re.MULTILINE)
    return m.group(1).strip() if m else "Article"


def convert(input_path: str, output_path: str, repo: str, branch: str) -> None:
    raw_base = GITHUB_RAW_BASE.format(repo=repo, branch=branch)

    md_text = Path(input_path).read_text(encoding="utf-8")
    title = extract_title(md_text)
    md_text = rewrite_image_urls(md_text, raw_base)

    # Convert MD -> HTML
    extensions = [
        TableExtension(),
        FencedCodeExtension(),
        CodeHiliteExtension(css_class="highlight", noclasses=True, guess_lang=False),
    ]
    html_body = markdown.markdown(md_text, extensions=extensions)

    html_full = TEMPLATE.format(title=title, body=html_body)

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    Path(output_path).write_text(html_full, encoding="utf-8")
    print(f"✓ Converted: {input_path} -> {output_path}")
    print(f"  Title: {title}")
    print(f"  Image base: {raw_base}")


def main():
    parser = argparse.ArgumentParser(description="Convert Markdown article to Medium-importable HTML")
    parser.add_argument("input", help="Path to the Markdown file")
    parser.add_argument("output", help="Path for the output HTML file")
    parser.add_argument("--repo", default="alexandreib/medium", help="GitHub repo (owner/name)")
    parser.add_argument("--branch", default="main", help="Git branch for raw image URLs")
    args = parser.parse_args()
    convert(args.input, args.output, args.repo, args.branch)


if __name__ == "__main__":
    main()
