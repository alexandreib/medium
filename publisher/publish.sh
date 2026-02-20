#!/usr/bin/env bash
#
# publish.sh — Convert article to HTML, push to GitHub, print Medium import URL.
#
# Usage:
#   ./publisher/publish.sh articles/article_3_portfolio_optimization_predicted_returns.md
#
# What it does:
#   1. Converts the Markdown article to clean HTML (with absolute GitHub image URLs)
#   2. Commits & pushes everything (article, images, HTML, notebook) to GitHub
#   3. Prints the URL to paste into Medium's importer (https://medium.com/p/import)
#
set -euo pipefail

REPO="alexandreib/medium"
BRANCH="main"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"

# --- Argument parsing ---
if [[ $# -lt 1 ]]; then
    echo "Usage: $0 <path/to/article.md> [--branch main] [--no-push]"
    echo ""
    echo "Example:"
    echo "  $0 articles/article_3_portfolio_optimization_predicted_returns.md"
    exit 1
fi

INPUT_MD="$1"
shift

NO_PUSH=false
while [[ $# -gt 0 ]]; do
    case "$1" in
        --branch) BRANCH="$2"; shift 2 ;;
        --no-push) NO_PUSH=true; shift ;;
        *) echo "Unknown option: $1"; exit 1 ;;
    esac
done

# Resolve paths
INPUT_PATH="$ROOT_DIR/$INPUT_MD"
BASENAME="$(basename "$INPUT_MD" .md)"
OUTPUT_HTML="$ROOT_DIR/articles/${BASENAME}.html"

if [[ ! -f "$INPUT_PATH" ]]; then
    echo "Error: File not found: $INPUT_PATH"
    exit 1
fi

# --- Step 1: Convert MD -> HTML ---
echo "=== Step 1: Converting Markdown to HTML ==="
cd "$ROOT_DIR"
PYTHON="${ROOT_DIR}/.venv/bin/python"
if [[ ! -x "$PYTHON" ]]; then
    PYTHON="$(command -v python3 || command -v python)"
fi
"$PYTHON" publisher/md_to_html.py "$INPUT_PATH" "$OUTPUT_HTML" \
    --repo "$REPO" --branch "$BRANCH"

# --- Step 2: Git commit & push ---
echo ""
echo "=== Step 2: Git commit & push ==="
cd "$ROOT_DIR"
git add -A
if git diff --cached --quiet; then
    echo "  No changes to commit."
else
    COMMIT_MSG="Update article: $BASENAME ($(date +%Y-%m-%d))"
    git commit -m "$COMMIT_MSG"
    echo "  Committed: $COMMIT_MSG"
fi

if [[ "$NO_PUSH" == false ]]; then
    git push origin "$BRANCH"
    echo "  Pushed to origin/$BRANCH"
else
    echo "  Skipping push (--no-push flag)"
fi

# --- Step 3: Print import URL ---
echo ""
echo "=== Step 3: Import into Medium ==="
# The HTML file on GitHub raw
RAW_URL="https://raw.githubusercontent.com/${REPO}/${BRANCH}/articles/${BASENAME}.html"
IMPORT_URL="https://medium.com/p/import"

echo ""
echo "  Your article HTML is hosted at:"
echo "    $RAW_URL"
echo ""
echo "  To import into Medium:"
echo "    1. Open: $IMPORT_URL"
echo "    2. Paste this URL: $RAW_URL"
echo "    3. Review the draft and publish"
echo ""
echo "Done."
