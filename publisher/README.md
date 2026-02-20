# Publisher

Automate publishing articles to Medium via their [URL import feature](https://help.medium.com/hc/en-us/articles/214550207-Importing-a-post-to-Medium).

## How it works

1. **`md_to_html.py`** converts the Markdown article to clean, semantic HTML with absolute GitHub image URLs
2. **`publish.sh`** runs the conversion, commits & pushes to GitHub, then prints the URL to paste into Medium's importer

## Usage

From the repo root:

```bash
./publisher/publish.sh articles/article_3_portfolio_optimization_predicted_returns.md
```

This will:
- Generate `articles/article_3_portfolio_optimization_predicted_returns.html`
- Commit and push all changes to GitHub
- Print the raw GitHub URL to import into Medium

### Options

```bash
# Dry run (convert only, don't push)
./publisher/publish.sh articles/article_3_portfolio_optimization_predicted_returns.md --no-push

# Use a different branch
./publisher/publish.sh articles/article_3_portfolio_optimization_predicted_returns.md --branch dev
```

### Manual import into Medium

1. Go to [https://medium.com/p/import](https://medium.com/p/import)
2. Paste the raw GitHub URL the script prints
3. Review the draft — check formatting, images, code blocks
4. Publish

## Notes

- Medium imports HTML, not Markdown — that's why we convert first
- Images must be served over HTTPS with proper content-type — GitHub raw URLs work
- Medium strips most CSS but preserves semantic tags (`h1-h6`, `pre`, `code`, `table`, `img`, etc.)
- Code blocks get syntax highlighting from Medium's own renderer
- After import, you may want to adjust: title image, tags, subtitle, and SEO description
