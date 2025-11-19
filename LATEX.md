# Dependencies for `_merge.sh`

This document describes all dependencies required for the `_merge.sh` script to function properly.

## Required Dependencies

### 1. `pandoc`

**Purpose:** Universal document converter used to generate PDFs from Markdown with proper LaTeX math rendering.

**Installation (macOS):**

```bash
brew install pandoc
```

**What happens if missing:** Script will exit with an error message.

---

## Optional Dependencies (for PDF Generation)

The script requires **at least one** of the following to generate PDFs. They are listed in order of recommendation:

### Option 1: LaTeX Engine (Recommended - Best Math Rendering)

**Purpose:** Provides native LaTeX rendering for mathematical equations, giving the highest quality output for documents with LaTeX math (`$...$` and `$$...$$`).

**Available Engines:**

- `xelatex` (preferred - best Unicode/font support)
- `lualatex` (preferred - modern LuaTeX engine with Unicode support)
- `pdflatex` (fallback - most common, but limited Unicode support)

**Note:** The script automatically prefers XeLaTeX/LuaLaTeX over pdfLaTeX for better Unicode character handling. If only pdfLaTeX is available, the script will automatically replace Unicode box-drawing characters (│, ├, └, etc.) with ASCII equivalents to prevent errors.

**Installation (macOS):**

```bash
# Install BasicTeX (minimal LaTeX distribution)
brew install --cask basictex

# Add TeX Live to PATH (required for tlmgr and LaTeX commands)
# Add this line to your ~/.zshrc or ~/.bash_profile:
export PATH="/usr/local/texlive/2025basic/bin/universal-darwin:$PATH"
# Or use the symlink (if available):
export PATH="/Library/TeX/texbin:$PATH"

# Reload your shell configuration
source ~/.zshrc  # or source ~/.bash_profile

# Update TeX Live package manager
sudo tlmgr update --self

# Install recommended font packages
sudo tlmgr install collection-fontsrecommended

# Install XeLaTeX and LuaLaTeX for better Unicode support (recommended)
sudo tlmgr install xetex lualatex
```

**Note:** If `tlmgr` command is not found after installation, you can use the full path:

```bash
# Find your TeX Live installation year
ls /usr/local/texlive/

# Use the full path (replace YYYY with your year, e.g., 2025basic)
sudo /usr/local/texlive/YYYYbasic/bin/universal-darwin/tlmgr update --self
sudo /usr/local/texlive/YYYYbasic/bin/universal-darwin/tlmgr install collection-fontsrecommended
sudo /usr/local/texlive/YYYYbasic/bin/universal-darwin/tlmgr install xetex lualatex
```

**Note:** BasicTeX is a minimal installation (~100MB). For a full LaTeX installation, use `brew install --cask mactex` (~4GB).

**What happens if missing:** Script will fall back to HTML+MathJax method (see below).

---

### Option 2: `wkhtmltopdf`

**Purpose:** Converts HTML (with MathJax-rendered math) to PDF. Used as a fallback when LaTeX is not available.

**Installation (macOS):**

```bash
brew install wkhtmltopdf
```

**What happens if missing:** Script will try the next option (weasyprint).

---

### Option 3: `weasyprint`

**Purpose:** Another HTML-to-PDF converter, alternative to wkhtmltopdf.

**Installation (macOS):**

```bash
pip install weasyprint
```

**What happens if missing:** Script will exit with an error and instructions.

---

## Dependency Priority

The script uses dependencies in this order:

1. **LaTeX Engine** (if available) → Direct PDF generation with native LaTeX math
   - Engine priority: `xelatex` > `lualatex` > `pdflatex`
   - XeLaTeX/LuaLaTeX are preferred for better Unicode support
   - If only pdfLaTeX is available, Unicode box-drawing characters are automatically replaced with ASCII equivalents
2. **HTML + MathJax** → If LaTeX unavailable, generates HTML with MathJax
   - Then converts HTML to PDF using:
     - `wkhtmltopdf` (preferred)
     - `weasyprint` (fallback)

## Unicode Character Handling

The script automatically preprocesses the markdown to handle Unicode characters:

- **Box-drawing characters** (│, ├, └, ┌, ┐, ┘, ─) are replaced with ASCII equivalents (|, `, +, -)
- This prevents LaTeX errors when using pdfLaTeX (which has limited Unicode support)
- If you have XeLaTeX or LuaLaTeX installed, these engines can handle Unicode natively, but the preprocessing still occurs for compatibility

## Quick Setup (Recommended)

For the best experience with mathematical equations, install:

```bash
# Required
brew install pandoc

# Recommended for math rendering
brew install --cask basictex

# Add TeX Live to PATH (add to ~/.zshrc or ~/.bash_profile)
echo 'export PATH="/usr/local/texlive/2025basic/bin/universal-darwin:$PATH"' >> ~/.zshrc
source ~/.zshrc

# Update TeX Live package manager
sudo tlmgr update --self

# Install recommended font packages
sudo tlmgr install collection-fontsrecommended

# Install XeLaTeX and LuaLaTeX for better Unicode support (recommended)
sudo tlmgr install xetex lualatex
```

## Minimal Setup (No LaTeX)

If you don't want to install LaTeX, you can use:

```bash
# Required
brew install pandoc

# HTML-to-PDF converter
brew install wkhtmltopdf
```

**Note:** Math rendering quality will be slightly lower than with LaTeX, but still functional.

## Verification

Check if dependencies are installed:

```bash
# Check pandoc
command -v pandoc && echo "✓ pandoc installed"

# Check LaTeX engines (in priority order)
command -v xelatex && echo "✓ xelatex installed (preferred)" || \
command -v lualatex && echo "✓ lualatex installed (preferred)" || \
command -v pdflatex && echo "✓ pdflatex installed (fallback)" || \
echo "✗ No LaTeX engine found"

# Check HTML-to-PDF converters
command -v wkhtmltopdf && echo "✓ wkhtmltopdf installed"
command -v weasyprint && echo "✓ weasyprint installed"
```

## Troubleshooting

### "pandoc not found"

- Install: `brew install pandoc`

### "No PDF converter found"

- Install one of the options above (LaTeX recommended)

### LaTeX math not rendering correctly

- Ensure you have a LaTeX engine installed (BasicTeX or MacTeX)
- Run the TeX Live manager update commands
- Check that the markdown uses proper LaTeX syntax: `$inline$` or `$$display$$`

### PDF generation fails with LaTeX

- Try installing additional LaTeX packages: `sudo tlmgr install <package-name>`
- Check LaTeX logs for specific missing packages
- Consider using the HTML+MathJax fallback method

### Unicode character errors (e.g., "Unicode character │ not set up for use with LaTeX")

- **Automatic fix:** The script automatically replaces Unicode box-drawing characters with ASCII equivalents
- **Better solution:** Install XeLaTeX or LuaLaTeX for native Unicode support:

  ```bash
  sudo /usr/local/texlive/2025basic/bin/universal-darwin/tlmgr install xetex lualatex
  ```

- The script will automatically prefer XeLaTeX/LuaLaTeX if available
- If you still see Unicode errors, check that XeLaTeX/LuaLaTeX are in your PATH
