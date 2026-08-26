#!/usr/bin/env python3
"""Generate a single-page site (docs/) showing the graphical abstract and every
figure in paper.tex with its caption, for GitHub Pages.

Figures are rendered faithfully:
  - tikz/Inkscape figures (\\input{...}) are compiled in build/ with the paper's
    preamble via the preview package, then rasterized;
  - plot figures (\\includegraphics[page=N]) are rasterized from the matching
    page of build/figures/paper/<name>.pdf.

Captions are extracted verbatim from paper.tex and lightly converted to HTML
(math is left for MathJax).
"""
import os
import re
import shutil
import subprocess

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PAPER = os.path.join(ROOT, "paper.tex")
SUPP = os.path.join(ROOT, "supplementary.tex")
BUILD = os.path.join(ROOT, "build")
FIGDIR = os.path.join(BUILD, "figures", "paper")
DOCS = os.path.join(ROOT, "docs")
OUTFIG = os.path.join(DOCS, "figures")

TITLE = ("A tale of perfect fit and phantom optima: how data-driven models can fail in real-time optimization")
AUTHORS = ('Prithvi Dake<sup>1,*</sup>, '
           'Rahul Bindlish<sup>2</sup>, '
           'James B. Rawlings<sup>1</sup>')
AFFILIATIONS = ('<sup>1</sup>Department of Chemical Engineering, '
                'University of California, Santa Barbara, CA 93106, USA'
                ' &nbsp;&middot;&nbsp; '
                '<sup>2</sup>Dow Chemical Company, TX, USA')
CORRESPONDING = ('<sup>*</sup>Corresponding author: '
                 '<a href="mailto:prithvidake@ucsb.edu">prithvidake@ucsb.edu</a>')

# Resource links shown as buttons in the header. Leave a URL empty to hide it.
GITHUB_URL = "https://github.com/dakeprithvi/2026c_structure_id_plant"
ZENODO_URL = "https://doi.org/10.5281/zenodo.21464341"
ARXIV_URL = "https://arxiv.org/abs/2608.23885v1"  # set once the preprint is posted

# Base URL for linking a plot to the script that generated it.
REPO_BASE = GITHUB_URL + "/blob/main"

GITHUB_SVG = ('<svg height="20" width="20" viewBox="0 0 16 16" '
              'fill="currentColor"><path d="M8 0C3.58 0 0 3.58 0 8c0 3.54 '
              '2.29 6.53 5.47 7.59.4.07.55-.17.55-.38 0-.19-.01-.82-.01-1.49'
              '-2.01.37-2.53-.49-2.69-.94-.09-.23-.48-.94-.82-1.13-.28-.15'
              '-.68-.52-.01-.53.63-.01 1.08.58 1.23.82.72 1.21 1.87.87 2.33'
              '.66.07-.52.28-.87.51-1.07-1.78-.2-3.64-.89-3.64-3.95 0-.87.31'
              '-1.59.82-2.15-.08-.2-.36-1.02.08-2.12 0 0 .67-.21 2.2.82.64'
              '-.18 1.32-.27 2-.27.68 0 1.36.09 2 .27 1.53-1.04 2.2-.82 2.2'
              '-.82.44 1.1.16 1.92.08 2.12.51.56.82 1.27.82 2.15 0 3.07-1.87 '
              '3.75-3.65 3.95.29.25.54.73.54 1.48 0 1.07-.01 1.93-.01 2.2 0 '
              '.21.15.46.55.38A8.013 8.013 0 0 0 16 8c0-4.42-3.58-8-8-8z"/>'
              '</svg>')
DOC_SVG = ('<svg height="20" width="20" viewBox="0 0 24 24" '
           'fill="currentColor"><path d="M14 2H6c-1.1 0-2 .9-2 2v16c0 1.1.9 2 '
           '2 2h12c1.1 0 2-.9 2-2V8l-6-6zm2 16H8v-2h8v2zm0-4H8v-2h8v2zm-3-5V3.5'
           'L18.5 9H13z"/></svg>')


def read(path):
    with open(path) as fh:
        return fh.read()


def match_brace(s, open_idx):
    """Return index of the '}' matching the '{' at s[open_idx]."""
    depth = 0
    for i in range(open_idx, len(s)):
        if s[i] == "{":
            depth += 1
        elif s[i] == "}":
            depth -= 1
            if depth == 0:
                return i
    raise ValueError("unbalanced braces")


def extract_figures(tex):
    figs = []
    for m in re.finditer(r"\\begin\{figure\*?\}(.*?)\\end\{figure\*?\}", tex, re.S):
        block = m.group(1)
        lab = re.search(r"\\label\{(fig:[^}]+)\}", block)
        label = lab.group(1) if lab else "fig:unknown%d" % len(figs)

        caption = ""
        cidx = block.find("\\caption")
        if cidx != -1:
            b = block.find("{", cidx)
            caption = block[b + 1:match_brace(block, b)]

        inp = re.search(r"\\input\{([^}]+)\}", block)
        inc = re.search(r"\\includegraphics\[([^\]]*)\]\{([^}]+)\}", block)
        fig = {"label": label, "caption": caption}
        if inp:
            wm = re.search(r"resizebox\{([0-9.]+)\\textwidth\}", block)
            fig.update(gtype="tikz", gsrc=inp.group(1),
                       width=float(wm.group(1)) if wm else 0.7)
        elif inc:
            pm = re.search(r"page\s*=\s*(\d+)", inc.group(1))
            fig.update(gtype="plot", gsrc=inc.group(2),
                       page=int(pm.group(1)) if pm else 1)
        else:
            continue
        figs.append(fig)
    return figs


def caption_to_html(cap):
    out = []
    for part in re.split(r"(\$[^$]*\$)", cap):
        if part.startswith("$") and part.endswith("$"):
            out.append("\\(" + part[1:-1] + "\\)")
            continue
        t = part
        t = re.sub(r"\\(cref|Cref|ref|eqref|citep|citet|cite)\{[^}]*\}", "", t)
        t = re.sub(r"\\label\{[^}]*\}", "", t)
        t = re.sub(r"\\textit\{([^}]*)\}", r"<em>\1</em>", t)
        t = re.sub(r"\\emph\{([^}]*)\}", r"<em>\1</em>", t)
        t = re.sub(r"\\textbf\{([^}]*)\}", r"<strong>\1</strong>", t)
        t = t.replace("\\%", "%").replace("\\_", "_").replace("\\&", "&amp;")
        t = t.replace("~", " ")
        out.append(t)
    s = re.sub(r"\s+", " ", "".join(out)).strip()
    # tidy artifacts left by stripped \cref/\citep ("... from .")
    for a, b in ((" .", "."), (" ,", ","), (" ;", ";"), (" )", ")"),
                 ("( ", "("), (" from .", "."), (" in .", "."),
                 (" see .", "."), ("  ", " ")):
        s = s.replace(a, b)
    return s


def img_name(label):
    return label.replace(":", "_") + ".png"


def render_tikz(figs):
    tikz = [f for f in figs if f["gtype"] == "tikz"]
    if not tikz:
        return
    lines = [
        r"\documentclass{article}",
        r"\usepackage[active,tightpage]{preview}",
        r"\setlength\PreviewBorder{8pt}",
        r"\usepackage[dvipsnames]{xcolor}",
        r"\usepackage{graphicx}",
        r"\usepackage{amsmath,amssymb,mathtools}",
        r"\usepackage{pifont}",
        r"\usepackage{tikz, tikzsettings}",
        r"\usepackage{mpcsymbols}",
        r"\usepackage[version=4]{mhchem}",
        r"\newcommand{\Rm}{\mathrm}",
        r"\graphicspath{{./}{./figures/}{./figures/paper/}}",
        r"\begin{document}",
    ]
    for f in tikz:
        lines.append(
            r"\begin{preview}{\bfseries\Large\resizebox{%g\textwidth}{!}{\input{%s}}}\end{preview}"
            % (f["width"], f["gsrc"]))
    lines.append(r"\end{document}")
    with open(os.path.join(BUILD, "_site_fig.tex"), "w") as fh:
        fh.write("\n".join(lines))
    subprocess.run(["pdflatex", "-interaction=nonstopmode", "-halt-on-error",
                    "_site_fig.tex"], cwd=BUILD, check=True,
                   stdout=subprocess.DEVNULL)
    pdf = os.path.join(BUILD, "_site_fig.pdf")
    for i, f in enumerate(tikz, start=1):
        out = os.path.join(OUTFIG, img_name(f["label"])[:-4])
        subprocess.run(["pdftoppm", "-png", "-scale-to-x", "1300",
                        "-scale-to-y", "-1", "-f", str(i), "-l", str(i),
                        "-singlefile", pdf, out], check=True)
        f["img"] = img_name(f["label"])
    for ext in (".tex", ".pdf", ".aux", ".log"):
        p = os.path.join(BUILD, "_site_fig" + ext)
        if os.path.exists(p):
            os.remove(p)


def render_plots(figs):
    for f in figs:
        if f["gtype"] != "plot":
            continue
        src = os.path.join(FIGDIR, f["gsrc"] + ".pdf")
        out = os.path.join(OUTFIG, img_name(f["label"])[:-4])
        subprocess.run(["pdftoppm", "-png", "-scale-to-x", "1600",
                        "-scale-to-y", "-1", "-f", str(f["page"]),
                        "-l", str(f["page"]), "-singlefile", src, out],
                       check=True)
        f["img"] = img_name(f["label"])


HTML_HEAD = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>{title}</title>
<script>
MathJax = {{ tex: {{ inlineMath: [['\\\\(', '\\\\)']] }} }};
</script>
<script src="https://cdn.jsdelivr.net/npm/mathjax@3/es5/tex-mml-chtml.js" async></script>
<style>
  body {{ font-family: Georgia, "Times New Roman", serif; max-width: 900px;
         margin: 0 auto; padding: 2rem 1.2rem; color: #1a1a1a; line-height: 1.5; }}
  header {{ border-bottom: 2px solid #ddd; margin-bottom: 2rem; padding-bottom: 1rem; }}
  h1 {{ font-size: 1.55rem; line-height: 1.25; margin: 0 0 .4rem; }}
  .authors {{ color: #555; font-size: 1rem; }}
  .authors sup, .affiliations sup {{ font-size: .7rem; }}
  .affiliations {{ color: #666; font-size: .9rem; margin-top: .3rem; }}
  .corresponding {{ color: #666; font-size: .85rem; margin-top: .25rem; }}
  .corresponding a {{ color: #0969da; text-decoration: none; }}
  .resource-buttons {{ display: flex; gap: .8rem; flex-wrap: wrap;
                       margin-top: 1rem; }}
  .resource-link {{ display: inline-flex; align-items: center; gap: 8px;
                    padding: .6rem 1.1rem; border-radius: 6px;
                    text-decoration: none; font-weight: 600; font-size: .9rem;
                    font-family: -apple-system, system-ui, sans-serif;
                    transition: transform .2s, box-shadow .2s; }}
  .resource-link:hover {{ transform: translateY(-2px);
                          box-shadow: 0 4px 12px rgba(0,0,0,.12); }}
  .github-link {{ background: #24292e; color: #fff; }}
  .zenodo-link {{ background: #1682d4; color: #fff; }}
  .arxiv-link {{ background: #b31b1b; color: #fff; }}
  figure {{ margin: 2.5rem 0; padding: 0; }}
  figure img {{ width: 100%; height: auto; border: 1px solid #eee;
                background: #fff; display: block; }}
  figure a img {{ cursor: pointer; transition: border-color .15s; }}
  figure a:hover img {{ border-color: #0969da; }}
  figcaption {{ font-size: .92rem; color: #333; margin-top: .6rem; }}
  .supp-head {{ font-size: 1.25rem; margin: 2.5rem 0 1rem;
                padding-top: 1.5rem; border-top: 2px solid #ddd; }}
  figcaption b {{ color: #000; }}
  .src {{ white-space: nowrap; }}
  .src a {{ color: #0969da; text-decoration: none; font-size: .85rem; }}
  .src a:hover {{ text-decoration: underline; }}
  footer {{ border-top: 1px solid #ddd; margin-top: 3rem; padding-top: 1rem;
            color: #888; font-size: .85rem; }}
</style>
</head>
<body>
<header>
  <h1>{title}</h1>
  <div class="authors">{authors}</div>
  <div class="affiliations">{affiliations}</div>
  <div class="corresponding">{corresponding}</div>
  {resources}
</header>
"""

HTML_FIG = """<figure id="{label}">
  {open}<img src="figures/{img}" alt="Figure {n}">{close}
  <figcaption><b>Figure {n}.</b> {caption}{source}</figcaption>
</figure>
"""

HTML_SUPP_HEAD = """<h2 class="supp-head">Supplementary material</h2>
"""

HTML_TAIL = """<footer>
<a href="{repo_url}">{repo}</a> &middot; maintained by
<a href="https://github.com/{user}">{user}</a>
</footer>
</body>
</html>
"""


def resources_html():
    btns = []
    for url, cls, svg, label in (
            (GITHUB_URL, "github-link", GITHUB_SVG, "View on GitHub"),
            (ZENODO_URL, "zenodo-link", DOC_SVG, "Data on Zenodo"),
            (ARXIV_URL, "arxiv-link", DOC_SVG, "Read on arXiv")):
        if url:
            btns.append('<a href="%s" class="resource-link %s">%s%s</a>'
                        % (url, cls, svg, label))
    if not btns:
        return ""
    return '<div class="resource-buttons">%s</div>' % "".join(btns)


def build_html(figs):
    parts = [HTML_HEAD.format(title=TITLE, authors=AUTHORS,
                              affiliations=AFFILIATIONS,
                              corresponding=CORRESPONDING,
                              resources=resources_html())]
    seen_supp = False
    for f in figs:
        if f["num"].startswith("S") and not seen_supp:
            seen_supp = True
            parts.append(HTML_SUPP_HEAD)
        open_, close, source = "", "", ""
        if f["gtype"] == "plot":
            py = f["gsrc"] + ".py"
            url = "%s/%s" % (REPO_BASE, py)
            open_ = '<a href="%s" title="View source: %s">' % (url, py)
            close = "</a>"
            source = (' <span class="src"><a href="%s">[source: %s]</a></span>'
                      % (url, py))
        parts.append(HTML_FIG.format(label=f["label"], img=f["img"],
                                     n=f["num"],
                                     open=open_, close=close, source=source,
                                     caption=caption_to_html(f["caption"])))
    user = GITHUB_URL.rstrip("/").split("/")[-2]
    repo = GITHUB_URL.rstrip("/").split("/")[-1]
    parts.append(HTML_TAIL.format(repo_url=GITHUB_URL, repo=repo, user=user))
    with open(os.path.join(DOCS, "index.html"), "w") as fh:
        fh.write("".join(parts))


def main():
    if os.path.isdir(OUTFIG):
        shutil.rmtree(OUTFIG)
    os.makedirs(OUTFIG)
    figs = extract_figures(read(PAPER))
    for i, f in enumerate(figs, start=1):
        f["num"] = str(i)
    supp = extract_figures(read(SUPP))
    for i, f in enumerate(supp, start=1):
        f["num"] = "S%d" % i
    figs += supp
    render_tikz(figs)
    render_plots(figs)
    build_html(figs)
    print("Built site: %d figures (%d supplementary) -> %s"
          % (len(figs), len(supp), DOCS))


if __name__ == "__main__":
    main()
