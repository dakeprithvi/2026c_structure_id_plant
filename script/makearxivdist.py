#!/usr/bin/env python3
"""
Make an arXiv-ready .zip file from a distribution .zip made by maketexdist.py.

arXiv's file checker deletes any X.pdf whose base name matches an X.tex in the
same submission, assuming the .pdf is a byproduct of the .tex. That assumption
is wrong for Inkscape's "PDF + LaTeX" figures, where X.tex is a text overlay
that \\includegraphics{X.pdf}. Such figures are renamed here (X.pdf becomes
X-fig.pdf, and the references in the bundled .tex files are updated) so that
arXiv keeps them.

arXiv also deletes the .aux files that the xr package needs for
\\externaldocument cross-references, so those labels are instead written to a
plain .tex file that is \\input by the document.

Duplicate entries are also collapsed, and files may be dropped from or added to
the bundle.
"""
import sys
import os
import argparse
import re
import zipfile

parser = argparse.ArgumentParser(add_help=False, epilog=__doc__)
parser.add_argument("--help", action="help", help="print help")
parser.add_argument("--output", help="name for output .zip file")
parser.add_argument("--drop", action="append", default=[], metavar="NAME",
                    help="file to remove from the bundle (may be repeated)")
parser.add_argument("--add", action="append", default=[], metavar="FILE",
                    help="file to add to the bundle (may be repeated)")
parser.add_argument("--xr-labels", action="append", default=[],
                    metavar="FILE",
                    help=".aux file whose labels replace an \\externaldocument"
                         " reference (may be repeated)")
parser.add_argument("--suffix", default="-fig",
                    help="suffix for renamed figure .pdf files")
parser.add_argument("zipfile", help=".zip file to convert")


def readbundle(zipname, drop=()):
    """Reads a .zip file and returns an ordered {name : bytes} dictionary."""
    drop = set(drop)
    files = {}
    with zipfile.ZipFile(zipname, "r") as dist:
        for info in dist.infolist():
            name = os.path.basename(info.filename)
            if name in drop:
                continue
            contents = dist.read(info)
            if name in files and files[name] != contents:
                raise ValueError("Duplicate entries for '{}' in '{}' differ!"
                                 .format(name, zipname))
            files[name] = contents
    return files


def fixpdfnameclashes(files, suffix="-fig"):
    """Renames each X.pdf that clashes with an X.tex. Returns renamed names."""
    texbases = set(os.path.splitext(f)[0] for f in files
                   if f.endswith(".tex"))
    renamed = {}
    for name in [f for f in files if f.endswith(".pdf")]:
        (base, ext) = os.path.splitext(name)
        if base in texbases:
            newname = base + suffix + ext
            if newname in files:
                raise ValueError("Cannot rename '{}': '{}' already exists!"
                                 .format(name, newname))
            files[newname] = files.pop(name)
            renamed[name] = newname

    # Update graphics references in the .tex files.
    for name in [f for f in files if f.endswith(".tex")]:
        source = files[name].decode()
        for (old, new) in renamed.items():
            base = os.path.splitext(old)[0]
            source = source.replace("{%s}" % old, "{%s}" % new)
            source = re.sub(r"(\\includegraphics[^{}]*)\{%s\}"
                            % re.escape(base),
                            r"\1{%s}" % os.path.splitext(new)[0], source)
        files[name] = source.encode()
    return renamed


def addxrlabels(files, auxfile):
    """Replaces an \\externaldocument reference by an \\input of its labels."""
    base = os.path.splitext(os.path.basename(auxfile))[0]
    with open(auxfile, "r") as aux:
        labels = [l for l in aux if l.startswith("\\newlabel{")]
    if not labels:
        raise ValueError("No labels found in '{}'!".format(auxfile))
    labelfile = base + "-labels.tex"
    files[labelfile] = ("%% Labels of {}, for arXiv, which deletes .aux files.\n"
                        .format(base + ".tex") + "".join(labels)).encode()

    external = re.compile(r"\\externaldocument(\[[^]]*\])?\{%s\}"
                          % re.escape(base))
    for name in [f for f in files if f.endswith(".tex")]:
        source = files[name].decode()
        (source, count) = external.subn(r"\\input{%s}"
                                        % os.path.splitext(labelfile)[0],
                                        source)
        if count:
            files[name] = source.encode()
    return labelfile


def makearxivdist(zipfile_, output=None, drop=(), add=(), xr_labels=(),
                  suffix="-fig"):
    """Makes an arXiv-ready .zip file from a distribution .zip file."""
    if output is None:
        output = os.path.splitext(zipfile_)[0] + "_arxiv.zip"
    files = readbundle(zipfile_, drop=drop)
    for file in add:
        with open(file, "rb") as f:
            files[os.path.basename(file)] = f.read()
    for auxfile in xr_labels:
        print("  arXiv bundle: added {} in place of \\externaldocument"
              .format(addxrlabels(files, auxfile)))
    renamed = fixpdfnameclashes(files, suffix=suffix)
    with zipfile.ZipFile(output, "w") as dist:
        for (name, contents) in files.items():
            dist.writestr(name, contents)
    return renamed


def main(args):
    """Runs main function."""
    args = vars(parser.parse_args(args))
    args["zipfile_"] = args.pop("zipfile")
    renamed = makearxivdist(**args)
    for (old, new) in renamed.items():
        print("  arXiv bundle: renamed {} to {} (clashed with {})"
              .format(old, new, os.path.splitext(old)[0] + ".tex"))


if __name__ == "__main__":
    main(sys.argv[1:])
