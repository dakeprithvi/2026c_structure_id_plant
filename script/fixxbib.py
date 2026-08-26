#!/usr/bin/env python3
"""
Repair entry types in a .bib file exported by bibexport.

bibexport does not recognize non-standard entry types (e.g. @software or
@dataset) and writes them as a bare "@{key," that BibTeX skips, so the
reference silently prints as [?]. Each such entry is repaired here by looking
its type up in the original .bib databases, which are taken from the
\\bibdata command of the corresponding .aux file (or given with --bib).
"""
import sys
import os
import argparse
import re
import subprocess

parser = argparse.ArgumentParser(add_help=False, epilog=__doc__)
parser.add_argument("--help", action="help", help="print help")
parser.add_argument("--aux",
                    help=".aux file listing the original .bib databases")
parser.add_argument("--bib", action="append", default=[], metavar="FILE",
                    help="original .bib database (may be repeated)")
parser.add_argument("--default-type", default="misc",
                    help="entry type to use if the original is not found")
parser.add_argument("bibfile", help=".bib file to repair")

BROKENENTRY = re.compile(r"^@\s*\{\s*([^,\s]+)\s*,", re.MULTILINE)


def getdatabases(auxfile):
    """Returns the .bib databases listed in an .aux file."""
    databases = []
    with open(auxfile, "r") as aux:
        for line in aux:
            match = re.match(r"\\bibdata\{(.*)\}", line.strip())
            if match:
                databases.extend(match.group(1).split(","))
    return [d if d.endswith(".bib") else d + ".bib" for d in databases]


def findfile(file):
    """Returns a path to a .bib file, searching with kpsewhich if needed."""
    if os.path.exists(file):
        return file
    kpsewhich = subprocess.run(["kpsewhich", file], stdout=subprocess.PIPE)
    path = kpsewhich.stdout.decode().strip().split("\n")[0]
    return path if path else None


def gettypes(keys, databases):
    """Returns a {key : entry type} dictionary read from .bib databases."""
    entry = re.compile(r"@(\w+)\s*\{\s*(%s)\s*,"
                       % "|".join(re.escape(k) for k in keys))
    types = {}
    for database in databases:
        path = findfile(database)
        if path is None:
            print("  Warning: could not find database '{}'".format(database))
            continue
        with open(path, "r", errors="replace") as bib:
            for match in entry.finditer(bib.read()):
                types.setdefault(match.group(2), match.group(1))
    return types


def fixxbib(bibfile, aux=None, bib=(), default_type="misc"):
    """Repairs entry types in a .bib file. Returns the repaired {key : type}."""
    with open(bibfile, "r") as f:
        contents = f.read()
    keys = BROKENENTRY.findall(contents)
    if not keys:
        return {}

    databases = list(bib)
    if aux is not None:
        databases.extend(getdatabases(aux))
    types = gettypes(keys, databases)
    for key in keys:
        if key not in types:
            print("  Warning: no entry type found for '{}'; using '{}'"
                  .format(key, default_type))
            types[key] = default_type

    contents = BROKENENTRY.sub(lambda m: "@%s{%s," % (types[m.group(1)],
                                                      m.group(1)), contents)
    with open(bibfile, "w") as f:
        f.write(contents)
    return types


def main(args):
    """Runs main function."""
    args = vars(parser.parse_args(args))
    fixed = fixxbib(args.pop("bibfile"), **args)
    for (key, type) in fixed.items():
        print("  Repaired entry type of {}: @{}".format(key, type))


if __name__ == "__main__":
    main(sys.argv[1:])
