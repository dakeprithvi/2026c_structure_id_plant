# [depends] %LIB%/utils.py
# [makes] pickle
#
# Collector: the VAc_param_fit_bbox_param_noise.py runs each drop their own
# pickle into pickle/ (one per case/mc_iter/noise), so parallel cluster jobs
# never clobber each other. This script grabs them all and merges them into ONE
# nested pickle named after itself, which is what the Makefile tracks (its rule
# expects <script>.py -> <script>.pickle).
#
# Per-run filename pattern produced by VAc_param_fit_bbox_param_noise.py:
#   {base}_{case_type}_{mc_iter}_{noise_tag}.pickle
#     base      : 'VAc_param_fit_bbox_param_noise'
#     noise_tag : 'noise' or 'nonoise'
#
# Output nested layout:
#   merged[base][case_type][mc_iter][noise_tag]
#     case_type : e.g. 'test', 'new'
#     mc_iter   : int

import sys, os, glob
sys.path.append('lib/')
from utils import PickleTool

# read the per-run pickles relative to this script, so the collector works the
# same whether it's run from the repo root or from the Makefile's build dir.
root = os.path.dirname(os.path.abspath(__file__))
pickle_dir = os.path.join(root, 'pickle')

# check the longer tag first, since the shorter one is a suffix of it
# ('noise' is a suffix of 'nonoise').
NOISE_TAGS = ('nonoise', 'noise')


def parse_name(fname):
    """Split a run-pickle filename into (base, case_type, mc_iter, noise_tag).

    Returns None for files that don't match the expected pattern.
    Example: VAc_param_fit_bbox_param_noise_test_3_nonoise.pickle
          -> ('VAc_param_fit_bbox_param_noise', 'test', 3, 'nonoise')
    """
    stem = fname[:-len('.pickle')]
    for ntag in NOISE_TAGS:
        if stem.endswith('_' + ntag):
            noise_tag = ntag
            stem = stem[:-len('_' + ntag)]
            break
    else:
        return None   # no noise tag -> not one of our run pickles
    try:
        base, case_type, mc = stem.rsplit('_', 2)
        mc_iter = int(mc)
    except ValueError:
        return None
    return base, case_type, mc_iter, noise_tag


merged = {}
n_files = 0
paths = sorted(glob.glob(os.path.join(pickle_dir, 'VAc_param_fit_bbox_param_noise*.pickle')))
for path in paths:
    parsed = parse_name(os.path.basename(path))
    if parsed is None:
        print(f'skip (unrecognized name): {os.path.basename(path)}')
        continue
    base, case_type, mc_iter, noise_tag = parsed
    node = (merged.setdefault(base, {})
                  .setdefault(case_type, {})
                  .setdefault(mc_iter, {}))
    node[noise_tag] = PickleTool.load(path, 'read')
    n_files += 1

# write the single consolidated pickle next to the cwd, named after this script
out_base = os.path.basename(__file__).replace('.py', '')
PickleTool.save(merged, f'{out_base}.pickle')

# brief summary of what was merged
print(f'merged {n_files} run pickles from {pickle_dir} into {out_base}.pickle')
for base in sorted(merged):
    for case_type in sorted(merged[base]):
        iters = sorted(merged[base][case_type])
        ntags = sorted({n for m in merged[base][case_type].values() for n in m})
        print(f'  {base} | {case_type}: mc_iters={iters} noise_tags={ntags}')
