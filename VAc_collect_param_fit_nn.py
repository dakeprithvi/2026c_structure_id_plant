# [depends] %LIB%/utils.py
# [makes] pickle
#
# Collector: the VAc_param_fit_nn_param_noise[_reduced].py runs each drop their
# own pickle into pickle/ (one per case/mc_iter/cheat/noise, plus a _direct
# variant), so parallel cluster jobs never clobber each other. This script grabs
# them all and merges them into ONE nested pickle named after itself, which is
# what the Makefile tracks (its rule expects <script>.py -> <script>.pickle).
#
# Per-run filename pattern produced by VAc_param_fit_nn_param_noise.py:
#   {base}_{case_type}_{mc_iter}_{cheat_tag}_{noise_tag}[_direct].pickle
#     base      : 'VAc_param_fit_nn_param_noise' or '..._reduced'
#     cheat_tag : 'cheat' or 'nocheat'
#     noise_tag : 'noise' or 'nonoise'
#     _direct   : present only for the cheat-pretraining result
#
# Output nested layout:
#   merged[base][case_type][mc_iter][cheat_tag][noise_tag][kind]
#     case_type : e.g. 'test', 'new'
#     mc_iter   : int
#     kind      : 'main' (the fit result) or 'direct' (cheat-pretraining result)

import sys, os, glob
sys.path.append('lib/')
from utils import PickleTool

# read the per-run pickles relative to this script, so the collector works the
# same whether it's run from the repo root or from the Makefile's build dir.
root = os.path.dirname(os.path.abspath(__file__))
pickle_dir = os.path.join(root, 'pickle')

# check the longer tag first in each pair, since the shorter one is a suffix of
# it ('cheat' is a suffix of 'nocheat'; 'noise' is a suffix of 'nonoise').
CHEAT_TAGS = ('nocheat', 'cheat')
NOISE_TAGS = ('nonoise', 'noise')


def parse_name(fname):
    """Split a run-pickle filename into
    (base, case_type, mc_iter, cheat_tag, noise_tag, kind).

    Returns None for files that don't match the expected pattern.
    Example: VAc_param_fit_nn_param_noise_reduced_test_3_cheat_noise_direct.pickle
          -> ('VAc_param_fit_nn_param_noise_reduced', 'test', 3, 'cheat', 'noise', 'direct')
    """
    stem = fname[:-len('.pickle')]
    kind = 'main'
    if stem.endswith('_direct'):
        kind = 'direct'
        stem = stem[:-len('_direct')]
    for ntag in NOISE_TAGS:
        if stem.endswith('_' + ntag):
            noise_tag = ntag
            stem = stem[:-len('_' + ntag)]
            break
    else:
        return None   # no noise tag -> not one of our (evolved) run pickles
    for ctag in CHEAT_TAGS:
        if stem.endswith('_' + ctag):
            cheat_tag = ctag
            stem = stem[:-len('_' + ctag)]
            break
    else:
        return None   # no cheat tag -> not one of our run pickles
    try:
        base, case_type, mc = stem.rsplit('_', 2)
        mc_iter = int(mc)
    except ValueError:
        return None
    return base, case_type, mc_iter, cheat_tag, noise_tag, kind


merged = {}
n_files = 0
paths = sorted(glob.glob(os.path.join(pickle_dir, 'VAc_param_fit_nn_param_noise*.pickle')))
for path in paths:
    parsed = parse_name(os.path.basename(path))
    if parsed is None:
        print(f'skip (unrecognized name): {os.path.basename(path)}')
        continue
    base, case_type, mc_iter, cheat_tag, noise_tag, kind = parsed
    node = (merged.setdefault(base, {})
                  .setdefault(case_type, {})
                  .setdefault(mc_iter, {})
                  .setdefault(cheat_tag, {})
                  .setdefault(noise_tag, {}))
    node[kind] = PickleTool.load(path, 'read')
    n_files += 1

# write the single consolidated pickle next to the cwd, named after this script
out_base = os.path.basename(__file__).replace('.py', '')
PickleTool.save(merged, f'{out_base}.pickle')

# brief summary of what was merged
print(f'merged {n_files} run pickles from {pickle_dir} into {out_base}.pickle')
for base in sorted(merged):
    for case_type in sorted(merged[base]):
        iters = sorted(merged[base][case_type])
        ctags = sorted({c for m in merged[base][case_type].values() for c in m})
        ntags = sorted({n for m in merged[base][case_type].values()
                        for c in m.values() for n in c})
        print(f'  {base} | {case_type}: mc_iters={iters} '
              f'cheat_tags={ctags} noise_tags={ntags}')
