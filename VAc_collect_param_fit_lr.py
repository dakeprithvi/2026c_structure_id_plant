# [depends] %LIB%/utils.py
# [makes] pickle
#
# Collector for the learning-rate / optimizer-mode sweep (vac_lr_all.job).
#
# The sweep runs two fitting scripts, each dropping one pickle per
# (case/mc/…/lr/mode) run into pickle/ so parallel cluster tasks never clobber
# each other.  This script grabs them all and merges them into ONE nested pickle
# named after itself (what the Makefile tracks: <script>.py -> <script>.pickle),
# which VAc_plot_lr.py then reads.
#
# Per-run filename patterns:
#   structured : VAc_param_fit_nn_param_noise_lr_{case}_{mc}_{cheat}_{noise}_{lr}_{mode}.pickle
#                VAc_param_fit_nn_param_noise_lr_{case}_{mc}_{cheat}_{noise}_direct.pickle
#   blackbox   : VAc_param_fit_bbox_param_noise_lr_{case}_{mc}_{noise}_{lr}_{mode}.pickle
#     cheat  : 'cheat' | 'nocheat'      noise : 'noise' | 'nonoise'
#     lr     : '0.001' | '0.01' | ...   mode  : 'adamw' | 'hybrid'
#     _direct: the cheat-pretraining stage (loss_rxn only); collected for
#              completeness, not used by the lr plot.
#
# Output nested layout (each leaf is the full per-run data dict, incl. loss/loss_v):
#   merged['struct'][case][mc][cheat][noise][mode][lr]   -> main fit
#   merged['struct'][case][mc][cheat][noise]['direct']   -> pretraining stage
#   merged['bbox'  ][case][mc][noise][mode][lr]          -> main fit

import sys, os, glob
sys.path.append('lib/')
from utils import PickleTool

root = os.path.dirname(os.path.abspath(__file__))
pickle_dir = os.path.join(root, 'pickle')

NN_BASE = 'VAc_param_fit_nn_param_noise_lr'
BBOX_BASE = 'VAc_param_fit_bbox_param_noise_lr'
MODES = ('adamw', 'hybrid')


def parse_name(fname):
    """Split an lr-run pickle filename into a family tag and its keys.

    Returns one of:
      ('struct', case, mc, cheat, noise, mode, lr)   main structured fit
      ('struct_direct', case, mc, cheat, noise)      cheat-pretraining stage
      ('bbox',   case, mc, noise, mode, lr)          main blackbox fit
    or None when the name doesn't match the sweep pattern.
    """
    stem = fname[:-len('.pickle')]
    if stem.startswith(NN_BASE + '_'):
        rest = stem[len(NN_BASE) + 1:]
        tok = rest.split('_')
        if tok[-1] == 'direct':
            # {case}_{mc}_{cheat}_{noise}_direct
            case, mc, cheat, noise = tok[0], tok[1], tok[2], tok[3]
            return 'struct_direct', case, int(mc), cheat, noise
        # {case}_{mc}_{cheat}_{noise}_{lr}_{mode}
        if len(tok) < 6 or tok[-1] not in MODES:
            return None
        mode, lr, noise, cheat, mc = tok[-1], tok[-2], tok[-3], tok[-4], tok[-5]
        case = '_'.join(tok[:-5])
        return 'struct', case, int(mc), cheat, noise, mode, float(lr)
    if stem.startswith(BBOX_BASE + '_'):
        rest = stem[len(BBOX_BASE) + 1:]
        tok = rest.split('_')
        # {case}_{mc}_{noise}_{lr}_{mode}
        if len(tok) < 5 or tok[-1] not in MODES:
            return None
        mode, lr, noise, mc = tok[-1], tok[-2], tok[-3], tok[-4]
        case = '_'.join(tok[:-4])
        return 'bbox', case, int(mc), noise, mode, float(lr)
    return None


merged = {'struct': {}, 'bbox': {}}
n_files = 0
paths = sorted(glob.glob(os.path.join(pickle_dir, 'VAc_param_fit_*_lr_*.pickle')))
for path in paths:
    parsed = parse_name(os.path.basename(path))
    if parsed is None:
        print(f'skip (unrecognized name): {os.path.basename(path)}')
        continue
    kind = parsed[0]
    payload = PickleTool.load(path, 'read')
    if kind == 'struct':
        _, case, mc, cheat, noise, mode, lr = parsed
        (merged['struct'].setdefault(case, {})
                         .setdefault(mc, {})
                         .setdefault(cheat, {})
                         .setdefault(noise, {})
                         .setdefault(mode, {}))[lr] = payload
    elif kind == 'struct_direct':
        _, case, mc, cheat, noise = parsed
        (merged['struct'].setdefault(case, {})
                         .setdefault(mc, {})
                         .setdefault(cheat, {})
                         .setdefault(noise, {}))['direct'] = payload
    else:  # bbox
        _, case, mc, noise, mode, lr = parsed
        (merged['bbox'].setdefault(case, {})
                       .setdefault(mc, {})
                       .setdefault(noise, {})
                       .setdefault(mode, {}))[lr] = payload
    n_files += 1

out_base = os.path.basename(__file__).replace('.py', '')
PickleTool.save(merged, f'{out_base}.pickle')

# brief summary of what was merged
print(f'merged {n_files} run pickles from {pickle_dir} into {out_base}.pickle')
for case in sorted(merged['struct']):
    for mc in sorted(merged['struct'][case]):
        for cheat in sorted(merged['struct'][case][mc]):
            for noise in sorted(merged['struct'][case][mc][cheat]):
                node = merged['struct'][case][mc][cheat][noise]
                modes = {m: sorted(node[m]) for m in node if m in MODES}
                has_direct = 'direct' in node
                print(f'  struct | {case} mc={mc} {cheat}/{noise}: '
                      f'lrs_by_mode={modes} direct={has_direct}')
for case in sorted(merged['bbox']):
    for mc in sorted(merged['bbox'][case]):
        for noise in sorted(merged['bbox'][case][mc]):
            node = merged['bbox'][case][mc][noise]
            modes = {m: sorted(node[m]) for m in node if m in MODES}
            print(f'  bbox   | {case} mc={mc} {noise}: lrs_by_mode={modes}')