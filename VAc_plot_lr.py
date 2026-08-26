# [depends] VAc_collect_param_fit_lr.pickle
#
# Learning-rate sweep: training/validation loss vs iteration.
#
# VAc_collect_param_fit_lr.py merges the sweep runs into one nested pickle:
#   struct[case][mc][cheat][noise][mode][lr] -> run dict (loss, loss_v, ...)
#   bbox  [case][mc][noise][mode][lr]        -> run dict
# with mode in {adamw, hybrid} (hybrid = AdamW then LBFGS line search) and four
# learning rates.
#
# One page per model SETTING (cmeas_rinit / cmeas / cmeas bbox, plus the noise
# variants once they are run).  Each page is a 2x2 grid:
#     rows    = training loss (top) / validation loss (bottom)
#     columns = AdamW (left) / AdamW->LBFGS (right)
# and every panel overlays the four learning-rate curves against iteration
# (log y).  The learning-rate legend is drawn only in the last panel; panels are
# tagged (a)-(d) just outside their top-left corner.

import sys, os
sys.path.append('lib/')
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from utils import PickleTool

os.makedirs('plots', exist_ok=True)
file_base = os.path.basename(__file__).replace('.py', '')
data = PickleTool.load('VAc_collect_param_fit_lr.pickle', 'read')

# master switch for figure titles (off for paper figures)
title = False

CASE = 'test'
MC = 0

# the two optimizer schemes, in column order, with display labels
MODE_COLS = [('adamw', 'Adam'), ('hybrid', r'Adam $\to$ LBFGS')]

# the two loss curves, in row order: (data key, row label)
ROW_KEYS = [('loss', 'Training loss'), ('loss_v', 'Validation loss')]

# fixed colour per learning rate (assigned in sorted order), so a given lr keeps
# the same colour across every panel and page.
LR_COLORS = ['#1f77b4', '#d62728', '#2ca02c', '#9467bd', '#ff7f0e', '#17becf']

# settings to plot, in order; each becomes one page when its data is present.
SETTINGS = [
    dict(fam='struct', cheat='cheat',   noise='nonoise',
         title=r'cmeas\_rinit'),
    dict(fam='struct', cheat='nocheat', noise='nonoise',
         title=r'cmeas'),
    dict(fam='struct', cheat='nocheat', noise='noise',
         title=r'cmeas\_noise '),
    dict(fam='bbox',                    noise='nonoise',
         title=r'bbox'),
    dict(fam='bbox',                    noise='noise',
         title=r'bbox\_noise (black-box)'),
]


def resolve(s):
    """Return {mode: {lr: run_dict}} for a setting, or None when absent."""
    try:
        if s['fam'] == 'struct':
            node = data['struct'][CASE][MC][s['cheat']][s['noise']]
        else:
            node = data['bbox'][CASE][MC][s['noise']]
    except (KeyError, TypeError):
        return None
    modes = {m: node[m] for m, _ in MODE_COLS if m in node}
    return modes or None


# per-row key for the frozen optimal-init (lr=0, no-training) reference line,
# read from the cheat-pretraining 'direct' node. Only drawn where present.
INIT_REF_KEYS = {'loss': 'loss_at_init', 'loss_v': 'loss_v_at_init'}


def init_ref(s):
    """{'loss': value, 'loss_v': value} at the optimal cmeas_rinit params, or None.

    These come from the cheat-pretraining 'direct' node and are only meaningful
    for the cmeas_rinit setting (struct/cheat), where the fit is initialized at
    that optimum. Absent -> None (no reference line drawn)."""
    if s['fam'] != 'struct' or s.get('cheat') != 'cheat':
        return None
    try:
        direct = data['struct'][CASE][MC][s['cheat']][s['noise']]['direct']
    except (KeyError, TypeError):
        return None
    if 'loss_at_init' not in direct or 'loss_v_at_init' not in direct:
        return None
    return {row: float(direct[k]) for row, k in INIT_REF_KEYS.items()}


def add_panel_tag(ax, tag):
    """Bold (a)/(b)/... just outside the panel's top-left corner."""
    ax.text(-0.16, 1.05, tag, transform=ax.transAxes, fontsize=11,
            fontweight='bold', va='bottom', ha='left')


def lr_color_map(all_lrs):
    return {lr: LR_COLORS[i % len(LR_COLORS)] for i, lr in enumerate(sorted(all_lrs))}


def setting_page(pdf, s):
    modes = resolve(s)
    if modes is None:
        print(f"  skip (no data): {s['title']}")
        return False

    all_lrs = sorted({lr for md in modes.values() for lr in md})
    colors = lr_color_map(all_lrs)
    ref = init_ref(s)   # frozen optimal-init reference (cmeas_rinit only), or None

    fig, axes = plt.subplots(2, 2, figsize=(8, 5.5), sharex=True, sharey='row',
                             squeeze=False, layout='constrained')
    if title:
        fig.suptitle(f'Learning-rate sweep — {s["title"]}', fontsize=13)

    tags = iter(['(a)', '(b)', '(c)', '(d)'])
    for r, (key, row_label) in enumerate(ROW_KEYS):
        for c, (mode, mode_label) in enumerate(MODE_COLS):
            ax = axes[r, c]
            add_panel_tag(ax, next(tags))
            md = modes.get(mode, {})
            for lr in sorted(md):
                y = np.asarray(md[lr][key], dtype=float)
                ax.plot(np.arange(y.size), y, color=colors[lr], lw=1.3,
                        label=f'{lr:g}')
            # frozen optimal-init loss (no training): the value the fit is handed
            # and immediately drifts away from. Drawn only for cmeas_rinit.
            if ref is not None:
                ax.axhline(ref[key], color='k', ls='-.', lw=1.1,
                           label='0')
            ax.set_yscale('log')
            ax.grid(True, which='both', ls=':', lw=0.4, alpha=0.5)
            if c != 0:
                ax.axvline(x=250, color='k', ls='--', lw=0.8, alpha=0.5)
            if r == 0:
                ax.set_title(mode_label, fontsize=11)
            if r == len(ROW_KEYS) - 1:
                ax.set_xlabel('Epoch')
            if c == 0:
                ax.set_ylabel(row_label)

    # learning-rate legend only in the last panel of each row, entries in
    # descending lr order with the lr=0 (no-training) reference on top.
    def ordered_legend(ax):
        handles, labels = ax.get_legend_handles_labels()
        # lr=0 (no-training reference) pinned on top, remaining lrs descending.
        order = sorted(range(len(labels)),
                       key=lambda i: (float(labels[i]) != 0.0, -float(labels[i])))
        ax.legend([handles[i] for i in order], [labels[i] for i in order],
                  title='learning rate', fontsize=8, title_fontsize=8,
                  frameon=True, loc='upper right')

    ordered_legend(axes[0, -1])
    ordered_legend(axes[-1, -1])
    pdf.savefig(fig)
    plt.close(fig)
    return True


with PdfPages(f'{file_base}.pdf') as pdf:
    n = sum(setting_page(pdf, s) for s in SETTINGS)
