# [depends] VAc_perturb_plant.pickle
# [depends] VAc_meas.pickle

import sys, os
sys.path.append('lib/')
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.lines import Line2D
from utils import PickleTool

os.makedirs('plots', exist_ok=True)
file_base = os.path.basename(__file__).replace('.py', '')
data = PickleTool.load('VAc_perturb_plant.pickle', 'read')
data_meas = PickleTool.load('VAc_meas.pickle', 'read')
data_meas = data_meas['test']  

# master switch for figure titles: True adds the suptitle to every page, False
# leaves the plots untitled (e.g. for paper figures).
title = False

# colors shared with VAc_plot_train_val_param_nn.py: every NN identified plant
# takes one palette entry; the true plant ('test') is red and the parametric
# model ('param') black.
_MODEL_PALETTE = ['blue', 'green', 'darkorange', 'purple', 'brown', 'teal', 'crimson']
TRUE_COLOR  = 'red'
PARAM_COLOR = 'black'

for plant in data:
    r1 = np.asarray(data[plant]['rate_1'], dtype=float).flatten()
    r2 = np.asarray(data[plant]['rate_2'], dtype=float).flatten()
    data[plant]['rate_ratio'] = np.where(r2 != 0, r1 / r2, np.nan)

# The four NN identified plants are the focus: each holds n samples, one per MC
# identification/fit, so its optimal outputs form a distribution. They are drawn
# in the canonical case order (rmeas -> cmeas_rinit -> cmeas -> cmeas_noise,
# matching VAc_plot_train_val_param_nn.py), each taking the next palette entry.
NN_ORDER  = ['rmeas', 'cmeas_rinit', 'cmeas', 'cmeas_noise']
nn_plants = [p for p in NN_ORDER if p in data]
# 'test' (true plant, red) and 'param' (parametric model, black) are single-value
# references overlaid as vertical lines on the histograms.
ref_plants = [p for p in ('test', 'param') if p in data]

# Each NN case must keep the SAME palette colour it has in
# VAc_plot_train_val_param_nn.py, where colours are assigned by an alphabetical
# sort of the flattened model-type string (`model_color` there). We rebuild that
# model-type string per case and reuse the identical sort so the mapping matches:
#   cmeas_rinit -> blue, rmeas -> green, cmeas_noise -> darkorange, cmeas -> purple.
_MBASE = 'nn_param_noise'
_NT_COORDS = {
    'rmeas':       ('cheat',   'nonoise', 'direct'),
    'cmeas_rinit':  ('cheat',   'nonoise', 'main'),
    'cmeas':       ('nocheat', 'nonoise', 'main'),
    'cmeas_noise': ('nocheat', 'noise',   'main'),
}
def _model_type(p):
    cheat, noise, kind = _NT_COORDS[p]
    nt = f'{_MBASE}_{cheat}_{noise}'
    return nt + ('_direct' if kind == 'direct' else '')

_nt_sorted = sorted(_model_type(p) for p in nn_plants)
plant_colors = {p: _MODEL_PALETTE[_nt_sorted.index(_model_type(p)) % len(_MODEL_PALETTE)]
                for p in nn_plants}
plant_colors['test']  = TRUE_COLOR
plant_colors['param'] = PARAM_COLOR

output_keys   = ['na_model', 'ne_model', 'no_model', 'alpha_model', 'ta_model', 'profit_model', 'hess_cond', 'rate_1', 'rate_2', 'rate_ratio']
range_map = {'na_model': 'na', 'ne_model': 'ne', 'no_model': 'no', 'alpha_model': 'alpha', 'ta_model': 'ta'}
output_labels = [r'$N_\mathrm{A}^*$', r'$N_\mathrm{Ee}^*$', r'$N_\mathrm{O}^*$', r'$\alpha^*$', r'$T^*$', 'Profit', r'$\kappa(H)$', r'$r_1$', r'$r_2$', r'$r_1/r_2$']

u_scale = {'na_model': 785 / 60, 'ne_model': 831 / 60, 'no_model': 831 / 60,
           'alpha_model': 1, 'beta_model': 1, 'ta_model': 150 + 273.15,
           'profit_loss': 100}  # fraction -> percent

# 'test' and 'param' get friendly labels; the NN keys are already display names
plant_legend = {'test': 'Plant', 'param': 'Param'}

def hist_ax(ax, arr, xlabel, color, range=None, scale=1.0):
    arr = np.asarray(arr, dtype=float).flatten() * scale
    arr = arr[~np.isnan(arr)]
    arr = arr[arr != 0]
    arr = np.round(arr, 8)
    ax.hist(arr, bins=20, range=range,color=color, edgecolor='black', alpha=0.7)
    ax.set_xlabel(xlabel)
    ax.set_ylabel('Count')


def ref_lines(ax, key, scale=1.0):
    """Overlay the single-value references (true plant red, param black) as
    vertical dashed lines, so the n-sample NN distributions can be read against
    them. Silently skips a reference whose value is missing/zero."""
    for ref in ref_plants:
        v = np.asarray(data[ref][key], dtype=float).flatten() * scale
        v = v[~np.isnan(v)]
        v = v[v != 0]
        if len(v):
            ax.axvline(v[0], color=plant_colors[ref], linestyle='--', linewidth=1.2)


def centered_axes(fig, n, ncols):
    # one axis per item, last (partial) row centered. Each plot spans two
    # gridspec columns; a partial row is offset by half a slot so it sits
    # centered under the full rows above.
    nrows = int(np.ceil(n / ncols))
    gs = fig.add_gridspec(nrows, 2 * ncols)
    axes = []
    for i in range(n):
        r = i // ncols
        m = min(ncols, n - r * ncols)          # plots in this row
        k = i - r * ncols                       # position within the row
        offset = (2 * ncols - 2 * m) // 2       # half-slot centering offset
        c0 = offset + 2 * k
        axes.append(fig.add_subplot(gs[r, c0:c0 + 2]))
    return axes


# --- Profit-loss summary: mean, std and max over the MC fits, per NN plant ---
# profit_loss is a fraction; scale to percent with the same factor used in plots.
_pl_scale = u_scale.get('profit_loss', 1.0)
print('Profit loss (%) summary over MC fits:')
print(f'  {"case":<14}{"n":>4}{"mean":>10}{"std":>10}{"worst":>10}')
for plant in nn_plants:
    pl = np.asarray(data[plant]['profit_loss'], dtype=float).flatten() * _pl_scale
    pl = pl[~np.isnan(pl)]
    if len(pl):
        print(f'  {plant:<14}{len(pl):>4}{np.mean(pl):>10.4g}'
              f'{np.std(pl):>10.4g}{np.min(pl):>10.4g}')
    else:
        print(f'  {plant:<14}{0:>4}{"--":>10}{"--":>10}{"--":>10}')

# --- Reduced-Hessian definiteness summary over MC fits ---
# eigcheck_hess is a per-fit categorical flag (pos_def / neg_def / *_semidef /
# indefinite / empty / undefined) describing the reduced Hessian at each optimum.
# Tabulate the per-case counts so its health can be read at a glance.
_eig_plants = [p for p in nn_plants + ref_plants
               if 'eigcheck_hess' in data.get(p, {})]
_eig_order = ['pos_def', 'neg_def', 'pos_semidef', 'neg_semidef',
              'indefinite', 'empty', 'undefined']
_all_labels = [str(x) for p in _eig_plants for x in data[p]['eigcheck_hess']]
# canonical categories that actually occur, then any unexpected labels
_cats = [c for c in _eig_order if c in _all_labels]
_cats += sorted(set(_all_labels) - set(_eig_order))
print('Reduced-Hessian definiteness summary over MC fits:')
print(f'  {"case":<18}{"n":>4}' + ''.join(f'{c:>13}' for c in _cats))
for plant in _eig_plants:
    labels = [str(x) for x in data[plant]['eigcheck_hess']]
    print(f'  {plant:<18}{len(labels):>4}'
          + ''.join(f'{labels.count(c):>13}' for c in _cats))

with PdfPages(f'{file_base}.pdf') as pdf:

    # --- Summary page: mean ± std bar chart, one bar group per plant ---
    n_out = len(output_keys)
    x = np.arange(n_out)
    width = 0.35

    fig, ax = plt.subplots(figsize=(8, 5))
    if title:
        fig.suptitle(r'NN-plant optimal outputs: $\sigma/\mu$', fontsize=13)
    # only the NN plants carry a distribution (n MC fits); the single-value
    # references have no spread, so they are omitted from the CV chart.
    for pi, plant in enumerate(nn_plants):
        cv_vals = np.zeros(n_out)
        for i, key in enumerate(output_keys):
            arr = np.asarray(data[plant][key], dtype=float).flatten()
            arr = arr[~np.isnan(arr)]
            arr = arr[arr != 0]
            mu = np.mean(arr) if len(arr) > 0 else 0
            if len(arr) > 0 and mu != 0:
                cv_vals[i] = np.std(arr) / mu * 100
        offsets = x + (pi - (len(nn_plants) - 1) / 2) * width
        # NOTE: sigma/mu is dimensionless and invariant under unit scaling,
        # so u_scale is intentionally NOT applied here.
        ax.bar(offsets, cv_vals, width=width,
               color=plant_colors.get(plant, 'steelblue'), edgecolor='black',
               alpha=0.7, label=plant_legend.get(plant, plant))
    ax.set_xticks(x)
    ax.set_xticklabels(output_labels)
    ax.set_ylabel(r'$\sigma/\mu$ (%)')
    ax.set_ylim(0, 100)
    ax.legend()
    plt.tight_layout()
    pdf.savefig(fig)
    plt.close(fig)

    # --- Per-plant histogram pages ---
    # show the first 5 outputs, the profit-loss distribution and the Hessian
    # condition number on a 3x3 grid (hess_cond in the 7th panel); the two
    # trailing cells are left blank.
    hist_keys   = output_keys[:5]   + ['profit_loss', 'hess_cond']
    hist_labels = output_labels[:5] + ['Profit loss (\%)', r'$\kappa(H)$']
    n_show = len(hist_keys)
    ncols, nrows = 3, 3
    for plant in nn_plants:
        fig, axes = plt.subplots(nrows, ncols, figsize=(2.5 * ncols, 2.5 * nrows))
        axes = axes.flatten()
        if title:
            fig.suptitle(f'Optimal-output distributions over MC fits — Plant: {plant_legend.get(plant, plant)}', fontsize=13)
        color = plant_colors.get(plant, 'steelblue')
        # multiple optima are possible — let the axes span the full spread
        tight = 0
        for ax, key, label in zip(axes, hist_keys, hist_labels):
            scale = u_scale.get(key, 1.0)
            range_for_key = None
            if key in range_map.keys():
                range_for_key = [data_meas['opti_conc']['physical_bounds'][range_map[key]][0], data_meas['opti_conc']['physical_bounds'][range_map[key]][1]]
            hist_ax(ax, data[plant][key], label, color, range=range_for_key,
                    scale=scale)
            ref_lines(ax, key, scale)
        for ax in axes[n_show:]:
            ax.axis('off')
        plt.tight_layout()
        pdf.savefig(fig)
        plt.close(fig)

    # --- Overlaid all-plant page ---
    # the same 7 outputs on one page, every plant overlaid as an outline (step)
    # histogram in its palette colour, with a single shared legend. Densities
    # (not raw counts) are plotted so plants with different numbers of
    # successful optima stay comparable, and all plants share one x-range per
    # panel so the outlines line up.
    def panel_range(key, scale):
        if key in range_map:
            b = data_meas['opti_conc']['physical_bounds'][range_map[key]]
            return [b[0], b[1]]
        # unbounded output (profit loss): share one range across all NN plants
        vals = np.concatenate([np.asarray(data[p][key], dtype=float).flatten()
                               for p in nn_plants]) * scale
        vals = vals[~np.isnan(vals)]
        vals = vals[vals != 0]
        return [vals.min(), vals.max()] if len(vals) else None

    ncols, nrows = 3, 3
    fig, axes = plt.subplots(nrows, ncols, figsize=(2. * ncols, 2. * nrows))
    axes = axes.flatten()
    if title:
        fig.suptitle('Optimal-output distributions across NN plants', fontsize=13)
    for ax, key, label in zip(axes, hist_keys, hist_labels):
        scale = u_scale.get(key, 1.0)
        rng = panel_range(key, scale)
        for plant in nn_plants:
            arr = np.asarray(data[plant][key], dtype=float).flatten() * scale
            arr = arr[~np.isnan(arr)]
            arr = arr[arr != 0]
            if len(arr) == 0:
                continue
            ax.hist(arr, bins=20, range=rng, density=True, histtype='step',
                    linewidth=1.5, color=plant_colors.get(plant, 'steelblue'))
        ref_lines(ax, key, scale)
        ax.set_xlabel(label)
        ax.set_ylabel('Density')
    for ax in axes[len(hist_keys):]:
        ax.axis('off')

    legend_plants = nn_plants + ref_plants
    handles = [Line2D([0], [0], color=plant_colors.get(p, 'steelblue'),
                      lw=1.5, linestyle='--' if p in ref_plants else '-',
                      label=plant_legend.get(p, p)) for p in legend_plants]
    fig.legend(handles=handles, loc='lower center',
               ncol=min(len(legend_plants), 4), frameon=False)
    fig.tight_layout(rect=[0, 0.08, 1, 1])
    pdf.savefig(fig)
    plt.close(fig)

    # --- Profit-loss comparison page: NN cases (columns) x seeds (rows) ---
    # one histogram per (seed, case): columns are the NN cases in canonical order
    # (a)-(d), rows are the outer-MC seeds. All panels share one x/y range so the
    # distributions are directly comparable. Requires the per-sample 'seed' tag
    # written by VAc_perturb_plant.py; the page is skipped if it is absent.
    pl_plants  = nn_plants
    seeds_all  = sorted({s for p in pl_plants for s in data[p].get('seed', [])})
    if pl_plants and seeds_all:
        scale = u_scale.get('profit_loss', 1.0)
        vals = np.concatenate([np.asarray(data[p]['profit_loss'], dtype=float).flatten()
                               for p in pl_plants]) * scale
        vals = vals[~np.isnan(vals)]
        vals = vals[vals != 0]
        # round exactly as the per-panel data is rounded below, otherwise
        # np.round can push the extreme samples a hair outside [min, max] and
        # hist(range=...) silently drops them (it is always the worst-case
        # sample that disappears).
        vals = np.round(vals, 8)
        rng = [vals.min(), vals.max()] if len(vals) else None

        ncol, nrow = len(pl_plants), len(seeds_all)
        fig, axes = plt.subplots(nrow, ncol, figsize=(1.7 * ncol, 1.6 * nrow),
                                 sharex=True, sharey=True, squeeze=False)
        if title:
            fig.suptitle('Profit loss across NN plants, by seed', fontsize=13)
        for row, seed in enumerate(seeds_all):
            for col, plant in enumerate(pl_plants):
                ax = axes[row][col]
                seed_arr = np.asarray(data[plant].get('seed', []))
                pl_arr   = np.asarray(data[plant]['profit_loss'], dtype=float)
                sel = pl_arr[seed_arr == seed] if len(seed_arr) else pl_arr[:0]
                # y axis is relative frequency in percent (each panel sums to
                # 100%), not raw count.
                a = np.asarray(sel, dtype=float).flatten() * scale
                a = a[~np.isnan(a)]
                a = a[a != 0]
                a = np.round(a, 8)
                if len(a):
                    ax.hist(a, bins=20, range=rng,
                            weights=np.full(len(a), 100.0 / len(a)),
                            color=plant_colors.get(plant, 'steelblue'),
                            edgecolor='black', alpha=0.7)
                ax.set_xlabel('Profit loss (\%)')
                ax.set_ylim(0, 100)
                # compare everything against the true plant only: its (zero-loss)
                # baseline as a black dotted line — no param reference here.
                if 'test' in data:
                    tv = np.asarray(data['test']['profit_loss'], dtype=float).flatten() * scale
                    tv = tv[~np.isnan(tv)]
                    if len(tv):
                        ax.axvline(tv[0], color='black', linestyle=':', linewidth=1.2)
                if row == 0:
                    ax.set_title(f'({chr(ord("a") + col)})')
                if row < nrow - 1:
                    ax.set_xlabel('')
                if col == 0:
                    ax.set_ylabel('Frequency (\%)')
                    # seed label in the top-left corner (1-indexed), not the ylabel
                    ax.text(0.05, 0.95, f'seed {row + 1}', transform=ax.transAxes,
                            ha='left', va='top')
                else:
                    ax.set_ylabel('')
        plt.tight_layout()
        pdf.savefig(fig)
        plt.close(fig)
