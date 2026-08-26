# [depends] %LIB%/utils.py
# [depends] %LIB%/config.py
# [depends] VAc_collect_param_fit_nn.pickle
# [depends] VAc_meas.pickle

import sys, os
sys.path.append('lib/')
import matplotlib.pyplot as plt
import numpy as np
from collections import defaultdict
from matplotlib.backends.backend_pdf import PdfPages
from utils import PickleTool
from config import labels, get_label
os.environ['JAX_PLATFORMS'] = 'cpu'
os.makedirs('plots', exist_ok=True)

# master switch for figure titles: True adds the suptitle to every page, False
# leaves the plots untitled (e.g. for paper figures).
title = False

# ── constants ─────────────────────────────────────────────────────────────────
# CI band is only drawn when a group has more than one MC iteration; with a
# single run (mc_iter 0 only) the fit pages are plain lines and the parity pages
# are rasterized scatter.
CI_LO, CI_HI = 5, 95          # percentile bounds for shaded confidence band

color_meas  = 'r'
color_input = 'k'
ms, mfc     = 3, 'none'
figsize_meas   = (9, 5)
figsize_fit    = (8, 3)
figsize_parity = (5, 5)
scut = 6 * 600                 # index limit for time-series display

plot_c = ['cEe', 'cEa', 'cA', 'cW', 'cV', 'cO', 'cC']
plot_u = ['alpha', 'beta', 'NO', 'NEe', 'NA', 'T2']

parity_cases = ['train_rv', 'train_rc', 'val_rv', 'val_rc']
parity_data_keys = {
    'train_rv': ('rtv_meas', 'rnv_meas'),
    'train_rc': ('rtc_meas', 'rnc_meas'),
    'val_rv':   ('rtv_val',  'rnv_val'),
    'val_rc':   ('rtc_val',  'rnc_val'),
}
parity_labels = {
    'train_rv': r'$r_{t,1}$ (train)',
    'train_rc': r'$r_{t,2}$ (train)',
    'val_rv':   r'$r_{v,1}$ (val)',
    'val_rc':   r'$r_{v,2}$ (val)',
}
# multiply raw pickle values by this scale before plotting
parity_scale = {'train_rv': 385., 'train_rc': 385.,
                'val_rv':   385., 'val_rc':   385.}

# measurements are red, control inputs are black, and each model gets its own
# color from this palette (assigned in main() once the model types are known).
_MODEL_PALETTE = ['blue', 'green', 'darkorange', 'purple', 'brown', 'teal', 'crimson']
file_base = os.path.basename(__file__).replace('.py', '')


# ── model-type decoding ─────────────────────────────────────────────────────────
# A flattened model type is  {mbase}_{cheat_tag}_{noise_tag}[_direct], e.g.
#   nn_param_noise_cheat_nonoise_direct  or  nn_param_noise_nocheat_noise
# matching the collector's nested layout (cheat_tag -> noise_tag -> kind).
# The four cases (besides the true plant / param baselines) decode to the labels
#   rmeas, cmeas_rinit, cmeas, cmeas_noise.
CHEAT_TAGS = ('nocheat', 'cheat')   # longer tag first ('cheat' is a suffix of it)
NOISE_TAGS = ('nonoise', 'noise')   # longer tag first ('noise' is a suffix of it)


def parse_nt(nt):
    """Decompose a flattened model type into (mbase, cheat_tag, noise_tag, kind).
    Tags absent from the name come back as '' (kind defaults to 'main')."""
    s = nt
    kind = 'main'
    if s.endswith('_direct'):
        kind, s = 'direct', s[:-len('_direct')]
    noise_tag = ''
    for tag in NOISE_TAGS:
        if s.endswith('_' + tag):
            noise_tag, s = tag, s[:-len('_' + tag)]
            break
    cheat_tag = ''
    for tag in CHEAT_TAGS:
        if s.endswith('_' + tag):
            cheat_tag, s = tag, s[:-len('_' + tag)]
            break
    return s, cheat_tag, noise_tag, kind


def base_case_name(cheat_tag, kind):
    """Map the (cheat_tag, kind) pair to its base case label (no noise tag).
    cheat+direct -> rmeas (fit to rate meas), cheat+main -> cmeas_rinit
    (rate-pretrained then ODE-fit), nocheat -> cmeas (fit to conc meas only)."""
    if cheat_tag == 'cheat' and kind == 'direct':
        return 'rmeas'
    if cheat_tag == 'cheat':
        return 'cmeas_rinit'
    if cheat_tag == 'nocheat':
        return 'cmeas'
    return ''


def display_name(nt):
    """Friendly label: the base case name, with '_noise' appended for the noisy
    ablation, e.g. rmeas, cmeas_rinit, cmeas, cmeas_noise."""
    mbase, cheat_tag, noise_tag, kind = parse_nt(nt)
    base = base_case_name(cheat_tag, kind) or nt
    return f'{base}_noise' if noise_tag == 'noise' else base


def model_base(nt):
    """Strip the cheat_tag/noise_tag/kind suffixes from a model type, leaving the
    producing script's base (e.g. nn_param_noise vs nn_param_noise_reduced) so
    the two can be told apart in the page titles."""
    return parse_nt(nt)[0]


# canonical display order: cheat-ablation case first (rmeas, cmeas_rinit, cmeas),
# then the noise ablation (nonoise before noise), so the four cases come out as
# rmeas, cmeas_rinit, cmeas, cmeas_noise everywhere model types are iterated.
_CASE_ORDER  = {'rmeas': 0, 'cmeas_rinit': 1, 'cmeas': 2}
_NOISE_ORDER = {'nonoise': 0, 'noise': 1}


def group_sort_key(nt, pc=''):
    """Sort key ordering model types by canonical case, then noise tag, then
    plant case."""
    mbase, cheat_tag, noise_tag, kind = parse_nt(nt)
    return (_CASE_ORDER.get(base_case_name(cheat_tag, kind), 99),
            _NOISE_ORDER.get(noise_tag, 99), pc, nt)


# ── CI helper ─────────────────────────────────────────────────────────────────
def ci_band(arrays):
    """Stack list of equal-shaped arrays; return (mean, lo, hi) over axis 0."""
    arr = np.array(arrays, dtype=float)
    return (arr.mean(axis=0),
            np.percentile(arr, CI_LO, axis=0),
            np.percentile(arr, CI_HI, axis=0))


# ── pages ─────────────────────────────────────────────────────────────────────
def plot_measurement_page(pdf, data_meas, plant_case):
    """Measurement snapshot."""
    fig = plt.figure(figsize=figsize_meas)
    days = data_meas[plant_case]['time'][-1] / 3600 / 24
    if title:
        fig.suptitle(f'Plant: {plant_case} — measurement snapshot (total {days:.1f} days)')
    for i, key in enumerate(plot_c + plot_u):
        ax = fig.add_subplot(3, 5, i + 1)
        color = color_input if key in plot_u else color_meas
        ax.plot(data_meas[plant_case]['time'][:scut] / 3600,
                data_meas[plant_case][key][:scut], color, ms=ms, mfc=mfc)
        if key in plot_c + plot_u:
            ax.axhline(y=data_meas[plant_case]['opti_conc'][key],
                       color='gray', ls='--', lw=0.5)
        ylabel_key = 'T2' if key == 'T_end' else key
        ax.set_ylabel(get_label(ylabel_key, labels))
        if i > 7:
            ax.set_xlabel(r'$t$ (hr)')
    plt.tight_layout()
    pdf.savefig(fig)
    plt.close(fig)


def plot_fit_page(pdf, data_meas, plant_case, mc_data_list, pred_key, page_title,
                  color, val_index=None):
    """Time-series fit (measurements in red, model in its own `color`).

    With multiple MC runs: shaded CI band + mean line across runs.
    With a single run (mc_iter 0 only): a plain line.

    val_index: if given, measurements are sliced [val_index : val_index+scut]
               so the validation window lines up with val_pred (which starts at 0).
    """
    preds = [np.array(d[pred_key]) for d in mc_data_list if pred_key in d]
    if not preds:
        return
    n_mc    = len(preds)
    min_len = min(p.shape[0] for p in preds)

    t_meas = data_meas[plant_case]['time'][:scut] / 3600   # relative x-axis
    n      = min(scut, min_len)
    t_pred = t_meas[:n]
    vi     = val_index if val_index is not None else 0

    if n_mc > 1:
        mean, lo, hi = ci_band([p[:min_len] for p in preds])   # (min_len, n_species)

    fig = plt.figure(figsize=figsize_fit)
    if title:
        fig.suptitle(page_title)
    for i, key in enumerate(plot_c):
        ax = fig.add_subplot(2, 4, i + 1)
        ax.plot(t_meas, data_meas[plant_case][key][vi:vi + scut],
                color_meas, ms=ms, mfc=mfc, lw=0.8, label='meas')
        if i == 7:   # T_end: no model output, just replot measurement
            ax.plot(t_pred, data_meas[plant_case]['T_end'][vi:vi + n],
                    color, lw=1, label='model')
        elif n_mc > 1:
            ax.fill_between(t_pred, lo[:n, i], hi[:n, i],
                            color=color, alpha=0.25,
                            label=f'{CI_LO}–{CI_HI}% CI')
            ax.plot(t_pred, mean[:n, i], color=color, lw=1.2, label='mean')
        else:
            ax.plot(t_pred, preds[0][:n, i], color=color, lw=1.2, label='fit')
        ax.set_ylabel(get_label(key, labels))
        if i >= 4:
            ax.set_xlabel(r'$t$ (hr)')
        if i == 0:
            ax.legend(fontsize=7)
    plt.tight_layout()
    pdf.savefig(fig)
    plt.close(fig)


def plot_parity_page(pdf, mc_data_list, color, page_title):
    """Parity plots in the model's own `color`.

    With multiple MC runs: shaded CI band + mean line (true rates averaged over
    MCs, NN rates sorted by that mean and banded over the MC dimension).
    With a single run: rasterized scatter of the raw points.
    """
    fig = plt.figure(figsize=figsize_parity)
    if title:
        fig.suptitle(page_title)
    for i, key in enumerate(parity_cases):
        xk, yk = parity_data_keys[key]
        scale  = parity_scale[key]
        x_all, y_all = [], []
        for d in mc_data_list:
            if xk not in d or yk not in d:
                continue
            x_all.append(np.array(d[xk]) * scale)
            y_all.append(np.array(d[yk]) * scale)
        if not x_all:
            continue
        min_len = min(len(x) for x in x_all)
        x_all   = np.array([x[:min_len] for x in x_all])  # (n_mc, n_pts)
        y_all   = np.array([y[:min_len] for y in y_all])  # (n_mc, n_pts)

        # sort points by mean true rate so fill_between is contiguous
        x_mean = x_all.mean(axis=0)
        idx    = np.argsort(x_mean)
        xs     = x_mean[idx]
        ys     = y_all[:, idx]                              # (n_mc, n_pts)

        ax = fig.add_subplot(2, 2, i + 1)
        if x_all.shape[0] == 1:
            # single MC run: no band/mean meaningful — rasterized scatter
            ax.scatter(xs, ys[0], color=color, s=4, alpha=0.5, rasterized=True)
            lims = [min(xs.min(), ys.min()), max(xs.max(), ys.max())]
        else:
            y_mean = ys.mean(axis=0)
            y_lo   = np.percentile(ys, CI_LO, axis=0)
            y_hi   = np.percentile(ys, CI_HI, axis=0)
            ax.fill_between(xs, y_lo, y_hi, color=color, alpha=0.3)
            ax.scatter(xs, y_mean, color=color, s=5, alpha=0.6, rasterized=True)
            lims = [min(xs.min(), y_lo.min()), max(xs.max(), y_hi.max())]
        ax.plot(lims, lims, 'k--', lw=1)
        ax.set_xlim(lims)
        ax.set_ylim(lims)
        ax.set_aspect('equal', adjustable='box')
        ax.set_title(parity_labels[key])
        if i >= 2:   # bottom row only; the top row shares the same x-axis
            ax.set_xlabel('True')
        if i % 2 == 0:   # left column only; the right column shares the y-axis
            ax.set_ylabel('NN')
    plt.tight_layout()
    pdf.savefig(fig)
    plt.close(fig)


def plot_val_parity_summary_page(pdf, plant_case, groups, model_color):
    """One page comparing the validation parity of every model type (case) for a
    plant: a 2 x N grid where each column is a model type, row 1 is the r_1
    validation parity (val_rv) and row 2 is r_2 (val_rc).

    `groups` is a list of (model_type, mc_data_list) tuples (the columns).
    The drawing logic mirrors plot_parity_page's per-panel rendering.
    """
    val_rows = [('val_rv', r'$r_{1}$'), ('val_rc', r'$r_{2}$')]
    n_cols   = len(groups)
    fig = plt.figure(figsize=(2 * n_cols, 5))
    if title:
        fig.suptitle(f'Validation parity comparison — plant: {plant_case}')
    for col, (nt, mc_data_list) in enumerate(groups):
        color = model_color[nt]
        for row, (case_key, rlabel) in enumerate(val_rows):
            ax = fig.add_subplot(2, n_cols, row * n_cols + col + 1)
            xk, yk = parity_data_keys[case_key]
            scale  = parity_scale[case_key]
            x_all, y_all = [], []
            for d in mc_data_list:
                if xk not in d or yk not in d:
                    continue
                x_all.append(np.array(d[xk]) * scale)
                y_all.append(np.array(d[yk]) * scale)
            if row == 0:   # column identifier on the top-row panels only
                ax.set_title(f'({chr(ord("a") + col)})')
            if not x_all:
                continue
            min_len = min(len(x) for x in x_all)
            x_all   = np.array([x[:min_len] for x in x_all])  # (n_mc, n_pts)
            y_all   = np.array([y[:min_len] for y in y_all])  # (n_mc, n_pts)

            # sort points by mean true rate so fill_between is contiguous
            x_mean = x_all.mean(axis=0)
            idx    = np.argsort(x_mean)
            xs     = x_mean[idx]
            ys     = y_all[:, idx]                              # (n_mc, n_pts)

            if x_all.shape[0] == 1:
                # single MC run: no band/mean meaningful — rasterized scatter
                ax.scatter(xs, ys[0], color=color, s=4, alpha=0.5,
                           rasterized=True)
                lims = [min(xs.min(), ys.min()), max(xs.max(), ys.max())]
            else:
                y_mean = ys.mean(axis=0)
                y_lo   = np.percentile(ys, CI_LO, axis=0)
                y_hi   = np.percentile(ys, CI_HI, axis=0)
                ax.fill_between(xs, y_lo, y_hi, color=color, alpha=0.3)
                ax.scatter(xs, y_mean, color=color, s=3, alpha=0.6, rasterized=True)
                lims = [min(xs.min(), y_lo.min()), max(xs.max(), y_hi.max())]
            ax.plot(lims, lims, 'k--', lw=1)
            ax.set_xlim(lims)
            ax.set_ylim(lims)
            ax.set_aspect('equal', adjustable='box')
            if row == 1:   # bottom row only; the top row shares the same x-axis
                ax.set_xlabel('Truth')
            if col == 0:   # first column: y-axis label + the r_1 / r_2 row text
                ax.set_ylabel('Prediction')
                ax.text(0.06, 0.94, rlabel, transform=ax.transAxes,
                        va='top', ha='left')
    plt.tight_layout()
    pdf.savefig(fig)
    plt.close(fig)


def plot_val_fit_overlay_page(pdf, data_meas, plant_case, groups, model_color):
    """One page: validation time-series fits as a (n_case x n_species) grid — one
    row per model type (case), one column per measured species (measurements in
    red, the case in its own color, mean + light CI band for multi-MC groups).
    Each row is titled by its case name (in the case's color) on the leftmost
    panel; each panel is tagged with its species c_j (no unit) at the top-left.

    `groups` is a list of (model_type, mc_data_list) tuples, in canonical order.
    """
    t_meas = data_meas[plant_case]['time'][:scut] / 3600
    vi     = data_meas[plant_case]['val_index']
    n_rows = len(groups)
    n_cols = len(plot_c)
    fig = plt.figure(figsize=(0.62 * 2 * n_cols, 0.8 * 1.5 * n_rows))
    if title:
        fig.suptitle(f'Validation fit comparison — plant: {plant_case}')
    for r, (nt, mc_data_list) in enumerate(groups):
        color = model_color[nt]
        preds = [np.array(d['val_pred']) for d in mc_data_list if 'val_pred' in d]
        if preds:
            min_len = min(p.shape[0] for p in preds)
            n       = min(scut, min_len)
            t_pred  = t_meas[:n]
            mean, lo, hi = (ci_band([p[:min_len] for p in preds])
                            if len(preds) > 1 else (preds[0], None, None))
        for c, key in enumerate(plot_c):
            ax = fig.add_subplot(n_rows, n_cols, r * n_cols + c + 1)
            if r != n_rows - 1:
                meas = data_meas[plant_case][key][vi:vi + scut]
            else:
                meas = d['val'][:n, c] +  d['val'][:n, c] * 1e-2 * np.random.normal(size=d['val'][:n, c].shape) 
            ax.plot(t_meas, meas,color_meas, lw=0.8)
            if preds:
                if lo is not None:
                    ax.fill_between(t_pred, lo[:n, c], hi[:n, c],
                                    color=color, alpha=0.15)
                ax.plot(t_pred, mean[:n, c], color=color, lw=1.0)
            # species tag (concentration symbol, no unit), just outside the
            # top-left of the plot area
            if r == 0:
                ax.text(0.02, 1.04, rf'$c_{{{key[1:]}}}$', transform=ax.transAxes,
                        va='bottom', ha='left', fontsize=8)
            if r == n_rows - 1:   # x-label on the bottom row only
                ax.set_xlabel(r'$t$ (hr)')
            if c == 0:            # row label (a),(b),(c),(d) in the case's color
                ax.text(-0.32, 1.04, f'({chr(ord("a") + r)})',
                        transform=ax.transAxes, va='bottom', ha='left',
                        color='k')
    plt.tight_layout()
    pdf.savefig(fig)
    plt.close(fig)


# ── main ──────────────────────────────────────────────────────────────────────
data_meas = PickleTool.load('VAc_meas.pickle', 'read')

# single consolidated pickle produced by VAc_collect_param_fit_nn.py:
#   merged[base][case_type][mc_iter][cheat_tag][noise_tag][kind]
merged = PickleTool.load('VAc_collect_param_fit_nn.pickle', 'read')

# Flatten the nested pickle into groups keyed by (model_type, plant_case), each
# holding an mc-ordered list of result dicts. A group is one
# (base, cheat_tag, noise_tag, kind) combination, encoded into the model_type
# string so display_name()/group_sort_key() and the page functions can decode it.
mc_maps = defaultdict(dict)   # (nt, plant_case) -> {mc_iter: data}
for base, cases in merged.items():
    mbase = base.replace('VAc_param_fit_', '')   # e.g. nn_param_noise
    for plant_case, iters in cases.items():
        for mc_iter, cheat_tags in iters.items():
            for cheat_tag, noise_tags in cheat_tags.items():
                for noise_tag, kinds in noise_tags.items():
                    for kind, d in kinds.items():
                        suffix = (f'_{cheat_tag}_{noise_tag}'
                                  + ('_direct' if kind == 'direct' else ''))
                        nt = mbase + suffix
                        mc_maps[(nt, plant_case)][mc_iter] = d

group_data = {key: [mc_map[i] for i in sorted(mc_map)]
              for key, mc_map in mc_maps.items()}

# plant cases discovered, kept only if measurement data exists
plant_cases = sorted({pc for _, pc in group_data})
for pc in [pc for pc in plant_cases if pc not in data_meas]:
    print(f'  WARNING: no measurement data for plant {pc!r} in VAc_meas.pickle — skipping')
plant_cases = [pc for pc in plant_cases if pc in data_meas]
group_data  = {(nt, pc): v for (nt, pc), v in group_data.items() if pc in plant_cases}
print(f'Plant cases: {plant_cases}')
print(f'Found {len(group_data)} groups across {len(plant_cases)} plant case(s).')

# assign a distinct color per model type (measurements stay red, inputs black)
model_types = sorted({nt for nt, _ in group_data})
model_color = {nt: _MODEL_PALETTE[i % len(_MODEL_PALETTE)]
               for i, nt in enumerate(model_types)}

with PdfPages(f'{file_base}.pdf') as pdf:

    # 1. Measurement snapshot (one page per plant)
    for plant_case in plant_cases:
        plot_measurement_page(pdf, data_meas, plant_case)

    # 2. Per-group: training fit, validation fit, parity
    for (nt, pc), mc_data_list in sorted(
            group_data.items(), key=lambda kv: group_sort_key(*kv[0])):
        n_mc       = len(mc_data_list)
        color      = model_color[nt]
        base_title = f'{display_name(nt)} [{model_base(nt)}] | plant: {pc} | n={n_mc} MC'
        print(f'  Plotting {base_title}')

        plot_fit_page(pdf, data_meas, pc, mc_data_list, 'meas_pred',
                      f'Training fit — {base_title}', color)
        plot_fit_page(pdf, data_meas, pc, mc_data_list, 'val_pred',
                      f'Validation fit — {base_title}', color,
                      val_index=data_meas[pc]['val_index'])
        plot_parity_page(pdf, mc_data_list, color,
                         f'Parity plots — {base_title}')

    # 3. Per-plant validation-parity comparison: 2 x N grid with one column per
    #    model type (case), row 1 = r_1 (val_rv), row 2 = r_2 (val_rc).
    #    Columns follow the canonical case order: rmeas, cmeas_rinit, cmeas.
    for plant_case in plant_cases:
        groups = sorted([(nt, mc_data_list)
                         for (nt, pc), mc_data_list in group_data.items()
                         if pc == plant_case],
                        key=lambda g: group_sort_key(g[0]))
        if groups:
            plot_val_parity_summary_page(pdf, plant_case, groups, model_color)

    # 4. Per-plant validation-fit comparison: every case overlaid on shared axes,
    #    one subplot per measured species. Cases follow the canonical order.
    for plant_case in plant_cases:
        groups = sorted([(nt, mc_data_list)
                         for (nt, pc), mc_data_list in group_data.items()
                         if pc == plant_case],
                        key=lambda g: group_sort_key(g[0]))
        if groups:
            plot_val_fit_overlay_page(pdf, data_meas, plant_case, groups, model_color)

# print the rmse_meas, rmse_val, training time sample averaged along with +- 1.96 std over the MC runs
def mc_mean_ci(mc_data_list, key, scale=1.0):
    """Mean and 1.96*std (sample std) of a scalar `key` over the MC runs in a
    group, or None if no run reports it (e.g. training_time on the direct fit).
    `scale` multiplies the values (e.g. 1/60 to convert minutes to hours)."""
    vals = [float(d[key]) * scale for d in mc_data_list
            if key in d and d[key] is not None and not np.isnan(float(d[key]))]
    if not vals:
        return None
    a = np.asarray(vals)
    return a.mean(), 1.96 * (a.std(ddof=1) if len(a) > 1 else 0.0)


def _fmt(stat):
    return f'{stat[0]:.4g} +/- {stat[1]:.4g}' if stat else 'n/a'


print('\nMC summary (mean +/- 1.96 std over MC runs):')
print(f'  {"case":16s} {"plant":8s} {"n":>3s}  {"rmse_meas":>20s}  {"rmse_val":>20s}  {"train_time (hrs)":>20s}')
for (nt, pc), mc_data_list in sorted(group_data.items(), key=lambda kv: group_sort_key(*kv[0])):
    print(f'  {display_name(nt):16s} {pc:8s} {len(mc_data_list):3d}  '
          f'{_fmt(mc_mean_ci(mc_data_list, "rmse_meas")):>20s}  '
          f'{_fmt(mc_mean_ci(mc_data_list, "rmse_val")):>20s}  '
          f'{_fmt(mc_mean_ci(mc_data_list, "training_time", scale=1/60)):>20s}')