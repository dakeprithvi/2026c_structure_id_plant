# [depends] VAc_hist.pickle
# [depends] VAc_meas.pickle
# [depends] VAc_perturb_plant_2d.pickle
# [depends] VAc_perturb_bbox_plant_2d.pickle
#
# Profit-landscape contours over two swept control inputs (from VAc_hist.py).
#
# VAc_hist.py now stores, for the analytic 'test' case, a single ground-truth
# surface, and for every MODEL case a dict {mc: surface} spanning all id MC
# iterations.  Each surface is that model's OWN objective (Plant.obj) over the
# (axis1, axis2) grid — its optimisation landscape.  We contour each landscape,
# mark the true optimum (black star), and overlay where each multistart seed
# (1/2/3) actually converged in the perturb study.
#
# Layout: one page for the true-plant landscape on its own, then ONE page per id
# MC iteration.  Each MC page is a 2x3 grid of the six model cases — the four
# structured models fill the first two columns (sharing one colour scale) and the
# two black-box models fill the third column (sharing their own colour scale,
# since the bbox objective is on a different numeric scale).  Panels are tagged
# (a)-(f) just outside their top-left corner.  => 1 true-plant page + 20 MC pages.

import sys, os
sys.path.append('lib/')
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.lines import Line2D
from utils import PickleTool

os.makedirs('plots', exist_ok=True)
file_base = os.path.basename(__file__).replace('.py', '')
data = PickleTool.load('VAc_hist.pickle', 'read')
data_meas = PickleTool.load('VAc_meas.pickle', 'read')['test']
oc = data_meas['opti_conc']

# master switch for figure titles (off for paper figures)
title = False

# canonical model order (matching VAc_plot_perturb_plant.py): structured models
# rmeas -> cmeas_rinit -> cmeas -> cmeas_noise fill the first two columns; the two
# black-box models fill the third column.
STRUCT_MODELS = ['rmeas', 'cmeas_rinit', 'cmeas', 'cmeas_noise']
BBOX_MODELS = ['cmeas_bbox', 'cmeas_noise_bbox']
CASE_LABEL = {'test': 'Plant (truth)', 'rmeas': 'rmeas', 'cmeas_rinit': 'rmeas\\_init',
              'cmeas': 'cmeas', 'cmeas_noise': 'cmeas\\_noise',
              'cmeas_bbox': 'bbox', 'cmeas_noise_bbox': 'bbox\\_noise'}

# row-major (a)-(f) placement into the 2x3 grid; the four structured panels sit in
# columns 0-1, the two bbox panels in column 2.
GRID_ORDER = ['rmeas', 'cmeas_rinit', 'cmeas_bbox',
              'cmeas', 'cmeas_noise', 'cmeas_noise_bbox']
PANEL_TAGS = ['(a)', '(b)', '(c)', '(d)', '(e)', '(f)']

# Final summary page: one featured id MC iteration per model case.  Each panel
# shows a different case at a different hand-picked MC (paired here in canonical
# order), laid out in the same 2x3 struct/bbox grid as the per-MC pages.
COLLECTION_MC = {'rmeas': 0, 'cmeas_rinit': 1, 'cmeas': 6, 'cmeas_noise': 11,
                 'cmeas_bbox': 7, 'cmeas_noise_bbox': 19}

# axis display: physical-unit scale + LaTeX label per control (shared with the
# u_scale map in VAc_plot_perturb_plant.py)
AX_SCALE = {'na': 785 / 60, 'ne': 831 / 60, 'no': 831 / 60,
            'alpha': 1, 'beta': 1, 'ta': 150 + 273.15}
AX_LABEL = {'alpha': r'$\alpha$', 'beta': r'$\beta$', 'no': r'$N_\mathrm{O}$',
            'ne': r'$N_\mathrm{Ee}$', 'na': r'$N_\mathrm{A}$', 'ta': r'$T$ (K)'}


def is_nested(case):
    """Model cases are stored as {mc: entry}; 'test' is a single flat entry."""
    return case in data and isinstance(data[case], dict) and 'profit_model' not in data[case]


def entry(case, mc):
    """The surface entry for a case; mc is ignored for the flat 'test' case."""
    return data[case][mc] if is_nested(case) else data[case]


# model cases actually present, and the sorted union of their MC iterations
struct_present = [c for c in STRUCT_MODELS if is_nested(c)]
bbox_present = [c for c in BBOX_MODELS if is_nested(c)]
model_present = struct_present + bbox_present
if 'test' not in data and not model_present:
    sys.exit('VAc_hist.pickle holds no cases')

all_mcs = sorted(set().union(*[set(data[c].keys()) for c in model_present])) if model_present else []

# grid + axis names are common to every case/mc
if 'test' in data:
    ref = data['test']
else:
    c0 = model_present[0]
    ref = data[c0][sorted(data[c0])[0]]
AXIS1, AXIS2 = ref['axis1'], ref['axis2']
s1, s2 = AX_SCALE.get(AXIS1, 1.0), AX_SCALE.get(AXIS2, 1.0)
X = np.asarray(ref['axis1_grid'], dtype=float) * s1
Y = np.asarray(ref['axis2_grid'], dtype=float) * s2
opt1, opt2 = oc[AXIS1] * s1, oc[AXIS2] * s2   # true optimum operating point

# per-seed RTO operating points from the perturb studies, overlaid to show where
# each multistart seed converged (interior peak, box corner, or shallow optimum).
PERT_SRC = {'cmeas': 'VAc_perturb_plant_2d.pickle', 'cmeas_noise': 'VAc_perturb_plant_2d.pickle',
            'rmeas': 'VAc_perturb_plant_2d.pickle', 'cmeas_rinit': 'VAc_perturb_plant_2d.pickle',
            'cmeas_bbox': 'VAc_perturb_bbox_plant_2d.pickle',
            'cmeas_noise_bbox': 'VAc_perturb_bbox_plant_2d.pickle'}
_pert_cache = {}


def _load_pert(fn):
    if fn not in _pert_cache:
        _pert_cache[fn] = PickleTool.load(fn, 'read') if os.path.exists(fn) else None
    return _pert_cache[fn]


def seed_points(case, mc):
    """(x, y, seed) RTO operating points for this case at id MC iteration `mc`,
    from the perturb pickle — the actual multistart optima behind the study."""
    fn = PERT_SRC.get(case)
    d = _load_pert(fn) if fn else None
    k1, k2 = f'{AXIS1}_model', f'{AXIS2}_model'
    if d is None or case not in d or mc is None or k1 not in d[case] or k2 not in d[case]:
        return []
    idm = np.asarray(d[case]['id_mc']).ravel()
    sd = np.asarray(d[case]['seed']).ravel()
    xv = np.asarray(d[case][k1], float).ravel() * s1
    yv = np.asarray(d[case][k2], float).ravel() * s2
    sel = idm == mc
    return list(zip(xv[sel], yv[sel], sd[sel]))


def surf(case, mc=None):
    return np.ma.masked_invalid(np.asarray(entry(case, mc)['profit_model'], dtype=float))


NLEVELS = 30


def family_levels(surfaces):
    """Shared (vmin, vmax, levels) across a list of masked surfaces."""
    v = np.ma.concatenate([s.ravel() for s in surfaces])
    vmin, vmax = float(v.min()), float(v.max())
    return vmin, vmax, np.linspace(vmin, vmax, NLEVELS)


def draw_landscape(ax, Z, ttl, levels, vmin, vmax, seed_pts=()):
    cf = ax.contourf(X, Y, Z, levels=levels, vmin=vmin, vmax=vmax, cmap='inferno')
    ax.contour(X, Y, Z, levels=levels, colors='k', linewidths=0.3, alpha=0.4)
    ax.plot(opt1, opt2, marker='*', color='black', markersize=7,
            markeredgecolor='k', linestyle='none', zorder=5)   # true optimum
    # where each multistart seed converged — all drawn as the same black dot, so
    # several dots simply mark the several optima (seed identity not encoded).
    for x, y, s in seed_pts:
        ax.plot(x, y, marker='o', color='black', markersize=5,
                markeredgecolor='k', linestyle='none', zorder=6,
                clip_on=False)   # keep box-corner/edge optima fully visible
    if ttl:
        ax.set_title(ttl, fontsize=10)
    return cf


def add_panel_tag(ax, tag):
    """Bold (a)/(b)/... just outside the panel's top-left corner."""
    ax.text(-0.18, 1.06, tag, transform=ax.transAxes, fontsize=11,
            fontweight='bold', va='bottom', ha='left')


def _finish_grid(fig, axes, cbars, legend_ax=None):
    """Shared axis labels, colour bar(s), and legend for a 2x3 struct/bbox page.

    `cbars` is a list of (cf, cbar_axes, (vmin, vmax, levels)) specs — two for the
    per-MC pages (structured block + bbox column), one for a single shared bar.
    `legend_ax`, if given, gets the star/dot legend."""
    for ax in axes:
        if ax.get_subplotspec().is_last_row():
            ax.set_xlabel(AX_LABEL.get(AXIS1, AXIS1))
        if ax.get_subplotspec().is_first_col():
            ax.set_ylabel(AX_LABEL.get(AXIS2, AXIS2))

    for cf, cbar_axes, vv in cbars:
        vmin, vmax, _ = vv
        cb = fig.colorbar(cf, ax=cbar_axes, location='right', shrink=1,
                          ticks=np.linspace(vmin, vmax, 6),
                          aspect=35)
        cb.ax.tick_params(labelsize=8)

    # small legend in one panel: star = plant, circle = model
    if legend_ax is not None:
        handles = [Line2D([0], [0], marker='*', color='black', markeredgecolor='k',
                          linestyle='none', markersize=7, label='plant'),
                   Line2D([0], [0], marker='o', color='black', markeredgecolor='k',
                          linestyle='none', markersize=5, label='model')]
        legend_ax.legend(handles=handles, loc='upper right', frameon=True,
                         fontsize=8, handletextpad=0.3, borderpad=0.4)


def mc_page(pdf, mc):
    """One 2x3 page of the six model landscapes at id MC iteration `mc`."""
    struct_here = [c for c in struct_present if mc in data[c]]
    bbox_here = [c for c in bbox_present if mc in data[c]]
    if not struct_here and not bbox_here:
        return

    vs = family_levels([surf(c, mc) for c in struct_here]) if struct_here else None
    vb = family_levels([surf(c, mc) for c in bbox_here]) if bbox_here else None

    fig, axes = plt.subplots(2, 3, figsize=(8., 5.),
                             sharex=True, sharey=True, squeeze=False,
                             layout='constrained')
    axes = axes.flatten()
    if title:
        fig.suptitle(f'Profit landscapes — identification MC {mc}', fontsize=13)

    cf_struct = cf_bbox = None
    for ax, case, tag in zip(axes, GRID_ORDER, PANEL_TAGS):
        add_panel_tag(ax, tag)
        is_struct = case in struct_present
        vv = vs if is_struct else vb
        if case not in data or mc not in data[case] or vv is None:
            ax.axis('off')
            continue
        vmin, vmax, levels = vv
        pts = seed_points(case, mc)
        if title:
            ttl = CASE_LABEL.get(case, case)
        else:
            ttl = None
        cf = draw_landscape(ax, surf(case, mc), ttl,
                            levels, vmin, vmax, seed_pts=pts)
        if is_struct:
            cf_struct = cf
        else:
            cf_bbox = cf

    # two colour bars: one for the structured block, one for the bbox column
    cbars = []
    if cf_struct is not None:
        cbars.append((cf_struct, [axes[0], axes[1], axes[3], axes[4]], vs))
    if cf_bbox is not None:
        cbars.append((cf_bbox, [axes[2], axes[5]], vb))
    _finish_grid(fig, axes, cbars,
                 legend_ax=axes[5] if cf_bbox is not None else None)
    pdf.savefig(fig)
    plt.close(fig)


def collection_page(pdf):
    """Final 2x3 summary page: each panel a different model case shown at its own
    hand-picked id MC iteration (paired in COLLECTION_MC).  Unlike the per-MC
    pages, all six panels share ONE colour scale — here every case's landscape
    spans roughly the same range, so a single bar keeps them directly comparable
    without washing out per-panel contour detail."""
    def present(c):
        return c in data and COLLECTION_MC.get(c) in data[c]

    here = [c for c in GRID_ORDER if present(c)]
    if not here:
        return

    # one shared colour scale across every panel on the page
    va = family_levels([surf(c, COLLECTION_MC[c]) for c in here])
    vmin, vmax, levels = va

    fig, axes = plt.subplots(2, 3, figsize=(8., 5.),
                             sharex=True, sharey=True, squeeze=False,
                             layout='constrained')
    axes = axes.flatten()
    if title:
        fig.suptitle('Profit landscapes — selected MC per case', fontsize=13)

    cf_any = None
    for ax, case, tag in zip(axes, GRID_ORDER, PANEL_TAGS):
        add_panel_tag(ax, tag)
        mc = COLLECTION_MC.get(case)
        if not present(case):
            ax.axis('off')
            continue
        pts = seed_points(case, mc)
        ttl = f'{CASE_LABEL.get(case, case)} — MC {mc}' if title else None
        cf_any = draw_landscape(ax, surf(case, mc), ttl,
                                levels, vmin, vmax, seed_pts=pts)

    # single colour bar spanning all six panels
    _finish_grid(fig, axes, [(cf_any, list(axes), va)],
                 legend_ax=axes[5] if present(GRID_ORDER[5]) else None)
    pdf.savefig(fig)
    plt.close(fig)


with PdfPages(f'{file_base}.pdf') as pdf:

    # --- Page 1: ground-truth plant landscape on its own ---
    if 'test' in data:
        vmin, vmax, levels = family_levels([surf('test')])
        fig, ax = plt.subplots(figsize=(3, 2.7))
        if title:
            fig.suptitle('Plant profit landscape', fontsize=13)
            ttl = CASE_LABEL['test']
        else:
            ttl = None
        cf = draw_landscape(ax, surf('test'), ttl, levels, vmin, vmax)
        ax.set_xlabel(AX_LABEL.get(AXIS1, AXIS1))
        ax.set_ylabel(AX_LABEL.get(AXIS2, AXIS2))
        fig.colorbar(cf, ax=ax)
        plt.tight_layout()
        pdf.savefig(fig)
        plt.close(fig)

    # --- One page per identification MC iteration ---
    for mc in all_mcs:
        mc_page(pdf, mc)

    # --- Final page: hand-picked MC per case (COLLECTION_MC) ---
    collection_page(pdf)