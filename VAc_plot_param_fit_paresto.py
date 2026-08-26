# [depends] %LIB%/utils.py
# [depends] VAc_param_fit_paresto.pickle

import sys
sys.path.append('lib/')
import os
import numpy as np
from utils import PickleTool
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages


D = PickleTool.load('VAc_param_fit_paresto.pickle', 'read')

species      = D['species']
time         = D['time']
pcat         = D['pcat']
noise_levels = D['noise_levels']
param_keys   = D['param_keys']
p_true       = D['p_true']
results      = D['results']
base_case    = D['base_case']
fits         = D['fits']

os.makedirs('plots', exist_ok=True)
plot_file_base = os.path.basename(__file__).replace('.py', '')
pdf = PdfPages(f'{plot_file_base}.pdf')

# --- base case: correct parameters, perfect IC ---
meas = base_case['meas']
pred = base_case['pred']
fig = plt.figure(figsize=(12, 6))
for i in range(len(species)):
    plt.subplot(2, 4, i + 1)
    plt.plot(time, meas[:, i], 'ro', markersize=3, label='meas')
    plt.plot(time, pred[:, i], 'b-', label='fit')
    plt.title(species[i])
plt.suptitle('Base case: correct parameters, perfect IC')
plt.tight_layout()
pdf.savefig()
plt.close()

# --- per-fit pages: measurement vs prediction, then rate parity ---
for f in fits:
    nl, ic_kind = f['nl'], f['ic_kind']
    meas_n, pred = f['meas_n'], f['pred']

    fig = plt.figure(figsize=(12, 6))
    for i in range(len(species)):
        plt.subplot(2, 4, i + 1)
        plt.plot(time / 3600, meas_n[:, i], 'ro', markersize=3, label='meas')
        plt.plot(time / 3600, pred[:, i], 'b-', label='fit')
        plt.ylabel(species[i])
        plt.xlabel(r'$t$ (hr)')
    plt.suptitle(f'{ic_kind} IC, noise level = {nl}')
    plt.tight_layout()
    pdf.savefig()
    plt.close()

    # rate parity: fitted-param rate vs true-param rate on the same trajectory
    fig = plt.figure(figsize=(8, 4))
    pairs = [(f['r1_true'], f['r1_fit'], r'$r_1$'),
             (f['r2_true'], f['r2_fit'], r'$r_2$')]
    for j, (rt, rf, name) in enumerate(pairs):
        ax = plt.subplot(1, 2, j + 1)
        ax.scatter(rt * pcat, rf * pcat, s=8, alpha=0.6)
        lo, hi = min(rt.min(), rf.min()), max(rt.max(), rf.max())
        lo, hi = lo * pcat, hi * pcat
        ax.plot([lo, hi], [lo, hi], 'k--', lw=1, label='y=x')
        ax.set_xlabel(f'{name} true')
        ax.set_ylabel(f'{name} fit')
        ax.set_title(name)
        ax.legend()
    plt.suptitle(f'Rate parity: {ic_kind} IC, noise = {nl}')
    plt.tight_layout()
    pdf.savefig()
    plt.close()

# --- summary: bbox/|theta| vs noise level for each parameter ---
n = len(param_keys)
ncols = 3
nrows = int(np.ceil(n / ncols))
fig, axes = plt.subplots(nrows, ncols, figsize=(4 * ncols, 3 * nrows), squeeze=False)
for idx, k in enumerate(param_keys):
    ax = axes[idx // ncols][idx % ncols]
    for ic_kind, marker in (('perfect', 'o-'), ('random', 's--')):
        theta = np.array(results[ic_kind][k]['theta'])
        bbox  = np.array(results[ic_kind][k]['bbox'])
        ax.loglog(noise_levels, bbox / np.abs(theta), marker, label=f'{ic_kind} IC')
    ax.set_xlabel('noise level')
    ax.set_ylabel(f'bbox / |{k}|')
    ax.set_title(k)
    ax.grid(True, which='both', alpha=0.3)
    ax.legend()
for idx in range(n, nrows * ncols):
    axes[idx // ncols][idx % ncols].axis('off')
fig.suptitle('Relative confidence width vs noise level')
plt.tight_layout()
pdf.savefig()
plt.close()

# --- summary: theta (with bbox error bars) vs noise level, per parameter ---
fig, axes = plt.subplots(nrows, ncols, figsize=(4 * ncols, 3 * nrows), squeeze=False)
for idx, k in enumerate(param_keys):
    ax = axes[idx // ncols][idx % ncols]
    for ic_kind, marker in (('perfect', 'o-'), ('random', 's--')):
        theta = np.array(results[ic_kind][k]['theta'])
        bbox  = np.array(results[ic_kind][k]['bbox'])
        ax.errorbar(noise_levels, theta, yerr=bbox, fmt=marker, label=f'{ic_kind} IC', capsize=3)
    ax.axhline(p_true[k], color='k', linestyle=':', label='true')
    ax.set_xscale('log')
    ax.set_xlabel('noise level')
    ax.set_ylabel(k)
    ax.set_title(k)
    ax.grid(True, which='both', alpha=0.3)
    ax.legend()
for idx in range(n, nrows * ncols):
    axes[idx // ncols][idx % ncols].axis('off')
fig.suptitle('Estimated parameter (+/- bbox) vs noise level')
plt.tight_layout()
pdf.savefig()
plt.close()

pdf.close()
