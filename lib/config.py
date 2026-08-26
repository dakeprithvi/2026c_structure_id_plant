# The file defines a dictionary with keys and labels for a set of variables,
# and provides a function to retrieve the corresponding label for a given key,
# including units where applicable.

keys = ['cEe','cEa','cA','cW','cV', 'cO', 'cC' , 'T_end','alpha', 'beta', 
        'T2', 'P2', 'NO', 'NEe', 'NA', 'time', 'N1f','Q', 'total_econ', 'penalty_term', 'objective']

labels = {
    'Q' : r'$Q$',
    'r1_end' : r'$r_1$',
    'r2_end' : r'$r_2$',
    'total_econ' : '\$',
    'cEe': r'$c_\mathrm{Ee}$',
    'cEa': r'$c_\mathrm{Ea}$',
    'cA': r'$c_\mathrm{A}$',
    'cW': r'$c_\mathrm{W}$',
    'cV': r'$c_\mathrm{V}$',
    'cO': r'$c_\mathrm{O}$',
    'cC': r'$c_{C}$',
    'alpha': r'$\alpha$',
    'beta': r'$\beta$',
    'T2': r'$T$',
    'P2': r'$P$',
    'NO': r'$N_\mathrm{O}^1$',
    'NEe': r'$N_\mathrm{Ee}^1$',
    'NA': r'$N_\mathrm{A}^1$',
    'time': r'$t$',
    'N1f': r'$N_{1f}$',
    'T_end': r'$T$',
    'units_conc': r'$ \ \mathrm{(mol/m^3)}$',
    'units_P': r'$ \ \mathrm{(kPa)}$',
    'units_T': r'$ \ \mathrm{(K)}$',
    'units_N': r'$ \ \mathrm{(mol/s)}$',
    'uses_conc': ['cEe', 'cEa', 'cA', 'cW', 'cV', 'cO', 'cC'],
    'uses_P': ['P2'],
    'uses_T': ['T2', 'T_end'],
    'uses_N': ['NO', 'NEe', 'NA'],
    'Profit': r'(\$/s)',
    'keys': keys
}

def get_label(key, key_labels=labels):
    label = key_labels[key]
    # Check if the key requires units
    if key in key_labels['uses_conc']:
        label = label[:-1] + key_labels['units_conc'][1:]
    elif key in key_labels['uses_P']:
        label = label[:-1] + key_labels['units_P'][1:]
    elif key in key_labels['uses_T']:
        label = label[:-1] + key_labels['units_T'][1:]
    elif key in key_labels['uses_N']:
        label = label[:-1] + key_labels['units_N'][1:]
    return label
