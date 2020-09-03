import matplotlib.pyplot as plt
import numpy as np
import pathlib

def plot_scans_2d(results, parameter_1, parameter_2, label_1, label_2, folder_name):
    print('-> plotting 2d scans')
    folder = pathlib.Path(folder_name)
    if not folder.exists():
        folder.mkdir(parents=True)
    folder.resolve(strict=True)
    _, d_1 = np.linspace(parameter_1.min(), parameter_1.max(), parameter_1.size, endpoint=True, retstep=True)
    _, d_2 = np.linspace(parameter_2.min(), parameter_2.max(), parameter_2.size, endpoint=True, retstep=True)
    midpoint_1 = np.linspace(parameter_1.min() - 0.5 * d_1, parameter_1.max() + 0.5 * d_1, parameter_1.size + 1, endpoint=True)
    midpoint_2 = np.linspace(parameter_2.min() - 0.5 * d_2, parameter_2.max() + 0.5 * d_2, parameter_2.size + 1, endpoint=True)
    plot_infos = [
        ('waist', 'Waist Position (cm)', lambda s, i: 100 * s.z_waist, None)
    ]
    for simple_name, fancy_name in (('end_of_gun', 'Gun End'), ('waist', 'Waist'), ('end_of_drift', 'Drift End')):
        plot_infos += [
            # simple_name, label, lambda
            (f'spot_{simple_name}', '$\\sigma_{\\bot}$ ($\\mu$m)', lambda s, i: 1e6 * s.std_r[getattr(s, i)], 'index_' + simple_name),
            (f'sigz_{simple_name}', '$\\sigma_{z}$ ($\\mu$m)', lambda s, i: 1e6 * s.std_z[getattr(s, i)], 'index_' + simple_name),
            (f'emit4_{simple_name}', '$\\sqrt{\\epsilon_{\mathrm{n}, \mathrm{4D}}}$ (nm)', lambda s, i: 1e9 * np.sqrt(s.emit_xy[getattr(s, i)]), 'index_' + simple_name),
            (f'emit6_{simple_name}', '$\\epsilon_{\mathrm{n}, \mathrm{6D}}$ ($\mathrm{m}^{3}$)', lambda s, i: s.emit_xyz[getattr(s, i)], 'index_' + simple_name),
            (f'espread_{simple_name}', '$\\frac{\\Delta E}{E}$ (%)', lambda s, i: 100 * s.energy_spread[getattr(s, i)], 'index_' + simple_name),
            (f'charge_{simple_name}', 'Charge (fC)', lambda s, i: 1e15 * s.charge[getattr(s, i)], 'index_' + simple_name)
        ]

    for simple_name, label, function, index in plot_infos:
        values = np.array([[function(s, index) for s in x] for x in results])
        plt.pcolormesh(midpoint_1, midpoint_2, values.T)
        plt.xlabel(label_1)
        plt.ylabel(label_2)
        plt.xlim(parameter_1.min(), parameter_1.max())
        plt.ylim(parameter_2.min(), parameter_2.max())
        plt.colorbar(label=label)
        plt.savefig(folder / f'{simple_name}.png', dpi=300)
        plt.clf()
    print('-> done plotting 2d scans')
