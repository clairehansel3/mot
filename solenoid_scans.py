#!/usr/bin/env python3

import daemon
import itertools
import json
import matplotlib.pyplot as plt
import mot
import multiprocessing
import numpy as np
import os
import pathlib
import pickle
import subprocess
import sys

processes = 25

magnetic_field_strengths = np.linspace(0.2, 0.8, 5, endpoint=True)
solenoid_lengths = np.linspace(0.10, 0.15, 25, endpoint=True)
solenoid_positions = np.linspace(0.1, 0.12, 25, endpoint=True)
scan_parameters = np.array(list(itertools.product(magnetic_field_strengths, solenoid_lengths, solenoid_positions)))

def run_simulation(i):
    print(f'running simulation {i}')
    magnetic_field_strength, solenoid_length, solenoid_position = scan_parameters[i]
    s = mot.MotGunSimulation(f'/a/chansel/mot/solenoid_scans/{i}')
    s.sigma_r = 0.001
    s.sigma_pr = 1.942e-26
    s.mot_z_offset = -0.002
    s.laser_phi_x = 0
    s.laser_phi_y = 0.5 * np.pi / 180
    s.laser_x = 0
    s.laser_y = 0
    s.laser_z = 0
    s.laser_width = 25e-6
    s.peak_density = 5e15
    s.peak_field = 100e6
    s.rf_frequency = 2.85648e9
    s.rf_phase_deg = 90
    s.pyramid_height = 0.0081
    s.gun_length = 0.12
    s.sol_radius = 0.1
    s.sol_strength = magnetic_field_strength
    s.sol_length = solenoid_length
    s.sol_z = solenoid_position
    s.ions_on = True
    s.spacecharge = 'approximate'
    s.initializeParticles()
    s.run()
    s.readData()
    s.analyze()
    s.deleteData()
    return i

class NpEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        else:
            return super(NpEncoder, self).default(obj)

def main():
    #with multiprocessing.Pool(processes) as pool:
    #    results = pool.imap_unordered(run_simulation, range(len(scan_parameters)))
    #    for result in results:
    #        print(f'simulation {result} complete')

    os.system('mkdir -p solenoid_scans.py.results')

    all_scan_results = None
    for i in range(len(scan_parameters)):
        with open(f'/a/chansel/mot/solenoid_scans/{i}/results/data.pickle', 'rb') as f:
            scan_results = pickle.load(f)
            if all_scan_results is None:
                all_scan_results = {key: {key2: [] for key2 in value.keys()} for key, value in scan_results.items()}
            for key, value in scan_results.items():
                for key2, value2 in value.items():
                    all_scan_results[key][key2].append(value2[0])
            with open(f'/a/chansel/mot/solenoid_scans/{i}/results/data.json', 'w+') as f:
                json.dump(scan_results, f, indent=4, cls=NpEncoder)

    shape = (len(magnetic_field_strengths), len(solenoid_lengths), len(solenoid_positions))
    sol_strength = np.array(all_scan_results['parameters']['sol_strength']).reshape(shape)
    sol_length = np.array(all_scan_results['parameters']['sol_length']).reshape(shape)
    sol_z = np.array(all_scan_results['parameters']['sol_z']).reshape(shape)
    time = np.array(all_scan_results['computed_values']['time']).reshape(shape)
    x_centroid = np.array(all_scan_results['computed_values']['x_centroid']).reshape(shape)
    y_centroid = np.array(all_scan_results['computed_values']['y_centroid']).reshape(shape)
    z_centroid = np.array(all_scan_results['computed_values']['z_centroid']).reshape(shape)
    x_std = np.array(all_scan_results['computed_values']['x_std']).reshape(shape)
    y_std = np.array(all_scan_results['computed_values']['y_std']).reshape(shape)
    z_std = np.array(all_scan_results['computed_values']['z_std']).reshape(shape)
    x_emit = np.array(all_scan_results['computed_values']['x_emit']).reshape(shape)
    y_emit = np.array(all_scan_results['computed_values']['y_emit']).reshape(shape)
    emit_6d = np.array(all_scan_results['computed_values']['emit_6d']).reshape(shape)
    charge = np.array(all_scan_results['computed_values']['charge']).reshape(shape)

    _, d1 = np.linspace(solenoid_lengths.min(), solenoid_lengths.max(), solenoid_lengths.size, endpoint=True, retstep=True)
    assert (_ == solenoid_lengths).all()
    _, d2 = np.linspace(solenoid_positions.min(), solenoid_positions.max(), solenoid_positions.size, endpoint=True, retstep=True)
    assert (_ == solenoid_positions).all()

    solenoid_lengths_midpoint = np.linspace(solenoid_lengths.min() - 0.5 * d1, solenoid_lengths.max() + 0.5 * d1, solenoid_lengths.size + 1, endpoint=True)
    solenoid_positions_midpoint = np.linspace(solenoid_positions.min() - 0.5 * d2, solenoid_positions.max() + 0.5 * d2, solenoid_positions.size + 1, endpoint=True)

    plot_infos = [
        ('Peak Solenoid Magnetic Field (T)', sol_strength, 1, 'sol_strength'),
        ('Solenoid Length (cm)', sol_length, 100, 'sol_length'),
        ('Solenoid Position (cm)', sol_z, 100, 'sol_z'),
        ('Time at Exit (ps)', time, 1e12, 'time'),
        ('$\\overline{x}$ ($\\mu$m)', x_centroid, 1e6, 'x_centroid'),
        ('$\\overline{y}$ ($\\mu$m)', y_centroid, 1e6, 'y_centroid'),
        ('$\\overline{z}$ (cm)', z_centroid, 100, 'x_centroid'),
        ('$\\sigma_x$ ($\\mu$m)', x_std, 1e6, 'x_std'),
        ('$\\sigma_y$ ($\\mu$m)', y_std, 1e6, 'y_std'),
        ('$\\sigma_z$ (mm)', z_std, 1e3, 'z_std'),
        ('$\\epsilon_x$ (nm)', x_emit, 1e9, 'x_emit'),
        ('$\\epsilon_y$ (nm)', y_emit, 1e9, 'y_emit'),
        ('$\\epsilon_{\\mathrm{6D}}$ (m$^3$)', emit_6d, 1, 'emit_6d'),
        ('$Q$ (fC)', charge, 1e15, 'charge')
    ]

    for i, magnetic_field_strength in enumerate(magnetic_field_strengths):
        for plot_info in plot_infos:
            plt.title('$B_{\\mathrm{max}} = ' + '{:.2f}'.format(magnetic_field_strength) + '$ T')
            plt.pcolormesh(solenoid_lengths_midpoint * 100, solenoid_positions_midpoint * 100, plot_info[2] * plot_info[1][i, :, :].T, vmin=(plot_info[2] * plot_info[1].min()), vmax=(plot_info[2] * plot_info[1].max()))
            plt.xlabel('Solenoid Length (cm)')
            plt.ylabel('Solenoid Position (cm)')
            plt.colorbar(label=plot_info[0])
            plt.savefig(f'solenoid_scans.py.results/{plot_info[3]}_{i}.png', dpi=300)
            plt.clf()

    os.system('tar -czvf solenoid_scans.py.results.tgz solenoid_scans.py.results')

if len(sys.argv) >= 2 and sys.argv[1] == '--no-detach':
    main()
else:
    if os.path.isfile(f'{__file__}.kill'):
        raise Exception(f'{__file__} is already running!')

    with open(f'{__file__}.log', 'w+') as f:
        with daemon.DaemonContext(
                stdout=f,
                stderr=f,
                working_directory=pathlib.Path(__file__).parent,
            ):
            with open(f'{__file__}.kill', 'w+') as f:
                f.write(f'#!/bin/bash\nkill -SIGINT {os.getpid()}\n')
            subprocess.run(['chmod', '+x', f'{__file__}.kill'], check=True)
            try:
                main()
                pathlib.Path(f'{__file__}.kill').unlink()
            except:
                pathlib.Path(f'{__file__}.kill').unlink()
                raise
