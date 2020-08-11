#!/usr/bin/env python3

import json
import matplotlib.pyplot as plt
import mot
import numpy as np
import os
import pathlib
import pickle

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

IOs = [True, False]
SCs = ['none', 'exact', 'approximate']

def do_scan(i, path, IO, SC):
    print(f'running simulation {i}')
    s = mot.MotGunSimulation(path)
    s.peak_density = 5e15
    s.sigma_r = 0.001
    s.sigma_pr = 1.9424e-26
    s.mot_z_offset = -0.002
    s.laser_phi_x = 0
    s.laser_phi_y = 0.5 * np.pi / 180
    s.laser_x = 0
    s.laser_y = 0
    s.laser_z = 0
    s.laser_width = 25e-6
    s.ions_on = IO
    s.peak_field = 100e6
    s.rf_frequency = 2.85648e9
    s.rf_phase_deg = 90
    s.pyramid_height = 0.0081
    s.sol_radius = 0.1
    s.sol_strength = 0.8
    s.sol_length = 0.05
    s.sol_z = 0.10
    s.spacecharge = SC
    s.tmax = 1.6e-9
    s.tstep = 1e-11
    s.initializeParticles(seed=12342398)
    s.run()
    s.readData()
    analyze(s.data, path, 0.12, 0.40, s.parametersDict())
    return i

def analyze(data, path, end_of_gun, end_of_drift, parameters_dict):
    results_folder = path / 'results'
    if not results_folder.exists():
        results_folder.mkdir(parents=True)
    results_folder.resolve(strict=True)
    ts = np.array([time_block.time for time_block in data.time_blocks])
    x_centroid = np.array([time_block.electrons['x'].mean() for time_block in data.time_blocks])
    y_centroid = np.array([time_block.electrons['y'].mean() for time_block in data.time_blocks])
    z_centroid = np.array([time_block.electrons['z'].mean() for time_block in data.time_blocks])
    x_std = np.array([time_block.electrons['x'].std() for time_block in data.time_blocks])
    y_std = np.array([time_block.electrons['y'].std() for time_block in data.time_blocks])
    r_std = np.sqrt(x_std ** 2 + y_std ** 2)
    z_std = np.array([time_block.electrons['z'].std() for time_block in data.time_blocks])
    beta_z = np.array([time_block.electrons['Bz'] for time_block in data.time_blocks])
    x_emit = np.array([np.sqrt(np.linalg.det(np.cov(np.array([
        time_block.electrons['x'],
        time_block.electrons['G'] * time_block.electrons['Bx']
    ])))) for time_block in data.time_blocks])
    y_emit = np.array([np.sqrt(np.linalg.det(np.cov(np.array([
        time_block.electrons['y'],
        time_block.electrons['G'] * time_block.electrons['By']
    ])))) for time_block in data.time_blocks])
    emit_6d = np.array([np.sqrt(np.linalg.det(np.cov(np.array([
        time_block.electrons['x'],
        time_block.electrons['y'],
        time_block.electrons['z'],
        time_block.electrons['G'] * time_block.electrons['Bx'],
        time_block.electrons['G'] * time_block.electrons['By'],
        time_block.electrons['G'] * time_block.electrons['Bz'],
    ])))) for time_block in data.time_blocks])
    emit_4d = np.array([np.sqrt(np.linalg.det(np.cov(np.array([
        time_block.electrons['x'],
        time_block.electrons['y'],
        time_block.electrons['G'] * time_block.electrons['Bx'],
        time_block.electrons['G'] * time_block.electrons['By'],
    ])))) for time_block in data.time_blocks])
    brightness_6d = np.array([len(time_block.electrons['x']) / emit_6d[i] for i, time_block in enumerate(data.time_blocks)])
    x_emit_nn = np.array([np.sqrt(np.linalg.det(np.cov(np.array([
        time_block.electrons['x'],
        time_block.electrons['G'] * time_block.electrons['Bx']
    ])))) for time_block in data.time_blocks])
    y_emit_nn = np.array([np.sqrt(np.linalg.det(np.cov(np.array([
        time_block.electrons['y'],
        time_block.electrons['G'] * time_block.electrons['By']
    ])))) for time_block in data.time_blocks])
    charge = np.array([-1.60217662e-19 * len(time_block.electrons['x']) for time_block in data.time_blocks])
    brightness_4d = -299792458 * charge / (z_std * np.pi * np.pi * x_emit_nn * y_emit_nn)
    end_of_gun_index = np.argmin(np.abs(z_centroid - end_of_gun))
    end_of_drift_index = np.argmin(np.abs(z_centroid - end_of_drift))
    waist_index = end_of_gun_index + np.argmin(r_std[end_of_gun_index:end_of_drift_index])
    waist = z_centroid[waist_index]

    with open(results_folder / 'data.pickle', 'wb') as f:
        def get_dict(index, name):
            return {
                f'time_index_{name}': (index, ''),
                f'time_{name}': (ts[index], 's'),
                f'x_centroid_{name}': (x_centroid[index], 'm'),
                f'y_centroid_{name}': (y_centroid[index], 'm'),
                f'z_centroid_{name}': (z_centroid[index], 'm'),
                f'x_std_{name}': (x_std[index], 'm'),
                f'y_std_{name}': (y_std[index], 'm'),
                f'z_std_{name}': (z_std[index], 'm'),
                f'r_std_{name}': (r_std[index], 'm'),
                f'x_emit_{name}': (x_emit[index], 'm'),
                f'y_emit_{name}': (y_emit[index], 'm'),
                f'emit_6d_{name}': (emit_6d[index], 'm^3'),
                f'emit_4d_{name}': (emit_4d[index], 'm^2'),
                f'brightness_4d_{name}': (brightness_4d[index], 'A/m^2'),
                f'brightness_6d_{name}': (brightness_6d[index], 'm^-3'),
                f'charge_{name}': (charge[index], 'C')
            }

        data_dictionary = {
            'parameters': parameters_dict,
            'computed_values': {
                **get_dict(end_of_gun_index, 'end_of_gun'),
                **get_dict(waist_index, 'waist'),
                **get_dict(end_of_drift_index, 'end_of_drift'),
                'waist_z': (waist, 'm')
            }
        }
        pickle.dump(data_dictionary, f)
        with open(results_folder / 'data.txt', 'w+') as f:
            f.write(str(data_dictionary))

    eog = ('gun_exit', 'At Gun Exit ($z = {:.2f}$cm)'.format(end_of_gun * 100), end_of_gun_index)
    wst = ('waist', 'At Beam Waist ($z = {:.2f}$cm)'.format(waist * 100), waist_index)
    eod = ('end_of_drift', 'End of Drift ($z = {:.2f}$cm)'.format(end_of_drift * 100), end_of_drift_index)
    for (name, title, index) in [eog, wst, eod]:
        plt.title(title)
        plt.scatter(1e6 * data.time_blocks[index].electrons['x'], data.time_blocks[index].electrons['G'] * data.time_blocks[index].electrons['Bx'], s=0.5)
        plt.xlabel('$x$ ($\\mu$m)')
        plt.ylabel('$p_x / m_e c$')
        plt.savefig(results_folder / f'x_phase_space_{name}.png', dpi=300)
        plt.clf()

        plt.title(title)
        plt.scatter(1e6 * data.time_blocks[index].electrons['y'], data.time_blocks[index].electrons['G'] * data.time_blocks[index].electrons['By'], s=0.5)
        plt.xlabel('$y$ ($\\mu$m)')
        plt.ylabel('$p_y / m_e c$')
        plt.savefig(results_folder / f'y_phase_space_{name}.png', dpi=300)
        plt.clf()

        plt.title(title)
        plt.scatter(1e3 * (data.time_blocks[index].electrons['z'] - z_centroid[index]), data.time_blocks[index].electrons['G'] * data.time_blocks[index].electrons['Bz'], s=0.5)
        plt.xlabel('$z$ (mm)')
        plt.ylabel('$p_z / m_e c$')
        plt.savefig(results_folder / f'z_phase_space_{name}.png', dpi=300)
        plt.clf()

    plt.plot(ts * 1e12, x_centroid * 1e6, label='x')
    plt.plot(ts * 1e12, y_centroid * 1e6, label='y')
    plt.axvline(x=ts[end_of_gun_index] * 1e12, color='black', linestyle='-', label='Gun Exit')
    plt.axvline(x=ts[waist_index] * 1e12, color='black', linestyle='--', label='Waist')
    plt.xlabel('time (ps)')
    plt.ylabel('centroid ($\\mu$m)')
    plt.legend()
    plt.savefig(results_folder / 'xy_centroid.png', dpi=300)
    plt.clf()

    plt.plot(ts * 1e12, z_centroid * 1e3)
    plt.axvline(x=ts[end_of_gun_index] * 1e12, color='black', linestyle='-', label='Gun Exit')
    plt.axvline(x=ts[waist_index] * 1e12, color='black', linestyle='--', label='Waist')
    plt.xlabel('time (ps)')
    plt.ylabel('z centroid (mm)')
    plt.legend()
    plt.savefig(results_folder / 'z_centroid.png', dpi=300)
    plt.clf()

    plt.plot(ts * 1e12, x_std * 1e6, label='x')
    plt.plot(ts * 1e12, y_std * 1e6, label='y')
    plt.axvline(x=ts[end_of_gun_index] * 1e12, color='black', linestyle='-', label='Gun Exit')
    plt.axvline(x=ts[waist_index] * 1e12, color='black', linestyle='--', label='Waist')
    plt.xlabel('time (ps)')
    plt.ylabel('$\\sigma$ ($\\mu$m)')
    plt.legend()
    plt.savefig(results_folder / 'xy_sigma.png', dpi=300)
    plt.clf()

    plt.plot(ts * 1e12, r_std * 1e6)
    plt.axvline(x=ts[end_of_gun_index] * 1e12, color='black', linestyle='-', label='Gun Exit')
    plt.axvline(x=ts[waist_index] * 1e12, color='black', linestyle='--', label='Waist')
    plt.xlabel('time (ps)')
    plt.ylabel('$\\sigma_r$ ($\\mu$m)')
    plt.legend()
    plt.savefig(results_folder / 'r_sigma.png', dpi=300)
    plt.clf()

    plt.plot(ts * 1e12, z_std * 1e3)
    plt.axvline(x=ts[end_of_gun_index] * 1e12, color='black', linestyle='-', label='Gun Exit')
    plt.axvline(x=ts[waist_index] * 1e12, color='black', linestyle='--', label='Waist')
    plt.xlabel('time (ps)')
    plt.ylabel('$\\sigma_z$ (mm)')
    plt.legend()
    plt.savefig(results_folder / 'z_sigma.png', dpi=300)
    plt.clf()

    plt.plot(ts * 1e12, x_emit * 1e6, label='x')
    plt.plot(ts * 1e12, y_emit * 1e6, label='y')
    plt.axvline(x=ts[end_of_gun_index] * 1e12, color='black', linestyle='-', label='Gun Exit')
    plt.axvline(x=ts[waist_index] * 1e12, color='black', linestyle='--', label='Waist')
    plt.xlabel('time (ps)')
    plt.ylabel('$\\epsilon_n$ ($\\mu$m)')
    plt.legend()
    plt.savefig(results_folder / 'xy_emit.png', dpi=300)
    plt.clf()

    plt.plot(ts * 1e12, emit_6d)
    plt.axvline(x=ts[end_of_gun_index] * 1e12, color='black', linestyle='-', label='Gun Exit')
    plt.axvline(x=ts[waist_index] * 1e12, color='black', linestyle='--', label='Waist')
    plt.xlabel('time (ps)')
    plt.ylabel('$\\epsilon_{6d}$ ($\\mathrm{m}^3$)')
    plt.legend()
    plt.savefig(results_folder / '6d_emit.png', dpi=300)
    plt.clf()

    plt.plot(ts * 1e12, emit_4d * 1e9 * 1e9)
    plt.axvline(x=ts[end_of_gun_index] * 1e12, color='black', linestyle='-', label='Gun Exit')
    plt.axvline(x=ts[waist_index] * 1e12, color='black', linestyle='--', label='Waist')
    plt.xlabel('time (ps)')
    plt.ylabel('$\\epsilon_{4d}$ ($\\mathrm{nm}^3$)')
    plt.legend()
    plt.savefig(results_folder / '4d_emit.png', dpi=300)
    plt.clf()

    plt.plot(ts * 1e12, brightness_6d)
    plt.axvline(x=ts[end_of_gun_index] * 1e12, color='black', linestyle='-', label='Gun Exit')
    plt.axvline(x=ts[waist_index] * 1e12, color='black', linestyle='--', label='Waist')
    plt.xlabel('time (ps)')
    plt.ylabel('$B_{6d}$ ($\\mathrm{m}^{-3}$)')
    plt.legend()
    plt.savefig(results_folder / '6d_bright.png', dpi=300)
    plt.clf()

    plt.plot(ts * 1e12, brightness_4d)
    plt.axvline(x=ts[end_of_gun_index] * 1e12, color='black', linestyle='-', label='Gun Exit')
    plt.axvline(x=ts[waist_index] * 1e12, color='black', linestyle='--', label='Waist')
    plt.xlabel('time (ps)')
    plt.ylabel('$B_{4d}$ ($\mathrm{A}\\mathrm{m}^{-2}$)')
    plt.legend()
    plt.savefig(results_folder / '4d_bright.png', dpi=300)
    plt.clf()

    plt.plot(ts * 1e12, charge)
    plt.axvline(x=ts[end_of_gun_index] * 1e12, color='black', linestyle='-', label='Gun Exit')
    plt.axvline(x=ts[waist_index] * 1e12, color='black', linestyle='--', label='Waist')
    plt.xlabel('time (ps)')
    plt.ylabel('$Q$ (C)')
    plt.legend()
    plt.savefig(results_folder / 'charge.png', dpi=300)
    plt.clf()







    plt.plot(z_centroid * 100, x_centroid * 1e6, label='x')
    plt.plot(z_centroid * 100, y_centroid * 1e6, label='y')
    plt.axvline(x=end_of_gun * 100, color='black', linestyle='-', label='Gun Exit')
    plt.axvline(x=waist * 100, color='black', linestyle='--', label='Waist')
    plt.xlabel('$z$ (cm)')
    plt.ylabel('centroid ($\\mu$m)')
    plt.legend()
    plt.savefig(results_folder / 'xy_centroid_2.png', dpi=300)
    plt.clf()

    plt.plot(z_centroid * 100, x_std * 1e6, label='x')
    plt.plot(z_centroid * 100, y_std * 1e6, label='y')
    plt.axvline(x=end_of_gun * 100, color='black', linestyle='-', label='Gun Exit')
    plt.axvline(x=waist * 100, color='black', linestyle='--', label='Waist')
    plt.xlabel('$z$ (cm)')
    plt.ylabel('$\\sigma$ ($\\mu$m)')
    plt.legend()
    plt.savefig(results_folder / 'xy_sigma_2.png', dpi=300)
    plt.clf()

    plt.plot(z_centroid * 100, r_std * 1e6)
    plt.axvline(x=end_of_gun * 100, color='black', linestyle='-', label='Gun Exit')
    plt.axvline(x=waist * 100, color='black', linestyle='--', label='Waist')
    plt.xlabel('$z$ (cm)')
    plt.ylabel('$\\sigma_r$ ($\\mu$m)')
    plt.legend()
    plt.savefig(results_folder / 'r_sigma_2.png', dpi=300)
    plt.clf()

    plt.plot(z_centroid * 100, z_std * 1e3)
    plt.axvline(x=end_of_gun * 100, color='black', linestyle='-', label='Gun Exit')
    plt.axvline(x=waist * 100, color='black', linestyle='--', label='Waist')
    plt.xlabel('$z$ (cm)')
    plt.ylabel('$\\sigma_z$ (mm)')
    plt.legend()
    plt.savefig(results_folder / 'z_sigma_2.png', dpi=300)
    plt.clf()

    plt.plot(z_centroid * 100, x_emit * 1e6, label='x')
    plt.plot(z_centroid * 100, y_emit * 1e6, label='y')
    plt.axvline(x=end_of_gun * 100, color='black', linestyle='-', label='Gun Exit')
    plt.axvline(x=waist * 100, color='black', linestyle='--', label='Waist')
    plt.xlabel('$z$ (cm)')
    plt.ylabel('$\\epsilon_n$ ($\\mu$m)')
    plt.legend()
    plt.savefig(results_folder / 'xy_emit_2.png', dpi=300)
    plt.clf()

    plt.plot(z_centroid * 100, emit_6d)
    plt.axvline(x=end_of_gun * 100, color='black', linestyle='-', label='Gun Exit')
    plt.axvline(x=waist * 100, color='black', linestyle='--', label='Waist')
    plt.xlabel('$z$ (cm)')
    plt.ylabel('$\\epsilon_{6d}$ ($\\mathrm{m}^3$)')
    plt.legend()
    plt.savefig(results_folder / '6d_emit_2.png', dpi=300)
    plt.clf()

    plt.plot(z_centroid * 100, emit_4d * 1e9 * 1e9)
    plt.axvline(x=end_of_gun * 100, color='black', linestyle='-', label='Gun Exit')
    plt.axvline(x=waist * 100, color='black', linestyle='--', label='Waist')
    plt.xlabel('$z$ (cm)')
    plt.ylabel('$\\epsilon_{4d}$ ($\\mathrm{nm}^2$)')
    plt.legend()
    plt.savefig(results_folder / '4d_emit_2.png', dpi=300)
    plt.clf()

    plt.plot(z_centroid * 100, brightness_6d)
    plt.axvline(x=end_of_gun * 100, color='black', linestyle='-', label='Gun Exit')
    plt.axvline(x=waist * 100, color='black', linestyle='--', label='Waist')
    plt.xlabel('$z$ (cm)')
    plt.ylabel('$B_{6d}$ ($\\mathrm{m}^{-3}$)')
    plt.legend()
    plt.savefig(results_folder / '6d_bright_2.png', dpi=300)
    plt.clf()

    plt.plot(z_centroid * 100, brightness_4d)
    plt.axvline(x=end_of_gun * 100, color='black', linestyle='-', label='Gun Exit')
    plt.axvline(x=waist * 100, color='black', linestyle='--', label='Waist')
    plt.xlabel('$z$ (cm)')
    plt.ylabel('$B_{4d}$ ($\mathrm{A}\\mathrm{m}^{-2}$)')
    plt.legend()
    plt.savefig(results_folder / '4d_bright_2.png', dpi=300)
    plt.clf()


    plt.plot(z_centroid * 100, charge)
    plt.axvline(x=end_of_gun * 100, color='black', linestyle='-', label='Gun Exit')
    plt.axvline(x=waist * 100, color='black', linestyle='--', label='Waist')
    plt.xlabel('$z$ (cm)')
    plt.ylabel('$Q$ (C)')
    plt.legend()
    plt.savefig(results_folder / 'charge_2.png', dpi=300)
    plt.clf()

if __name__ == '__main__':
    mot.run(__file__, do_scan, IOs, SCs, processes=6, detach=True)
    prin
