#!/usr/bin/env python3

import json
import itertools
import matplotlib.pyplot as plt
import mot
import numpy as np
import os
import pathlib
import pickle
import sys
import multiprocessing

SCANS_INDEX = int(sys.argv[1])

os.system(f'mkdir -p scans_{SCANS_INDEX}.results')
qwer = open(f'scans_{SCANS_INDEX}.results/finalize.sh', 'w+')
qwerl = multiprocessing.Lock()

qwer.write('#!/bin/bash\n')
qwer.write('for i in `seq 0 $(expr $(ls scans | wc -l) - 1)`; do movie scans/$i/x_phase_space_frame_%d.png scans/$i/x_phase_space.mp4; done &\n')
qwer.write('for i in `seq 0 $(expr $(ls scans | wc -l) - 1)`; do movie scans/$i/y_phase_space_frame_%d.png scans/$i/y_phase_space.mp4; done &\n')
qwer.write('for i in `seq 0 $(expr $(ls scans | wc -l) - 1)`; do movie scans/$i/z_phase_space_frame_%d.png scans/$i/z_phase_space.mp4; done &\n')
qwer.write('wait\n')
qwer.write('find . -name \'*frame*\' | xargs rm\n')
qwer.close()
os.system(f'chmod +x scans_{SCANS_INDEX}.results/finalize.sh')

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

#magnetic_field_strengths = np.linspace(-0.8, 0.8, SCANS_INDEX, endpoint=True)
solenoid_lengths = np.linspace(0.01, 0.20, SCANS_INDEX, endpoint=True)
solenoid_positions = np.linspace(0.01, 0.20, SCANS_INDEX, endpoint=True)

particles_dir = pathlib.Path(f'/b/chansel/mot/scans_{SCANS_INDEX}/particles')

init_on = True

def set_params(s):
    s.peak_density = 2e16
    s.sigma_r = 0.0005
    s.sigma_pr = 1.9424e-26
    s.mot_z_offset = -0.0005
    s.laser_phi_x = 0
    s.laser_phi_y = 0.5 * np.pi / 180
    s.laser_x = 0
    s.laser_y = 0
    s.laser_z = 0
    s.laser_width = 25e-6
    s.ions_on = False

def init_particles():
    s = mot.MotGunSimulation(particles_dir)
    set_params(s)
    s.initializeParticles(seed = 9423424)

def do_scan(i, path, solenoid_length, solenoid_position):
    print(f'running simulation {i}: ', path)
    s = mot.MotGunSimulation(path)
    set_params(s)
    s.peak_field = 100e6
    s.rf_frequency = 2.85648e9
    s.rf_phase_deg = 90
    s.pyramid_height = 0.0081
    s.sol_radius = 0.1
    s.sol_strength = -0.8 #magnetic_field_strength
    s.sol_length = solenoid_length
    s.sol_z = solenoid_position
    s.spacecharge = 'none'
    s.tmax = 1.6e-9
    s.tstep = 2e-11
    if s.ions_on:
        (path / 'ions.gdf').symlink_to(particles_dir / 'ions.gdf')
    (path / 'beam.gdf').symlink_to(particles_dir / 'beam.gdf')
    s.run()
    s.readData()
    analyze(i, s.data, path, 0.12, 0.4, s.parametersDict())
    s.deleteData()
    return i

def analyze(simnum, data, path, end_of_gun, end_of_drift, parameters_dict):
    print("--> analyzing data")
    results_folder = path / 'results'
    if not results_folder.exists():
        results_folder.mkdir(parents=True)
    results_folder.resolve(strict=True)

    r'''
    output_file = s.folder / 'output.txt'
    ts = []
    x_centroid = []
    y_centroid = []
    z_centroid = []
    x_std = []
    y_std = []
    z_std = []
    beta_z = []
    x_emit = []
    y_emit = []
    emit_6d = []
    emit_4d = []
    brightness_6d = []
    x_emit_nn = []
    y_emit_nn = []
    charge = []
    for time_block in mot.getTimeBlocks(output_file):
        ts.append(time_block.time)
        x_centroid.append(time_block.electrons['x'].mean())
        y_centroid.append(time_block.electrons['y'].mean())
        z_centroid.append(time_block.electrons['z'].mean())
        x_std.append(time_block.electrons['x'].std())
        y_std.append(time_block.electrons['y'].std())
        z_std.append(time_block.electrons['z'].std())
        beta_z.append(time_block.electrons['Bz'])
        x_emit.append(np.sqrt(np.linalg.det(np.cov(np.array([
            time_block.electrons['x'],
            time_block.electrons['G'] * time_block.electrons['Bx']
        ])))))
        y_emit.append(np.sqrt(np.linalg.det(np.cov(np.array([
            time_block.electrons['y'],
            time_block.electrons['G'] * time_block.electrons['By']
        ])))))
        emit_6d.append(np.sqrt(np.linalg.det(np.cov(np.array([
            time_block.electrons['x'],
            time_block.electrons['y'],
            time_block.electrons['z'],
            time_block.electrons['G'] * time_block.electrons['Bx'],
            time_block.electrons['G'] * time_block.electrons['By'],
            time_block.electrons['G'] * time_block.electrons['Bz'],
        ])))))
        emit_4d.append(np.sqrt(np.linalg.det(np.cov(np.array([
            time_block.electrons['x'],
            time_block.electrons['y'],
            time_block.electrons['G'] * time_block.electrons['Bx'],
            time_block.electrons['G'] * time_block.electrons['By'],
        ])))))
        brightness_6d.append(len(time_block.electrons['x']) / emit_6d[-1])
        x_emit_nn.append(np.sqrt(np.linalg.det(np.cov(np.array([
            time_block.electrons['x'],
            time_block.electrons['G'] * time_block.electrons['Bx']
        ])))))
        y_emit_nn.append(np.sqrt(np.linalg.det(np.cov(np.array([
            time_block.electrons['y'],
            time_block.electrons['G'] * time_block.electrons['By']
        ])))))
        charge.append(-1.60217662e-19 * len(time_block.electrons['x']))

    ts = np.array(ts)
    x_centroid = np.array(x_centroid)
    y_centroid = np.array(y_centroid)
    z_centroid = np.array(z_centroid)
    x_std = np.array(x_std)
    y_std = np.array(y_std)
    z_std = np.array(z_std)
    beta_z = np.array(beta_z)
    x_emit = np.array(x_emit)
    y_emit = np.array(y_emit)
    emit_6d = np.array(emit_6d)
    emit_4d = np.array(emit_4d)
    brightness_6d = np.array(brightness_6d)
    x_emit_nn = np.array(x_emit_nn)
    y_emit_nn = np.array(y_emit_nn)
    charge = np.array(charge)

    r_std = np.sqrt(x_std ** 2 + y_std ** 2)
    brightness_4d = -299792458 * charge / (z_std * np.pi * np.pi * x_emit_nn * y_emit_nn)
    '''

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

    print('--> plotting phase space')
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
        plt.xlabel('$z - \\overline{z}$ (mm)')
        plt.ylabel('$p_z / m_e c$')
        plt.savefig(results_folder / f'z_phase_space_{name}.png', dpi=300)
        plt.clf()


    def smart_lower_bound(data_set, center):
        if not center:
            return max([data_set.mean() - 4 * data_set.std(), data_set.min()])
        else:
            slb = max([data_set.mean() - 4 * data_set.std(), data_set.min()])
            sub =  min([data_set.mean() + 4 * data_set.std(), data_set.max()])
            return -max([abs(slb), abs(sub)])


    def smart_upper_bound(data_set, center):
        if not center:
            return min([data_set.mean() + 4 * data_set.std(), data_set.max()])
        else:
            slb = max([data_set.mean() - 4 * data_set.std(), data_set.min()])
            sub =  min([data_set.mean() + 4 * data_set.std(), data_set.max()])
            return max([abs(slb), abs(sub)])

    ctr = True
    min_x = min([smart_lower_bound(block.electrons['x'], center=ctr) for block in data.time_blocks])
    max_x = max([smart_upper_bound(block.electrons['x'], center=ctr) for block in data.time_blocks])
    min_px = min([smart_lower_bound(block.electrons['G'] * block.electrons['Bx'], center=ctr) for block in data.time_blocks])
    max_px = max([smart_upper_bound(block.electrons['G'] * block.electrons['Bx'], center=ctr) for block in data.time_blocks])
    min_y = min([smart_lower_bound(block.electrons['y'], center=ctr) for block in data.time_blocks])
    max_y = max([smart_upper_bound(block.electrons['y'], center=ctr) for block in data.time_blocks])
    min_py = min([smart_lower_bound(block.electrons['G'] * block.electrons['By'], center=ctr) for block in data.time_blocks])
    max_py = max([smart_upper_bound(block.electrons['G'] * block.electrons['By'], center=ctr) for block in data.time_blocks])
    min_z = min([smart_lower_bound(block.electrons['z'] - block.electrons['z'].mean(), center=ctr) for block in data.time_blocks])
    max_z = max([smart_upper_bound(block.electrons['z'] - block.electrons['z'].mean(), center=ctr) for block in data.time_blocks])
    min_pz = min([smart_lower_bound(block.electrons['G'] * block.electrons['Bz'], center=ctr) for block in data.time_blocks])
    max_pz = max([smart_upper_bound(block.electrons['G'] * block.electrons['Bz'], center=ctr) for block in data.time_blocks])


    for index, t in enumerate(ts):
        #print(f'--> plotting phase space {index} of {len(ts)}')
        plt.title(f'$t = {1e12 * t:.2f}$ ps, z = {1e3 * z_centroid[index]:.2f}mm')
        plt.scatter(1e6 * data.time_blocks[index].electrons['x'], data.time_blocks[index].electrons['G'] * data.time_blocks[index].electrons['Bx'], s=0.5)
        plt.xlim(1e6 * min_x, 1e6 * max_x)
        plt.ylim(min_px, max_px)
        plt.xlabel('$x$ ($\\mu$m)')
        plt.ylabel('$p_x / m_e c$')
        plt.savefig(results_folder / f'x_phase_space_frame_{index}.png', dpi=300)
        plt.clf()

        plt.title(f'$t = {1e12 * t:.2f}$ ps, z = {1e3 * z_centroid[index]:.2f}mm')
        plt.scatter(1e6 * data.time_blocks[index].electrons['y'], data.time_blocks[index].electrons['G'] * data.time_blocks[index].electrons['By'], s=0.5)
        plt.xlim(1e6 * min_y, 1e6 * max_y)
        plt.ylim(min_py, max_py)
        plt.xlabel('$y$ ($\\mu$m)')
        plt.ylabel('$p_y / m_e c$')
        plt.savefig(results_folder / f'y_phase_space_frame_{index}.png', dpi=300)
        plt.clf()

        plt.title(f'$t = {1e12 * t:.2f}$ ps, z = {1e3 * z_centroid[index]:.2f}mm')
        plt.scatter(1e3 * (data.time_blocks[index].electrons['z'] - z_centroid[index]), data.time_blocks[index].electrons['G'] * data.time_blocks[index].electrons['Bz'], s=0.5)
        plt.xlim(1e3 * min_z, 1e3 * max_z)
        plt.ylim(min_pz, max_pz)
        plt.xlabel('$z - \\overline{z}$ (mm)')
        plt.ylabel('$p_z / m_e c$')
        plt.savefig(results_folder / f'z_phase_space_frame_{index}.png', dpi=300)
        plt.clf()

    #try:
    #    with qwerl:
    #        qwer.write(f'ffmpeg -i scans/{simnum}/x_phase_space_frame_%d.png -c:v libx264 -pix_fmt yuv420p scans/{simnum}/x_phase_space.mp4 &\n')
    #        qwer.write(f'ffmpeg -i scans/{simnum}/y_phase_space_frame_%d.png -c:v libx264 -pix_fmt yuv420p scans/{simnum}/y_phase_space.mp4 &\n')
    #        qwer.write(f'ffmpeg -i scans/{simnum}/z_phase_space_frame_%d.png -c:v libx264 -pix_fmt yuv420p scans/{simnum}/z_phase_space.mp4 &\n')
    #        qwer.write(f'wait\n')
    #        qwer.write(f'rm scans/{simnum}/*_phase_space_frame_*.png\n')
    #        qwer.flush()
    #except Exception as e:
    #    print(e)

    print('--> plotting everything else')

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

    plt.plot(ts * 1e12, x_emit * 1e9, label='x')
    plt.plot(ts * 1e12, y_emit * 1e9, label='y')
    plt.axvline(x=ts[end_of_gun_index] * 1e12, color='black', linestyle='-', label='Gun Exit')
    plt.axvline(x=ts[waist_index] * 1e12, color='black', linestyle='--', label='Waist')
    plt.xlabel('time (ps)')
    plt.ylabel('$\\epsilon_n$ (nm)')
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
    plt.ylim(0, 3 * brightness_6d[len(brightness_6d) // 2])
    plt.xlabel('time (ps)')
    plt.ylabel('$B_{6d}$ ($\\mathrm{m}^{-3}$)')
    plt.legend()
    plt.savefig(results_folder / '6d_bright.png', dpi=300)
    plt.clf()

    plt.plot(ts * 1e12, brightness_4d)
    plt.axvline(x=ts[end_of_gun_index] * 1e12, color='black', linestyle='-', label='Gun Exit')
    plt.axvline(x=ts[waist_index] * 1e12, color='black', linestyle='--', label='Waist')
    plt.ylim(0, 3 * brightness_4d[len(brightness_4d) // 2])
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

    plt.plot(z_centroid * 100, x_emit * 1e9, label='x')
    plt.plot(z_centroid * 100, y_emit * 1e9, label='y')
    plt.axvline(x=end_of_gun * 100, color='black', linestyle='-', label='Gun Exit')
    plt.axvline(x=waist * 100, color='black', linestyle='--', label='Waist')
    plt.xlabel('$z$ (cm)')
    plt.ylabel('$\\epsilon_n$ (nm)')
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
    plt.ylim(0, 3 * brightness_6d[len(brightness_6d) // 2])
    plt.xlabel('$z$ (cm)')
    plt.ylabel('$B_{6d}$ ($\\mathrm{m}^{-3}$)')
    plt.legend()
    plt.savefig(results_folder / '6d_bright_2.png', dpi=300)
    plt.clf()

    plt.plot(z_centroid * 100, brightness_4d)
    plt.axvline(x=end_of_gun * 100, color='black', linestyle='-', label='Gun Exit')
    plt.axvline(x=waist * 100, color='black', linestyle='--', label='Waist')
    plt.ylim(0, 3 * brightness_4d[len(brightness_4d) // 2])
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

def analyze2():

    os.system(f'mkdir -p scans_{SCANS_INDEX}.results/scans')

    all_scan_results = None
    for i in range(len(os.listdir(f'/b/chansel/mot/scans_{SCANS_INDEX}')) - 1):
        os.system(f'mkdir -p scans_{SCANS_INDEX}.results/scans/{i}')
        os.system(f'cp /b/chansel/mot/scans_{SCANS_INDEX}/{i}/results/* scans_{SCANS_INDEX}.results/scans/{i}/')
        with open(f'/b/chansel/mot/scans_{SCANS_INDEX}/{i}/results/data.pickle', 'rb') as f:
            scan_results = pickle.load(f)
            if all_scan_results is None:
                all_scan_results = {key: {key2: [] for key2 in value.keys()} for key, value in scan_results.items()}
            for key, value in scan_results.items():
                for key2, value2 in value.items():
                    all_scan_results[key][key2].append(value2[0])
            with open(f'/b/chansel/mot/scans_{SCANS_INDEX}/{i}/results/data.json', 'w+') as f:
                json.dump(scan_results, f, indent=4, cls=NpEncoder)

    shape = (len(solenoid_lengths), len(solenoid_positions))#(len(magnetic_field_strengths), len(solenoid_lengths), len(solenoid_positions))
    sol_strength = np.array(all_scan_results['parameters']['sol_strength']).reshape(shape)
    sol_length = np.array(all_scan_results['parameters']['sol_length']).reshape(shape)
    sol_z = np.array(all_scan_results['parameters']['sol_z']).reshape(shape)

    _, d1 = np.linspace(solenoid_lengths.min(), solenoid_lengths.max(), solenoid_lengths.size, endpoint=True, retstep=True)
    assert (_ == solenoid_lengths).all()
    _, d2 = np.linspace(solenoid_positions.min(), solenoid_positions.max(), solenoid_positions.size, endpoint=True, retstep=True)
    assert (_ == solenoid_positions).all()

    solenoid_lengths_midpoint = np.linspace(solenoid_lengths.min() - 0.5 * d1, solenoid_lengths.max() + 0.5 * d1, solenoid_lengths.size + 1, endpoint=True)
    solenoid_positions_midpoint = np.linspace(solenoid_positions.min() - 0.5 * d2, solenoid_positions.max() + 0.5 * d2, solenoid_positions.size + 1, endpoint=True)

    plot_infos = [
        ('Peak Solenoid Magnetic Field (T)', sol_strength, 1, 'sol_strength'),
        ('Solenoid Length (cm)', sol_length, 100, 'sol_length'),
        ('Solenoid Position (cm)', sol_z, 100, 'sol_z')
    ]

    for name, name2 in (('end_of_gun', 'Gun End'), ('waist', 'Waist'), ('end_of_drift', 'Drift End')):
        time = np.array(all_scan_results['computed_values'][f'time_{name}']).reshape(shape)
        x_centroid = np.array(all_scan_results['computed_values'][f'x_centroid_{name}']).reshape(shape)
        y_centroid = np.array(all_scan_results['computed_values'][f'y_centroid_{name}']).reshape(shape)
        z_centroid = np.array(all_scan_results['computed_values'][f'z_centroid_{name}']).reshape(shape)
        x_std = np.array(all_scan_results['computed_values'][f'x_std_{name}']).reshape(shape)
        y_std = np.array(all_scan_results['computed_values'][f'y_std_{name}']).reshape(shape)
        z_std = np.array(all_scan_results['computed_values'][f'z_std_{name}']).reshape(shape)
        x_emit = np.array(all_scan_results['computed_values'][f'x_emit_{name}']).reshape(shape)
        y_emit = np.array(all_scan_results['computed_values'][f'y_emit_{name}']).reshape(shape)
        emit_6d = np.array(all_scan_results['computed_values'][f'emit_6d_{name}']).reshape(shape)
        emit_4d = np.array(all_scan_results['computed_values'][f'emit_4d_{name}']).reshape(shape)
        bright_4d = np.array(all_scan_results['computed_values'][f'brightness_4d_{name}']).reshape(shape)
        bright_6d = np.array(all_scan_results['computed_values'][f'brightness_6d_{name}']).reshape(shape)
        charge = np.array(all_scan_results['computed_values'][f'charge_{name}']).reshape(shape)
        plot_infos += [
            ('Time at ' + name2 + '(ps)', time, 1e12, f'time_{name}'),
            ('$Q_{\\mathrm{' + name2 + '}}$ (fC)', charge, 1e15, f'charge_{name}'),
            ('$\\overline{x}_{\\mathrm{' + name2 + '}}$ ($\\mu$m)', x_centroid, 1e6, f'x_centroid_{name}'),
            ('$\\overline{y}_{\\mathrm{' + name2 + '}}$ ($\\mu$m)', y_centroid, 1e6, f'y_centroid_{name}'),
            ('$\\overline{z}_{\\mathrm{' + name2 + '}}$ (cm)', z_centroid, 100, f'z_centroid_{name}'),
            ('$\\sigma_{x, \\mathrm{' + name2 + '}}$ ($\\mu$m)', x_std, 1e6, f'x_std_{name}'),
            ('$\\sigma_{y, \\mathrm{' + name2 + '}}$ ($\\mu$m)', y_std, 1e6, f'y_std_{name}'),
            ('$\\sigma_{z, \\mathrm{' + name2 + '}}$ (mm)', z_std, 1e3, f'z_std_{name}'),
            ('$\\epsilon_{x, \\mathrm{' + name2 + '}}$ (nm)', x_emit, 1e9, f'x_emit_{name}'),
            ('$\\epsilon_{y, \\mathrm{' + name2 + '}}$ (nm)', y_emit, 1e9, f'y_emit_{name}'),
            ('$B_{\\mathrm{4D}, \\mathrm{' + name2 + '}}$ ($\\mathrm{A}\\mathrm{m}^{-2}$)', bright_4d, 1, f'4d_bright_{name}'),
            ('$B_{\\mathrm{6D}, \\mathrm{' + name2 + '}}$ ($\\mathrm{m}^{-3}$)', bright_6d, 1, f'6d_bright_{name}'),
            ('$\\epsilon_{\\mathrm{6D}, \\mathrm{' + name2 + '}}$ (m$^3$)', emit_6d, 1, f'emit_6d_{name}'),
            ('$\\epsilon_{\\mathrm{4D}, \\mathrm{' + name2 + '}}$ (nm$^2$)', emit_4d, 1e9 * 1e9, f'emit_4d_{name}')
        ]

    for plot_info in plot_infos:
        #for i, magnetic_field_strength in enumerate(magnetic_field_strengths):
        #plt.title('$B_{\\mathrm{max}} = ' + '{:.2f}'.format(magnetic_field_strength) + '$ T')
        plt.pcolormesh(solenoid_lengths_midpoint * 100, solenoid_positions_midpoint * 100, plot_info[2] * plot_info[1].T, vmin=(plot_info[2] * plot_info[1].min()), vmax=(plot_info[2] * plot_info[1].max()))
        plt.xlabel('Solenoid Length (cm)')
        plt.ylabel('Solenoid Position (cm)')
        plt.colorbar(label=plot_info[0])
        plt.savefig(f'scans_{SCANS_INDEX}.results/{plot_info[3]}.png', dpi=300)
        plt.clf()
        #print(f'ffmpeg -i scans_{SCANS_INDEX}.results/{plot_info[3]}_%d.png -c:v libx264 -pix_fmt yuv420p scans_{SCANS_INDEX}.results/{plot_info[3]}.mp4')
        #os.system(f'ffmpeg -i scans_{SCANS_INDEX}.results/{plot_info[3]}_%d.png -c:v libx264 -pix_fmt yuv420p scans_{SCANS_INDEX}.results/{plot_info[3]}.mp4')
        #os.system(f'rm scans_{SCANS_INDEX}.results/{plot_info[3]}_*.png')

    #qwer.close()
    #with qwerl:
    #    qwer.write(f'tar -czvf scans_{SCANS_INDEX}.results.tgz scans_{SCANS_INDEX}.results && rm -rf scans_{SCANS_INDEX}.results\n')

    with open(f'scans_{SCANS_INDEX}.results/map.txt', 'w+') as f:
        for i, (solenoid_length, solenoid_position) in enumerate(itertools.product(solenoid_lengths, solenoid_positions)):
            f.write(f'{i:>4}: sol_len = {solenoid_length:.3f}, sol_pos = {solenoid_position:.3f}\n')

    os.system(f'tar -cvf scans_{SCANS_INDEX}.results.tar scans_{SCANS_INDEX}.results && rm -rf scans_{SCANS_INDEX}.results')


if __name__ == '__main__':
    if init_on:
        init_particles()
    mot.run(__file__, do_scan, solenoid_lengths, solenoid_positions, processes=20, detach=True, suffix=SCANS_INDEX, data_folder_path='/b/chansel/mot')
    analyze2()
    print('PROGRAM DONE')
