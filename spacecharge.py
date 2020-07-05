#!/usr/bin/env python3

import daemon
import itertools
import mot
import multiprocessing
import matplotlib.pyplot as plt
import numpy as np
import os
import pathlib
import subprocess
import sys

processes = 1

scan_parameters = ['approximate', 'exact']


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

def analyze(data1, data2, title_1, title_2, gun_length, results_folder):
    names = ('z', 'x', 'y', 'bz', 'bx', 'by', 'g')
    labels = ('$z - ct$ (mm)', '$x$ (mm)', '$y$ (mm)', '$\\beta_z$', '$\\beta_x$', '$\\beta_y$', '$\\gamma$')
    is_center = (False, True, True, False, True, True, False)
    functions = [
        lambda data, time: 1000 * (data['z'] - 299792458 * time),
        lambda data, time: 1000 * data['x'],
        lambda data, time: 1000 * data['y'],
        lambda data, time: data['Bz'],
        lambda data, time: data['Bx'],
        lambda data, time: data['By'],
        lambda data, time: data['G']
    ]

    electron_mins = [min([smart_lower_bound(functions[i](block.electrons, block.time), center=is_center[i]) for block in data1.time_blocks]
                       + [smart_lower_bound(functions[i](block.electrons, block.time), center=is_center[i]) for block in data2.time_blocks]) for i in range(len(names))]
    electron_maxs = [max([smart_upper_bound(functions[i](block.electrons, block.time), center=is_center[i]) for block in data1.time_blocks]
                       + [smart_upper_bound(functions[i](block.electrons, block.time), center=is_center[i]) for block in data2.time_blocks]) for i in range(len(names))]

    electron_hists_1 = [[np.histogram(functions[i](block.electrons, block.time), range=(electron_mins[i], electron_maxs[i]), bins=200) for block in data1.time_blocks] for i in range(len(names))]
    electron_hists_2 = [[np.histogram(functions[i](block.electrons, block.time), range=(electron_mins[i], electron_maxs[i]), bins=200) for block in data2.time_blocks] for i in range(len(names))]
    electron_hist_maxs = [max([hist1[i][0].max() for i in range(len(data1.time_blocks))] + [hist2[i][0].max() for i in range(len(data2.time_blocks))]) for (hist1, hist2) in zip(electron_hists_1, electron_hists_2)]

    for i in range(len(names)):
        name = names[i]
        label = labels[i]
        function = functions[i]
        for j, time_block in enumerate(data1.time_blocks):
            e_counts_1, e_bins_1 = electron_hists_1[i][j]
            e_counts_2, e_bins_2 = electron_hists_2[i][j]
            fig, (ax1, ax2) = plt.subplots(nrows=1, ncols=2)
            fig.suptitle('t = ' + str(int(round(time_block.time * 1e12))) + 'ps')
            ax1.set_title(title_1)
            ax1.hist(0.5 * (e_bins_1[:-1] + e_bins_1[1:]), e_bins_1, weights=e_counts_1)
            if name == 'z':
                ax1.plot([1000 * (gun_length - 299792458 * time_block.time), 1000 * (gun_length - 299792458 * time_block.time)], [0, electron_hist_maxs[i]], 'k')
            ax1.set_xlim(electron_mins[i], electron_maxs[i])
            ax1.set_ylim(0, electron_hist_maxs[i])
            ax1.set_xlabel(label)
            ax2.set_title(title_2)
            ax2.hist(0.5 * (e_bins_2[:-1] + e_bins_2[1:]), e_bins_2, weights=e_counts_1)
            if name == 'z':
                ax2.plot([1000 * (gun_length - 299792458 * time_block.time), 1000 * (gun_length - 299792458 * time_block.time)], [0, electron_hist_maxs[i]], 'k')
            ax2.set_xlim(electron_mins[i], electron_maxs[i])
            ax2.set_ylim(0, electron_hist_maxs[i])
            ax2.set_xlabel(label)
            os.system(f'mkdir -p {results_folder}/frames/{name}')
            fig.savefig(f'{results_folder}/frames/{name}/t_{j}.png', dpi=200)
            plt.close(fig)
        os.system(f'rm {results_folder}/electrons_{name}.mp4')
        print(f'-> making {name} histogram')
        subprocess.run(f'movie {results_folder}/frames/{name}/t_%d.png {results_folder}/{name}.mp4', check=True, shell=True, capture_output=True)

def analyze2(data1, data2, title_1, title_2, gun_length, results_folder):
    results_folder_path = pathlib.Path(results_folder)
    if not results_folder_path.exists():
        results_folder_path.mkdir(parents=True)
    results_folder_path.resolve(strict=True)
    ts = np.array([[time_block.time for time_block in data.time_blocks] for data in [data1, data2]])
    x_centroid = np.array([[time_block.electrons['x'].mean() for time_block in data.time_blocks] for data in [data1, data2]])
    y_centroid = np.array([[time_block.electrons['y'].mean() for time_block in data.time_blocks] for data in [data1, data2]])
    z_centroid = np.array([[time_block.electrons['z'].mean() for time_block in data.time_blocks] for data in [data1, data2]])
    x_std = np.array([[time_block.electrons['x'].std() for time_block in data.time_blocks] for data in [data1, data2]])
    y_std = np.array([[time_block.electrons['y'].std() for time_block in data.time_blocks] for data in [data1, data2]])
    z_std = np.array([[time_block.electrons['z'].std() for time_block in data.time_blocks] for data in [data1, data2]])
    x_emit = np.array([[np.sqrt(np.linalg.det(np.cov(np.array([
        time_block.electrons['x'],
        time_block.electrons['G'] * time_block.electrons['Bx']
    ])))) for time_block in data.time_blocks] for data in [data1, data2]])
    y_emit = np.array([[np.sqrt(np.linalg.det(np.cov(np.array([
        time_block.electrons['y'],
        time_block.electrons['G'] * time_block.electrons['By']
    ])))) for time_block in data.time_blocks] for data in [data1, data2]])
    emit_6d = np.array([[np.sqrt(np.linalg.det(np.cov(np.array([
        time_block.electrons['x'],
        time_block.electrons['y'],
        time_block.electrons['z'],
        time_block.electrons['G'] * time_block.electrons['Bx'],
        time_block.electrons['G'] * time_block.electrons['By'],
        time_block.electrons['G'] * time_block.electrons['Bz'],
    ])))) for time_block in data.time_blocks] for data in [data1, data2]])
    charge = np.array([[-1.60217662e-19 * len(time_block.electrons['x']) for time_block in data.time_blocks] for data in [data1, data2]])
    screen_index_1 = np.argmin(np.abs(z_centroid[0,:] - gun_length))
    screen_index_2 = np.argmin(np.abs(z_centroid[1,:] - gun_length))

    assert (ts[0, :] == ts[1, :]).all()

    plt.title('At Gun Exit')
    plt.scatter(1e6 * data1.time_blocks[screen_index_1].electrons['x'], data1.time_blocks[screen_index_1].electrons['G'] * data1.time_blocks[screen_index_1].electrons['Bx'], s=0.5, label=title_1)
    plt.scatter(1e6 * data2.time_blocks[screen_index_2].electrons['x'], data2.time_blocks[screen_index_2].electrons['G'] * data2.time_blocks[screen_index_2].electrons['Bx'], s=0.5, label=title_2)
    plt.xlabel('$x$ ($\\mu$m)')
    plt.ylabel('$p_x / m_e c$')
    plt.legend()
    plt.savefig(results_folder_path / 'x_phase_space.png', dpi=300)
    plt.clf()

    plt.title('At Gun Exit')
    plt.scatter(1e6 * data1.time_blocks[screen_index_1].electrons['y'], data1.time_blocks[screen_index_1].electrons['G'] * data1.time_blocks[screen_index_1].electrons['By'], s=0.5, label=title_1)
    plt.scatter(1e6 * data2.time_blocks[screen_index_2].electrons['y'], data2.time_blocks[screen_index_2].electrons['G'] * data2.time_blocks[screen_index_2].electrons['By'], s=0.5, label=title_2)
    plt.xlabel('$y$ ($\\mu$m)')
    plt.ylabel('$p_y / m_e c$')
    plt.legend()
    plt.savefig(results_folder_path / 'y_phase_space.png', dpi=300)
    plt.clf()

    plt.title('At Gun Exit')
    plt.scatter(1e3 * (data1.time_blocks[screen_index_1].electrons['z'] - gun_length), data1.time_blocks[screen_index_1].electrons['G'] * data1.time_blocks[screen_index_1].electrons['Bz'], s=0.5, label=title_1)
    plt.scatter(1e3 * (data2.time_blocks[screen_index_2].electrons['z'] - gun_length), data2.time_blocks[screen_index_2].electrons['G'] * data2.time_blocks[screen_index_2].electrons['Bz'], s=0.5, label=title_2)
    plt.xlabel('$z$ (mm)')
    plt.ylabel('$p_z / m_e c$')
    plt.legend()
    plt.savefig(results_folder_path / 'z_phase_space.png', dpi=300)
    plt.clf()

    plt.plot(ts[0,:] * 1e12, x_centroid[0,:] * 1e6, label=(title_1 + ' x'))
    plt.plot(ts[1,:] * 1e12, x_centroid[1,:] * 1e6, label=(title_2 + ' x'))
    plt.plot(ts[0,:] * 1e12, y_centroid[0,:] * 1e6, label=(title_1 + ' y'))
    plt.plot(ts[1,:] * 1e12, y_centroid[1,:] * 1e6, label=(title_2 + ' y'))
    assert screen_index_1 == screen_index_2
    plt.axvline(x=ts[0, screen_index_1] * 1e12, color='black', label='Gun Exit')
    plt.xlabel('time (ps)')
    plt.ylabel('centroid ($\\mu$m)')
    plt.legend()
    plt.savefig(results_folder_path / 'xy_centroid.png', dpi=300)
    plt.clf()

    plt.plot(ts[0, :] * 1e12, z_centroid[0, :] * 1e3, label=title_1)
    plt.plot(ts[1, :] * 1e12, z_centroid[1, :] * 1e3, label=title_2)
    plt.axvline(x=ts[0, screen_index_1] * 1e12, color='black', label='Gun Exit')
    plt.xlabel('time (ps)')
    plt.ylabel('z centroid (mm)')
    plt.legend()
    plt.savefig(results_folder_path / 'z_centroid.png', dpi=300)
    plt.clf()

    plt.plot(ts[0,:] * 1e12, x_std[0,:] * 1e6, label=(title_1 + ' x'))
    plt.plot(ts[1,:] * 1e12, x_std[1,:] * 1e6, label=(title_2 + ' x'))
    plt.plot(ts[0,:] * 1e12, y_std[0,:] * 1e6, label=(title_1 + ' y'))
    plt.plot(ts[1,:] * 1e12, y_std[1,:] * 1e6, label=(title_2 + ' y'))
    assert screen_index_1 == screen_index_2
    plt.axvline(x=ts[0, screen_index_1] * 1e12, color='black', label='Gun Exit')
    plt.xlabel('time (ps)')
    plt.ylabel('$\\sigma$ ($\\mu$m)')
    plt.legend()
    plt.savefig(results_folder_path / 'xy_sigma.png', dpi=300)
    plt.clf()

    plt.plot(ts[0, :] * 1e12, z_std[0, :] * 1e3, label=title_1)
    plt.plot(ts[1, :] * 1e12, z_std[1, :] * 1e3, label=title_2)
    assert screen_index_1 == screen_index_2
    plt.axvline(x=ts[0, screen_index_1] * 1e12, color='black', label='Gun Exit')
    plt.xlabel('time (ps)')
    plt.ylabel('$\\sigma_z$ (mm)')
    plt.legend()
    plt.savefig(results_folder_path / 'z_sigma.png', dpi=300)
    plt.clf()

    plt.plot(ts[0, :] * 1e12, x_emit[0, :] * 1e6, label=(title_1 + ' x'))
    plt.plot(ts[1, :] * 1e12, x_emit[1, :] * 1e6, label=(title_2 + ' x'))
    plt.plot(ts[0, :] * 1e12, y_emit[0, :] * 1e6, label=(title_1 + ' y'))
    plt.plot(ts[1, :] * 1e12, y_emit[1, :] * 1e6, label=(title_2 + ' y'))
    assert screen_index_1 == screen_index_2
    plt.axvline(x=ts[0, screen_index_1] * 1e12, color='black', label='Gun Exit')
    plt.xlabel('time (ps)')
    plt.ylabel('$\\epsilon_n$ ($\\mu$m)')
    plt.legend()
    plt.savefig(results_folder_path / 'xy_emit.png', dpi=300)
    plt.clf()

    plt.plot(ts[0,:] * 1e12, emit_6d[0,:], label=title_1)
    plt.plot(ts[1,:] * 1e12, emit_6d[1,:], label=title_2)
    assert screen_index_1 == screen_index_2
    plt.axvline(x=ts[0, screen_index_1] * 1e12, color='black', label='Gun Exit')
    plt.xlabel('time (ps)')
    plt.ylabel('$\\epsilon_{6d}$ ($\\mathrm{m}^3$)')
    plt.legend()
    plt.savefig(results_folder_path / '6d_emit.png', dpi=300)
    plt.clf()

    plt.plot(ts[0,:] * 1e12, charge[0, :], label=title_1)
    plt.plot(ts[1,:] * 1e12, charge[1, :], label=title_2)
    assert screen_index_1 == screen_index_2
    plt.axvline(x=ts[0, screen_index_1] * 1e12, color='black', label='Gun Exit')
    plt.xlabel('time (ps)')
    plt.ylabel('$Q$ (C)')
    plt.legend()
    plt.savefig(results_folder_path / 'charge.png', dpi=300)
    plt.clf()

def analyze3(data1, data2, title_1, title_2, gun_length, results_folder):
    names = ('z', 'x', 'y', 'bz', 'bx', 'by', 'g')
    labels = ('$z - ct$ (mm)', '$x$ (mm)', '$y$ (mm)', '$\\beta_z$', '$\\beta_x$', '$\\beta_y$', '$\\gamma$')
    is_center = (False, True, True, False, True, True, False)
    functions = [
        lambda data, time: 1000 * (data['z'] - 299792458 * time),
        lambda data, time: 1000 * data['x'],
        lambda data, time: 1000 * data['y'],
        lambda data, time: data['Bz'],
        lambda data, time: data['Bx'],
        lambda data, time: data['By'],
        lambda data, time: data['G']
    ]

    electron_mins = [min([smart_lower_bound(functions[i](block.electrons, block.time), center=is_center[i]) for block in data1.time_blocks]
                       + [smart_lower_bound(functions[i](block.electrons, block.time), center=is_center[i]) for block in data2.time_blocks]) for i in range(len(names))]
    electron_maxs = [max([smart_upper_bound(functions[i](block.electrons, block.time), center=is_center[i]) for block in data1.time_blocks]
                       + [smart_upper_bound(functions[i](block.electrons, block.time), center=is_center[i]) for block in data2.time_blocks]) for i in range(len(names))]

    for i in range(len(names)):
        name1 = names[i]
        label1 = labels[i]
        function1 = functions[i]
        for j in range(i + 1, len(names)):
            name2 = names[j]
            label2 = labels[j]
            function2 = functions[j]
            for k in range(len(data1.time_blocks)):

                assert data1.time_blocks[k].time == data2.time_blocks[k].time

                fig, (ax1, ax2) = plt.subplots(nrows=1, ncols=2)
                fig.suptitle('t = ' + str(int(round(data1.time_blocks[k].time * 1e12))) + 'ps')
                ax1.set_title(title_1)
                ax1.scatter(function1(data1.time_blocks[k].electrons, data1.time_blocks[k].time), function2(data1.time_blocks[k].electrons, data1.time_blocks[k].time), s=0.5)
                if name1 == 'z':
                    ax1.plot([1000 * (gun_length - 299792458 * data1.time_blocks[k].time), 1000 * (gun_length - 299792458 * data1.time_blocks[k].time)], [electron_mins[j], electron_maxs[j]], 'k')
                if name2 == 'z':
                    ax1.plot([electron_mins[i], electron_maxs[i]], [1000 * (gun_length - 299792458 * data1.time_blocks[k].time), 1000 * (gun_length - 299792458 * data1.time_blocks[k].time)], 'k')
                ax1.set_xlabel(label1)
                ax1.set_ylabel(label2)
                ax1.set_xlim(electron_mins[i], electron_maxs[i])
                ax1.set_ylim(electron_mins[j], electron_maxs[j])
                ax2.set_title(title_2)
                ax2.scatter(function1(data2.time_blocks[k].electrons, data2.time_blocks[k].time), function2(data2.time_blocks[k].electrons, data2.time_blocks[k].time), s=0.5)
                if name1 == 'z':
                    ax2.plot([1000 * (gun_length - 299792458 * data2.time_blocks[k].time), 1000 * (gun_length - 299792458 * data2.time_blocks[k].time)], [electron_mins[j], electron_maxs[j]], 'k')
                if name2 == 'z':
                    ax2.plot([electron_mins[i], electron_maxs[i]], [1000 * (gun_length - 299792458 * data2.time_blocks[k].time), 1000 * (gun_length - 299792458 * data2.time_blocks[k].time)], 'k')
                ax2.set_xlabel(label1)
                ax2.set_xlim(electron_mins[i], electron_maxs[i])
                ax2.set_ylim(electron_mins[j], electron_maxs[j])
                ax2.set_yticklabels([])

                os.system(f'mkdir -p {results_folder}/frames/{name1}_{name2}')
                fig.savefig(f'{results_folder}/frames/{name1}_{name2}/t_{k}.png', dpi=200)
                plt.close(fig)
            os.system(f'rm {results_folder}/{name1}_{name2}.mp4')
            print(f'-> making {name1}-{name2} plot')
            subprocess.run(f'movie {results_folder}/frames/{name1}_{name2}/t_%d.png {results_folder}/{name1}_{name2}.mp4', check=True, shell=True, capture_output=True)

def main():
    s_ions = mot.MotGunSimulation(f'/a/chansel/mot/spacecharge/0')
    s_ions.readData()
    s_no_ions = mot.MotGunSimulation(f'/a/chansel/mot/spacecharge/1')
    s_no_ions.readData()
    os.system('mkdir -p spacecharge.py.results')
    analyze(s_ions.data, s_no_ions.data, 'Approximate', 'Exact', 0.12, 'spacecharge.py.results')
    analyze2(s_ions.data, s_no_ions.data, 'Approximate', 'Exact', 0.12, 'spacecharge.py.results')
    analyze3(s_ions.data, s_no_ions.data, 'Approximate', 'Exact', 0.12, 'spacecharge.py.results')
    os.system('rm -rf spacecharge.py.results/frames')
    os.system('tar -czvf spacecharge.py.results.tgz spacecharge.py.results')


def run_simulation(i):
    print(f'running simulation {i}')
    spacecharge = scan_parameters[i]
    s = mot.MotGunSimulation(f'/a/chansel/mot/spacecharge/{i}')
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
    s.sol_strength = 0.8
    s.sol_length = 0.1
    s.sol_z = 0.1
    s.ions_on = True
    s.spacecharge = spacecharge
    s.initializeParticles(seed=95034)
    s.run()
    s.readData()
    s.analyze()
    return i

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
            with multiprocessing.Pool(processes) as pool:
                results = pool.imap_unordered(run_simulation, range(len(scan_parameters)))
                for result in results:
                    print(f'simulation {result} complete')
            main()
            pathlib.Path(f'{__file__}.kill').unlink()
        except:
            pathlib.Path(f'{__file__}.kill').unlink()
            raise
