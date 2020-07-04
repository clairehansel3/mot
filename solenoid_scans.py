#!/usr/bin/env python3

import daemon
import itertools
import mot
import multiprocessing
import numpy as np
import os
import pathlib
import subprocess
import sys

processes = 25

magnetic_field_strengths = [0.8] #np.linspace(0.2, 0.8, 10, endpoint=True)
solenoid_lengths = np.linspace(0.05, 0.2, 25, endpoint=True)
solenoid_positions = np.linspace(-0.0081, 0.12, 25, endpoint=True)
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
        except:
            pathlib.Path(f'{__file__}.kill').unlink()
            raise
