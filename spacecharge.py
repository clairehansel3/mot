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

processes = 2

scan_parameters = ['approximate', 'exact']

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
    s.initializeParticles()
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
            pathlib.Path(f'{__file__}.kill').unlink()
        except:
            pathlib.Path(f'{__file__}.kill').unlink()
            raise
