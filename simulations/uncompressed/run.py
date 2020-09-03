#!/usr/bin/env python3

import tempfile
import os
import mot
import numpy as np
import pygmo as pg
import matplotlib.pyplot as plt
import daemon
import pathlib
import shutil
import signal
import subprocess
import multiprocessing

class MotGunOptimizationProblem(object):

    def __init__(self, dir):
        self.dir = dir

    def fitness(self, x):
        print('thread: ', multiprocessing.current_process().name, flush=True)
        with tempfile.TemporaryDirectory(dir=self.dir) as dir:
            s = mot.MotGunSimulation(dir)
            s.peak_density = 2e16
            s.sigma_r = 0.0005
            s.sigma_pr = 1.9424e-26
            s.mot_z_offset = x[0] #-0.0005
            s.laser_phi_x = 0
            s.laser_phi_y = 0.5 * np.pi / 180
            s.laser_x = 0
            s.laser_y = 0
            s.laser_z = 0
            s.laser_width = 25e-6
            s.ions_on = False
            s.peak_field = 100e6
            s.rf_frequency = 2.85648e9
            s.rf_phase_deg = x[1] #90
            s.pyramid_height = 0.0081
            s.sol_radius = 0.1
            s.sol_strength = -0.48252636
            s.sol_length = 0.01422984
            s.sol_z = 0.18436283
            s.spacecharge = 'none'
            s.tmax = 1.6e-9
            s.tstep = 1e-11
            s.z_end_of_gun = 0.12
            s.z_end_of_drift = 0.4
            s.initializeParticles()
            s.run()
            s.readData()
            s.deleteDataFiles()
            s.analyze()
        return [np.sqrt(s.emit_xy[s.index_end_of_gun]) * 1e9, -np.abs(s.charge[s.index_end_of_gun]) * 1e15]

    def get_nobj(self):
        return 2

    def get_bounds(self):
        return ([-0.001, 60], [0.0, 120])

def optimize(results, dir):
    pg.mp_bfe.init_pool(50)
    prob = pg.problem(MotGunOptimizationProblem(dir))
    bfe = pg.bfe(pg.mp_bfe())
    nsga2 = pg.nsga2()
    nsga2.set_bfe(bfe)
    algo = pg.algorithm(nsga2)
    pop = pg.population(prob=prob, size=256, b=bfe)
    iteration = 1
    while True:
        print(f"\033[31mITERATION: {iteration}\033[m")
        plt.title(f'Iteration {iteration}')
        plt.xlabel('Emittance (nm)')
        plt.ylabel('Charge (fC)')
        pg.plot_non_dominated_fronts(pop.get_f())
        plt.savefig(results / f'{iteration}.png', dpi=300)
        plt.clf()
        assert len(pop.get_x()) == len(pop.get_f())
        with open(results / f'iteration_{iteration}.txt', 'w+') as f:
            f.write('[Mot Z Offset (mm), Phase (deg)] -> [Emittance 4D sqrt (nm), Charge (fC)]\n')
            for i in range(len(pop.get_x())):
                f.write('{} -> {}\n'.format(pop.get_x()[i], pop.get_f()[i]))
        pop = algo.evolve(pop)
        iteration += 1

def main():
    with tempfile.TemporaryDirectory(dir='/b/chansel/mot/tmp') as dir:
        script = pathlib.Path(__file__).resolve(strict=True)
        simulation_name = script.parent.name
        log = script.parent / 'log'
        kill_script = script.parent / 'kill'
        results = script.parent / 'results'
        if kill_script.is_file():
            raise Exception(f'Simulation {simulation_name} is already running')
        if results.exists():
            shutil.rmtree(results)
        results.mkdir()
        print(f'Running simulation {simulation_name}')
        with open(log, 'w', buffering=1) as f:
            with daemon.DaemonContext(
                stdout=f,
                stderr=f,
                working_directory=script.parent
            ):
                with open(kill_script, 'w+') as f:
                    f.write(f'#!/bin/bash\n')
                    f.write(f'rm -rf {dir}\n')
                    f.write(f'kill -SIGINT {os.getpid()}\n')
                try:
                    subprocess.run(['chmod', '+x', kill_script], check=True)
                    optimize(results, dir)
                    kill_script.unlink()
                except:
                    kill_script.unlink()
                    raise

if __name__ == '__main__':
    main()
