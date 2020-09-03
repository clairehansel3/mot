#!/usr/bin/env python3
import mot
import numpy as np
import os

IOs = [True, False]
SCs = ['none', 'approximate']

def do_scan(i, path, IO, SC):
    os.system(f'echo "sim={i}, io={IO}, sc={SC}" >> map')
    s = mot.MotGunSimulation(path)
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
    s.ions_on = IO
    s.peak_field = 100e6
    s.rf_frequency = 2.85648e9
    s.rf_phase_deg = 90
    s.pyramid_height = 0.0081
    s.sol_radius = 0.1
    s.sol_strength = -0.48252636
    s.sol_length = 0.01422984
    s.sol_z = 0.18436283
    s.spacecharge = SC
    s.tmax = 1.6e-9
    s.tstep = 1e-11
    s.z_end_of_gun = 0.12
    s.z_end_of_drift = 0.4
    s.initializeParticles(seed=849324)
    s.run()
    s.readData()
    s.analyze()
    s.movie()
    s.plot()
    s.deleteData()
    s.save()
    return i, None

def main():
    mot.run(__file__, do_scan, IOs, SCs, processes=6, detach=True, data_folder_path='/a/chansel/mot')

if __name__ == '__main__':
    main()
