import ctypes
import numpy as np
import pathlib
import platform
import subprocess
from .analysis import analyze
from .output import Data

vacuum_permeability = 1.2566370621219e-6

input_file = lambda self: f'''# Simulation Parameters
peak_field = {self.peak_field:.16e}; # [V/m]
rf_frequency = {self.rf_frequency:.16e}; # [Hz]
rf_phase_deg = {self.rf_phase_deg:.16e}; # [deg]
pyramid_height = {self.pyramid_height:.16e}; # [m]
gun_length = {self.gun_length:.16e}; # [m]
sol_z = {self.sol_z:.16e}; # [m]
sol_radius = {self.sol_radius:.16e}; # [m]
sol_length = {self.sol_length:.16e}; # [m]
sol_current = {(2 * self.sol_strength * np.sqrt((0.5 * self.sol_length) ** 2 + self.sol_radius ** 2) / vacuum_permeability):.16e}; # [A]

# Beam
setfile("beam", "{self.folder / 'beam.gdf'}");
{'' if self.ions_on else '#'}setfile("ions", "{self.folder / 'ions.gdf'}");

# Field
rf_phase = rf_phase_deg / deg;
rf_angular_frequency = 2 * pi * rf_frequency;
map3D_TM("wcs", "z", 0, "{self.field_file_path}", "x", "y", "z", "Ex", "Ey", "Ez", "Bx", "By", "Bz", peak_field, rf_phase, rf_angular_frequency);
bzsolenoid("wcs", "z", sol_z, sol_radius, sol_length, sol_current);

# Boundary
rmax("wcs", "I", 0.002);
zminmax("wcs", "I", -pyramid_height, 2 * gun_length);

# Simulation
{'spacecharge3Dmesh' if self.spacecharge == 'approximate' else 'spacecharge3D'}();
snapshot(0, 5e-10, 5e-12);
'''

shared_library_extension = 'dylib' if platform.system() == 'Darwin' else 'so'
sample_cxx_library = ctypes.cdll.LoadLibrary(pathlib.Path(__file__).parent.parent / f'libsample.{shared_library_extension}')

class MotGunSimulation(object):

    parameters = {
        'sigma_r': 'm',
        'sigma_pr': 'mc',
        'mot_z_offset': 'm',
        'laser_phi_x': 'rad',
        'laser_phi_y': 'rad',
        'laser_x': 'm',
        'laser_y': 'm',
        'laser_z': 'm',
        'laser_width': 'm',
        'peak_density': 'm^-3',
        'peak_field': 'V/m',
        'rf_frequency': 'Hz',
        'rf_phase_deg': 'deg',
        'pyramid_height': 'm',
        'gun_length': 'm',
        'sol_radius': 'm',
        'sol_strength': 'T',
        'sol_length': 'm',
        'sol_z': 'm',
        'ions_on': '',
        'spacecharge': ''
    }

    def __init__(self, folder, dict=None):
        self.folder = pathlib.Path(folder)
        if not self.folder.exists():
            self.folder.mkdir(parents=True)
        self.folder.resolve(strict=True)
        self.field_file_path = (pathlib.Path(__file__).parent.parent / 'field.gdf').resolve(strict=True)
        if dict is not None:
            for key, item in dict.items():
                setattr(self, key, item)

    def initializeParticles(self):
        print('-> sampling particles')
        sample_cxx_library.sample(
            ctypes.c_char_p(str(self.folder / 'beam.txt').encode('utf-8')),
            ctypes.c_char_p(str(self.folder / 'ions.txt').encode('utf-8')),
            ctypes.c_double(self.peak_density),
            ctypes.c_double(self.sigma_r),
            ctypes.c_double(self.sigma_pr),
            ctypes.c_double(self.mot_z_offset),
            ctypes.c_double(self.laser_phi_x),
            ctypes.c_double(self.laser_phi_y),
            ctypes.c_double(self.laser_x),
            ctypes.c_double(self.laser_y),
            ctypes.c_double(self.laser_z),
            ctypes.c_double(self.laser_width),
            ctypes.c_bool(self.ions_on)
        )
        print('-> binarizing initial electron distribution')
        subprocess.run(['asci2gdf', '-o', self.folder / 'beam.gdf', self.folder / 'beam.txt'], check=True)
        print('-> deleting initial electron distribution ascii data')
        (self.folder / 'beam.txt').unlink()
        if self.ions_on:
            print('-> binarizing initial ion distribution')
            subprocess.run(['asci2gdf', '-o', self.folder / 'ions.gdf', self.folder / 'ions.txt'], check=True)
            print('-> deleting initial ion distribution ascii data')
            (self.folder / 'ions.txt').unlink()

    def run(self):
        print('-> writing input file')
        with open(self.folder / 'mot.in', 'w+') as f:
            f.write(input_file(self))
        print('-> running gpt')
        subprocess.run(['gpt', '-o', self.folder / 'output.gdf', self.folder / 'mot.in'], check=True)
        print('-> decoding binary gpt output')
        subprocess.run(['gdf2a', '-w32', '-o', self.folder / 'output.txt', self.folder / 'output.gdf'], check=True)
        print('-> deleting binary gpt output')
        (self.folder / 'output.gdf').unlink()

    def readData(self):
        self.data = Data(self.folder / 'output.txt')

    def deleteData(self):
        print('-> deleting initial electron distribution binary data')
        (self.folder / 'beam.gdf').unlink()
        if self.ions_on:
            print('-> deleting initial ion distribution binary data')
            (self.folder / 'beam.gdf').unlink()
        print('-> deleting ascii output data')
        (self.folder / 'output.txt').unlink()

    def analyze(self, scan_dict=None):
        parameters_dict = {key: (getattr(self, key), value) for key, value in self.parameters.items()}
        analyze(self.data, self.gun_length, self.folder / 'results', scan_dict, parameters_dict)
