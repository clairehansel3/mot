import ctypes
import matplotlib.pyplot as plt
import numpy as np
import pathlib
import pickle
import platform
import shutil
import subprocess

vacuum_permeability = 1.2566370621219e-6
elementary_charge = 1.60217662e-19
c_light = 299792458

input_file = lambda self: f'''# Simulation Parameters
peak_field = {self.peak_field:.16e}; # [V/m]
rf_frequency = {self.rf_frequency:.16e}; # [Hz]
rf_phase_deg = {self.rf_phase_deg:.16e}; # [deg]
pyramid_height = {self.pyramid_height:.16e}; # [m]
sol_z = {self.sol_z:.16e}; # [m]
sol_radius = {self.sol_radius:.16e}; # [m]
sol_length = {self.sol_length:.16e}; # [m]
sol_current = {(2 * self.sol_strength * np.sqrt((0.5 * self.sol_length) ** 2 + self.sol_radius ** 2) / (vacuum_permeability * self.sol_length)):.16e}; # [A]

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
zminmax("wcs", "I", -pyramid_height, 1);

# Simulation
{'spacecharge3Dmesh();' if self.spacecharge == 'approximate' else ('spacecharge3D();' if self.spacecharge == 'exact' else '')}
snapshot(0, {self.tmax:.16}, {self.tstep:.16});
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
        'sol_radius': 'm',
        'sol_strength': 'T',
        'sol_length': 'm',
        'sol_z': 'm',
        'ions_on': '',
        'spacecharge': '',
        'tmax': 's',
        'tstep': 's'
    }

    def __init__(self, folder):
        self.folder = pathlib.Path(folder)
        if not self.folder.exists():
            self.folder.mkdir(parents=True)
        self.folder.resolve(strict=True)
        self.field_file_path = (pathlib.Path(__file__).parent.parent / 'field.gdf').resolve(strict=True)
        self.results_folder = self.folder / 'results'
        if not self.results_folder.exists():
            self.results_folder.mkdir()
        self.results_folder.resolve(strict=True)

    def save(self):
        assert not hasattr(self, 'data')
        with open(self.folder / 'object.pickle', 'wb') as f:
            pickle.dump(self, f)

    @staticmethod
    def load(folder):
        path = pathlib.Path(folder)
        path.resolve(strict=True)
        with open(path / 'object.pickle', 'rb') as f:
            return pickle.load(f)

    def initializeParticles(self, seed=0):
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
            ctypes.c_uint(seed),
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

    def deleteDataFiles(self):
        print('-> deleting initial electron distribution binary data')
        (self.folder / 'beam.gdf').unlink()
        if self.ions_on:
            print('-> deleting initial ion distribution binary data')
            (self.folder / 'ions.gdf').unlink()
        print('-> deleting ascii output data')
        (self.folder / 'output.txt').unlink()

    def deleteData(self):
        del self.data

    def analyze(self):
        print('-> analyzing data')
        self.t = np.array([time_block.time for time_block in self.data.time_blocks])
        self.mu_x = np.array([time_block.electrons['x'].mean() for time_block in self.data.time_blocks])
        self.mu_y = np.array([time_block.electrons['y'].mean() for time_block in self.data.time_blocks])
        self.mu_z = np.array([time_block.electrons['z'].mean() for time_block in self.data.time_blocks])
        self.mu_bx = np.array([time_block.electrons['Bx'].mean() for time_block in self.data.time_blocks])
        self.mu_by = np.array([time_block.electrons['By'].mean() for time_block in self.data.time_blocks])
        self.mu_bz = np.array([time_block.electrons['Bz'].mean() for time_block in self.data.time_blocks])
        self.mu_g = np.array([time_block.electrons['G'].mean() for time_block in self.data.time_blocks])
        self.std_x = np.array([time_block.electrons['x'].std() for time_block in self.data.time_blocks])
        self.std_y = np.array([time_block.electrons['y'].std() for time_block in self.data.time_blocks])
        self.std_z = np.array([time_block.electrons['z'].std() for time_block in self.data.time_blocks])
        self.std_r = np.sqrt(self.std_x ** 2 + self.std_y ** 2)
        self.std_bx = np.array([time_block.electrons['Bx'].std() for time_block in self.data.time_blocks])
        self.std_by = np.array([time_block.electrons['By'].std() for time_block in self.data.time_blocks])
        self.std_bz = np.array([time_block.electrons['Bz'].std() for time_block in self.data.time_blocks])
        self.std_g = np.array([time_block.electrons['G'].std() for time_block in self.data.time_blocks])
        self.emit_x = np.array([np.sqrt(np.linalg.det(np.cov(np.array([
            time_block.electrons['x'],
            time_block.electrons['G'] * time_block.electrons['Bx']
        ])))) for time_block in self.data.time_blocks])
        self.emit_y = np.array([np.sqrt(np.linalg.det(np.cov(np.array([
            time_block.electrons['y'],
            time_block.electrons['G'] * time_block.electrons['By']
        ])))) for time_block in self.data.time_blocks])
        self.emit_z = np.array([np.sqrt(np.linalg.det(np.cov(np.array([
            time_block.electrons['z'],
            time_block.electrons['G'] * time_block.electrons['Bz']
        ])))) for time_block in self.data.time_blocks])
        self.emit_xy =  np.array([np.sqrt(np.linalg.det(np.cov(np.array([
            time_block.electrons['x'],
            time_block.electrons['G'] * time_block.electrons['Bx'],
            time_block.electrons['y'],
            time_block.electrons['G'] * time_block.electrons['By'],
        ])))) for time_block in self.data.time_blocks])
        self.emit_xyz =  np.array([np.sqrt(np.linalg.det(np.cov(np.array([
            time_block.electrons['x'],
            time_block.electrons['y'],
            time_block.electrons['z'],
            time_block.electrons['G'] * time_block.electrons['Bx'],
            time_block.electrons['G'] * time_block.electrons['By'],
            time_block.electrons['G'] * time_block.electrons['Bz'],
        ])))) for time_block in self.data.time_blocks])
        self.energy_spread = self.std_g / self.mu_g
        self.charge = np.array([elementary_charge * len(time_block.electrons['x']) for time_block in self.data.time_blocks])
        self.index_end_of_gun = np.argmin(np.abs(self.mu_z - self.z_end_of_gun))
        self.index_end_of_drift = np.argmin(np.abs(self.mu_z - self.z_end_of_drift))
        self.index_waist = (self.index_end_of_gun - 1) + np.argmin(self.std_r[self.index_end_of_gun - 1:self.index_end_of_drift])
        self.z_waist = self.mu_z[self.index_waist]
        print('-> done analyzing data')

    def writeComputedParameters(self):
        data = {
            't': (self.t[self.index_waist], 's'),
            'mu_x': (self.mu_x[self.index_waist], 'm'),
            'mu_y': (self.mu_y[self.index_waist], 'm'),
            'mu_z': (self.mu_z[self.index_waist], 'm'),
            'mu_bx': (self.mu_bx[self.index_waist], ''),
            'mu_by': (self.mu_by[self.index_waist], ''),
            'mu_bz': (self.mu_bz[self.index_waist], ''),
            'mu_g': (self.mu_g[self.index_waist], ''),
            'std_x': (self.std_x[self.index_waist], 'm'),
            'std_y': (self.std_y[self.index_waist], 'm'),
            'spot': (self.std_r[self.index_waist], 'm'),
            'std_z': (self.std_z[self.index_waist], 'm'),
            'std_bx': (self.std_bx[self.index_waist], ''),
            'std_by': (self.std_by[self.index_waist], ''),
            'std_bz': (self.std_bz[self.index_waist], ''),
            'std_g': (self.std_g[self.index_waist], ''),
            'emit_x': (self.emit_x[self.index_waist], 'm'),
            'emit_y': (self.emit_y[self.index_waist], 'm'),
            'emit_4d_sqrt': (np.sqrt(self.emit_xy[self.index_waist]), 'm'),
            'emit_6d': (self.emit_xyz[self.index_waist], 'm^3'),
            'espread': (100 * self.energy_spread[self.index_waist], '%'),
            'charge': (self.charge[self.index_waist], 'C')
        }
        with open(self.results_folder / 'data.txt', 'w+') as f:
            for key, (value, unit) in data.items():
                f.write(f'{key}: {value:.16e} {unit}\n')

    def plot(self):
        print('-> plotting')
        eog = ('gun_exit', 'At Gun Exit ($z = {:.2f}$cm)'.format(self.z_end_of_gun * 100), self.index_end_of_gun)
        wst = ('waist', 'At Beam Waist ($z = {:.2f}$cm)'.format(self.z_waist * 100), self.index_waist)
        eod = ('end_of_drift', 'End of Drift ($z = {:.2f}$cm)'.format(self.z_end_of_drift * 100), self.index_end_of_drift)
        if hasattr(self, 'data'):
            for i, (name, title, index) in enumerate([eog, wst, eod]):
                xx = 1e6 * self.data.time_blocks[index].electrons['x']
                xy = self.data.time_blocks[index].electrons['G'] * self.data.time_blocks[index].electrons['Bx']
                plt.title(title)
                plt.scatter(xx, xy, s=0.5, alpha=0.2)
                plt.xlabel('$x$ ($\\mu$m)')
                plt.ylabel('$p_x / m_e c$')
                plt.savefig(self.results_folder / f'x_phase_space_{i}_{name}.png', dpi=300)
                plt.clf()
                yx = 1e6 * self.data.time_blocks[index].electrons['y']
                yy = self.data.time_blocks[index].electrons['G'] * self.data.time_blocks[index].electrons['By']
                plt.title(title)
                plt.scatter(yx, yy, s=0.5, alpha=0.2)
                plt.xlabel('$y$ ($\\mu$m)')
                plt.ylabel('$p_y / m_e c$')
                plt.savefig(self.results_folder / f'y_phase_space_{i}_{name}.png', dpi=300)
                plt.clf()
                zx = 1e3 * (self.data.time_blocks[index].electrons['z'] - self.mu_z[index])
                zy = self.data.time_blocks[index].electrons['G'] * self.data.time_blocks[index].electrons['Bz']
                plt.title(title)
                plt.scatter(zx, zy, s=0.5, alpha=0.2)
                plt.xlabel('$z - \\overline{z}$ (mm)')
                plt.ylabel('$p_z / m_e c$')
                plt.savefig(self.results_folder / f'z_phase_space_{i}_{name}.png', dpi=300)
                plt.clf()
        else:
            print('-> skipping phase space plot (not stored)')

        plot_infos = (
            ([(self.mu_x * 1e6, 'x'), (self.mu_y * 1e6, 'y')], 'centroid ($\\mu$m)', 'mu_xy'),
            ([(self.mu_z * 1e3, 'z')], 'centroid (mm)', 'mu_z'),
            ([(self.std_x * 1e6, 'x'), (self.std_y * 1e6, 'y'), (self.std_r * 1e6, 'r')], '$\\sigma$ ($\\mu$m)', 'std_xyr'),
            ([(self.std_z * 1e6, 'z')], '$\\sigma$ ($\\mu$m)', 'std_z'),
            ([(self.emit_x * 1e9, 'x'), (self.emit_y * 1e9, 'y'), (np.sqrt(self.emit_xy) * 1e9, '4D (sqrt)')], '$\\epsilon_n$ (nm)', 'emit_xy4'),
            ([(self.emit_z * 1e6, 'z')], '$\\epsilon_n$ ($\\mu$m)', 'emit_z'),
            ([(self.emit_xyz,)], '$\\epsilon_{\\mathrm{6D}}$ ($\\mathrm{m}^3$)', 'emit_6'),
            ([(self.energy_spread * 100,)], '$\\Delta E / E$ (%)', 'energy_spread'),
            ([(self.charge * 1e15,)], 'Charge (fC)', 'q'),

        )

        for plot_info in plot_infos:
            for line in plot_info[0]:
                if len(line) == 1:
                    plt.plot(self.t * 1e12, line[0])
                elif len(line) == 2:
                    plt.plot(self.t * 1e12, line[0], label=line[1])
                else:
                    assert False
            plt.axvline(x=self.t[self.index_end_of_gun] * 1e12, color='black', linestyle='-', label='Gun Exit')
            plt.axvline(x=self.t[self.index_waist] * 1e12, color='black', linestyle='--', label='Waist')
            plt.xlabel('time (ps)')
            plt.ylabel(plot_info[1])
            plt.legend()
            plt.savefig(self.results_folder / ('t_' + plot_info[2] + '.png'), dpi=300)
            plt.clf()

        for plot_info in plot_infos:
            for line in plot_info[0]:
                if len(line) == 1:
                    plt.plot(self.mu_z * 100, line[0])
                elif len(line) == 2:
                    plt.plot(self.mu_z * 100, line[0], label=line[1])
                else:
                    assert False
            plt.axvline(x=self.z_end_of_gun * 100, color='black', linestyle='-', label='Gun Exit')
            plt.axvline(x=self.z_waist * 100, color='black', linestyle='--', label='Waist')
            plt.xlabel('$z$ (cm)')
            plt.ylabel(plot_info[1])
            plt.legend()
            plt.savefig(self.results_folder / ('z_' + plot_info[2] + '.png'), dpi=300)
            plt.clf()

        print('-> done plotting')

    def movie(self, bins=100):
        if hasattr(self, 'data'):
            print('-> plotting frames')
            for name in ('x', 'y', 'z'):
                if not (self.results_folder / 'frames' / name).exists():
                    (self.results_folder / 'frames' / name).mkdir(parents=True)
                get_x = lambda block: (block.electrons[name] if name != 'z' else block.electrons[name] - block.electrons[name].mean())
                get_p = lambda block: (block.electrons['G'] * block.electrons['B' + name])
                lb = lambda data: data.mean() - 4 * data.std()
                ub = lambda data: data.mean() + 4 * data.std()
                x_min = min(lb(get_x(block)) for block in self.data.time_blocks)
                x_max = max(ub(get_x(block)) for block in self.data.time_blocks)
                p_min = min(lb(get_p(block)) for block in self.data.time_blocks)
                p_max = max(ub(get_p(block)) for block in self.data.time_blocks)
                hist_values = []
                vmax = None
                for block in self.data.time_blocks:
                    H, x_edges, y_edges = np.histogram2d(get_x(block), get_p(block), bins=bins, range=((x_min, x_max), (p_min, p_max)))
                    vmax = H.max() if vmax is None else max((vmax, H.max()))
                    hist_values.append((H, x_edges, y_edges))
                for index, t in enumerate(self.t):
                    plt.title(f'$t = {1e12 * t:.2f}$ ps, z = {1e3 * self.mu_z[index]:.2f}mm')
                    H, x_edges, y_edges = hist_values[index]
                    X, Y = np.meshgrid(x_edges, y_edges)
                    plt.pcolormesh(X * (1e3 if name == 'z' else 1e6), Y, H.T, vmin=0, vmax=vmax)
                    if name != 'z':
                        plt.xlabel(f'${name}$ ($\\mu$m)')
                    else:
                        plt.xlabel('$z - \\overline{z}$ (mm)')
                    plt.ylabel(f'$p_{name} / m_e c$')
                    plt.savefig(self.results_folder / 'frames' / name / f'{index}.png', dpi=300)
                    plt.clf()
                print(f'-> ffmpeg {name}')
                subprocess.run(('ffmpeg', '-i', str(self.results_folder / 'frames' / name) + '/%d.png', '-c:v', 'libx264', '-pix_fmt', 'yuv420p', str(self.results_folder / (name + '.mp4'))), check=True, capture_output=True)
                print(f'-> done ffmpeg {name}')
                shutil.rmtree(str(self.results_folder / 'frames' / name))
            shutil.rmtree(str(self.results_folder / 'frames'))
            print('-> done plotting frames')
        else:
            print('-> skipping movie (phase space not stored)')

class Block(object):

    def __init__(self, f, time):
        self.time = time
        column_names = f.readline().split()
        rows = []
        for line in f:
            if line.isspace():
                break
            rows.append([float(value) for value in line.split()])
        assert all([len(row) == len(column_names) for row in rows])
        column_data = map(np.array, map(list, zip(*rows)))
        if len(rows) == 0:
            self.data = {key: np.array([]) for key in column_names}
        else:
            self.data = dict(zip(column_names, column_data))
        self.particles = len(rows)
        electrons_mask = (self.data['q'] < 0)
        ions_mask = (self.data['q'] > 0)
        self.electrons = {key:value[electrons_mask] for key, value in self.data.items()}
        self.ions = {key:value[ions_mask] for key, value in self.data.items()}

    def __getitem__(self, *args, **kwargs):
        return self.data.__getitem__(*args, **kwargs)

    def __repr__(self):
        return 'Time Block (t = {:.2f} ps)\nColumns: '.format(self.time * 1e12) + ' '.join(self.data.keys()) + '\nParticles: ' + str(self.particles) + '\n'

class Data(object):

    def __init__(self, filename):
        print('-> reading data')
        self.filename = filename
        self.blocks = []
        self.times = []
        self.time_blocks = []
        self.positions = []
        self.position_blocks = []
        with open(filename, 'r') as f:
            f.readline()
            f.readline()
            self.cputime = float(f.readline().split()[1])
            for line in f:
                assert line.split()[0] == 'time'
                #print(f'reading time block {len(self.time_blocks) + 1}, t = {float(line.split()[1])*1e12:.1f}ps')
                block = Block(f, float(line.split()[1]))
                self.blocks.append(block)
                self.times.append(block.time)
                self.time_blocks.append(block)
        print('-> done reading data')

    def __repr__(self):
        return f'Data\nBlocks: {len(self.blocks)}\n' + ''.join(block.__repr__() for block in self.blocks)
