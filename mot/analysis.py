import pickle
import matplotlib.pyplot as plt
import numpy as np
import pathlib

def analyze(data, gun_length, results_folder, parameters_dict):
    results_folder_path = pathlib.Path(results_folder)
    if not results_folder_path.exists():
        results_folder_path.mkdir(parents=True)
    results_folder_path.resolve(strict=True)
    ts = np.array([time_block.time for time_block in data.time_blocks])
    x_centroid = np.array([time_block.electrons['x'].mean() for time_block in data.time_blocks])
    y_centroid = np.array([time_block.electrons['y'].mean() for time_block in data.time_blocks])
    z_centroid = np.array([time_block.electrons['z'].mean() for time_block in data.time_blocks])
    x_std = np.array([time_block.electrons['x'].std() for time_block in data.time_blocks])
    y_std = np.array([time_block.electrons['y'].std() for time_block in data.time_blocks])
    z_std = np.array([time_block.electrons['z'].std() for time_block in data.time_blocks])
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
    charge = np.array([-1.60217662e-19 * len(time_block.electrons['x']) for time_block in data.time_blocks])
    screen_index = np.argmin(np.abs(z_centroid - gun_length))
    with open(results_folder_path / 'data.pickle', 'wb') as f:
        data_dictionary = {
            'parameters': parameters_dict,
            'computed_values': {
                'time_index': (screen_index, ''),
                'time': (ts[screen_index], 's'),
                'x_centroid': (x_centroid[screen_index], 'm'),
                'y_centroid': (y_centroid[screen_index], 'm'),
                'z_centroid': (z_centroid[screen_index], 'm'),
                'x_std': (x_std[screen_index], 'm'),
                'y_std': (y_std[screen_index], 'm'),
                'z_std': (z_std[screen_index], 'm'),
                'x_emit': (x_emit[screen_index], 'm'),
                'y_emit': (y_emit[screen_index], 'm'),
                'emit_6d': (emit_6d[screen_index], 'm^3'),
                'charge': (charge[screen_index], 'C')
            }
        }
        pickle.dump(data_dictionary, f)

    plt.title('At Gun Exit')
    plt.scatter(1e6 * data.time_blocks[screen_index].electrons['x'], data.time_blocks[screen_index].electrons['G'] * data.time_blocks[screen_index].electrons['Bx'], s=0.5)
    plt.xlabel('$x$ ($\\mu$m)')
    plt.ylabel('$p_x / m_e c$')
    plt.savefig(results_folder_path / 'x_phase_space.png', dpi=300)
    plt.clf()

    plt.title('At Gun Exit')
    plt.scatter(1e6 * data.time_blocks[screen_index].electrons['y'], data.time_blocks[screen_index].electrons['G'] * data.time_blocks[screen_index].electrons['By'], s=0.5)
    plt.xlabel('$y$ ($\\mu$m)')
    plt.ylabel('$p_y / m_e c$')
    plt.savefig(results_folder_path / 'y_phase_space.png', dpi=300)
    plt.clf()

    plt.title('At Gun Exit')
    plt.scatter(1e3 * (data.time_blocks[screen_index].electrons['z'] - gun_length), data.time_blocks[screen_index].electrons['G'] * data.time_blocks[screen_index].electrons['Bz'], s=0.5)
    plt.xlabel('$z$ (mm)')
    plt.ylabel('$p_z / m_e c$')
    plt.savefig(results_folder_path / 'z_phase_space.png', dpi=300)
    plt.clf()

    plt.plot(ts * 1e12, x_centroid * 1e6, label='x')
    plt.plot(ts * 1e12, y_centroid * 1e6, label='y')
    plt.axvline(x=ts[screen_index] * 1e12, color='black', label='Gun Exit')
    plt.xlabel('time (ps)')
    plt.ylabel('centroid ($\\mu$m)')
    plt.legend()
    plt.savefig(results_folder_path / 'xy_centroid.png', dpi=300)
    plt.clf()

    plt.plot(ts * 1e12, z_centroid * 1e3)
    plt.axvline(x=ts[screen_index] * 1e12, color='black', label='Gun Exit')
    plt.xlabel('time (ps)')
    plt.ylabel('z centroid (mm)')
    plt.legend()
    plt.savefig(results_folder_path / 'z_centroid.png', dpi=300)
    plt.clf()

    plt.plot(ts * 1e12, x_std * 1e6, label='x')
    plt.plot(ts * 1e12, y_std * 1e6, label='y')
    plt.axvline(x=ts[screen_index] * 1e12, color='black', label='Gun Exit')
    plt.xlabel('time (ps)')
    plt.ylabel('$\\sigma$ ($\\mu$m)')
    plt.legend()
    plt.savefig(results_folder_path / 'xy_sigma.png', dpi=300)
    plt.clf()

    plt.plot(ts * 1e12, z_std * 1e3)
    plt.axvline(x=ts[screen_index] * 1e12, color='black', label='Gun Exit')
    plt.xlabel('time (ps)')
    plt.ylabel('$\\sigma_z$ (mm)')
    plt.legend()
    plt.savefig(results_folder_path / 'z_sigma.png', dpi=300)
    plt.clf()

    plt.plot(ts * 1e12, x_emit * 1e6, label='x')
    plt.plot(ts * 1e12, y_emit * 1e6, label='y')
    plt.axvline(x=ts[screen_index] * 1e12, color='black', label='Gun Exit')
    plt.xlabel('time (ps)')
    plt.ylabel('$\\epsilon_n$ ($\\mu$m)')
    plt.legend()
    plt.savefig(results_folder_path / 'xy_emit.png', dpi=300)
    plt.clf()

    plt.plot(ts * 1e12, emit_6d)
    plt.axvline(x=ts[screen_index] * 1e12, color='black', label='Gun Exit')
    plt.xlabel('time (ps)')
    plt.ylabel('$\\epsilon_{6d}$ ($\\mathrm{m}^2$)')
    plt.legend()
    plt.savefig(results_folder_path / '6d_emit.png', dpi=300)
    plt.clf()

    plt.plot(ts * 1e12, charge)
    plt.axvline(x=ts[screen_index] * 1e12, color='black', label='Gun Exit')
    plt.xlabel('time (ps)')
    plt.ylabel('$Q$ (C)')
    plt.legend()
    plt.savefig(results_folder_path / 'charge.png', dpi=300)
    plt.clf()
