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
    plt.ylabel('$\\epsilon_{6d}$ ($\\mathrm{m}^3$)')
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


r'''
def dualCorrelationMovie(data_1, data_2, title_1, title_2):

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

    electron_mins = [min([smart_lower_bound(functions[i](block.electrons, block.time), center=is_center[i]) for block in self.time_blocks]) for i in range(len(names))]
    ion_mins = [min([smart_lower_bound(functions[i](block.ions, 0.0), center=is_center[i]) for block in self.time_blocks]) for i in range(len(names))]
    electron_maxs = [max([smart_upper_bound(functions[i](block.electrons, block.time), center=is_center[i]) for block in self.time_blocks]) for i in range(len(names))]
    ion_maxs = [max([smart_upper_bound(functions[i](block.ions, 0.0), center=is_center[i]) for block in self.time_blocks]) for i in range(len(names))]

    electron_mins = [min([smart_lower_bound(functions[i](block.electrons, block.time), center=is_center[i]) for block in self.time_blocks]) for i in range(len(names))]
    ion_mins = [min([smart_lower_bound(functions[i](block.ions, 0.0), center=is_center[i]) for block in self.time_blocks]) for i in range(len(names))]
    electron_maxs = [max([smart_upper_bound(functions[i](block.electrons, block.time), center=is_center[i]) for block in self.time_blocks]) for i in range(len(names))]
    ion_maxs = [max([smart_upper_bound(functions[i](block.ions, 0.0), center=is_center[i]) for block in self.time_blocks]) for i in range(len(names))]

    for i in range(len(names)):
        name1 = names[i]
        label1 = labels[i]
        function1 = functions[i]
        for j in range(i + 1, len(names)):
            name2 = names[j]
            label2 = labels[j]
            function2 = functions[j]
            for k, time_block in enumerate(self.time_blocks):
                plt.title('Electrons, t = ' + str(int(round(time_block.time * 1e12))) + 'ps')
                plt.scatter(function1(time_block.electrons, time_block.time), function2(time_block.electrons, time_block.time), s=0.5)
                if name1 == 'z':
                    plt.plot([1000 * (gun_length - 299792458 * time_block.time), 1000 * (gun_length - 299792458 * time_block.time)], [electron_mins[j], electron_maxs[j]], 'k')
                if name2 == 'z':
                    plt.plot([electron_mins[i], electron_maxs[i]], [1000 * (gun_length - 299792458 * time_block.time), 1000 * (gun_length - 299792458 * time_block.time)], 'k')
                plt.xlabel(label1)
                plt.ylabel(label2)
                plt.xlim(electron_mins[i], electron_maxs[i])
                plt.ylim(electron_mins[j], electron_maxs[j])
                os.system(f'mkdir -p {results_folder}/frames/electrons/{name1}_{name2}')
                plt.savefig(f'{results_folder}/frames/electrons/{name1}_{name2}/t_{k}.png', dpi=200)
                plt.clf()
                plt.title('Ions, t = ' + str(int(round(time_block.time * 1e12))) + 'ps')
                plt.scatter(function1(time_block.ions, 0.0), function2(time_block.ions, 0.0), s=0.5)
                plt.xlabel('z (mm)' if label1 == '$z - ct$ (mm)' else label1)
                plt.ylabel('z (mm)' if label2 == '$z - ct$ (mm)' else label2)
                plt.xlim(ion_mins[i], ion_maxs[i])
                plt.ylim(ion_mins[j], ion_maxs[j])
                os.system(f'mkdir -p {results_folder}/frames/ions/{name1}_{name2}')
                plt.savefig(f'{results_folder}/frames/ions/{name1}_{name2}/t_{k}.png', dpi=200)
                plt.clf()
            os.system(f'rm {results_folder}/electrons_{name1}_{name2}.mp4')
            os.system(f'rm {results_folder}/ions_{name1}_{name2}.mp4')
            print(f'-> making {name1}-{name2} plot')
            subprocess.run(f'movie {results_folder}/frames/electrons/{name1}_{name2}/t_%d.png {results_folder}/electrons_{name1}_{name2}.mp4', check=True, shell=True, capture_output=True)
            subprocess.run(f'movie {results_folder}/frames/ions/{name1}_{name2}/t_%d.png {results_folder}/ions_{name1}_{name2}.mp4', check=True, shell=True, capture_output=True)














def histogramMovie(data, gun_length, results_folder):
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

    electron_mins = [min([smart_lower_bound(functions[i](block.electrons, block.time), center=is_center[i]) for block in self.time_blocks]) for i in range(len(names))]
    ion_mins = [min([smart_lower_bound(functions[i](block.ions, 0.0), center=is_center[i]) for block in self.time_blocks]) for i in range(len(names))]
    electron_maxs = [max([smart_upper_bound(functions[i](block.electrons, block.time), center=is_center[i]) for block in self.time_blocks]) for i in range(len(names))]
    ion_maxs = [max([smart_upper_bound(functions[i](block.ions, 0.0), center=is_center[i]) for block in self.time_blocks]) for i in range(len(names))]

    for block in self.time_blocks:
        print('time: ', block.time)
        print(functions[0](block.electrons, block.time))
        print(smart_lower_bound(functions[0](block.electrons, block.time), center=is_center[0]))
        print(smart_upper_bound(functions[0](block.electrons, block.time), center=is_center[0]))
        print('done')

    electron_hists = [[np.histogram(functions[i](block.electrons, block.time), range=(electron_mins[i], electron_maxs[i]), bins=200) for block in self.time_blocks] for i in range(len(names))]
    ion_hists = [[np.histogram(functions[i](block.ions, 0.0), range=(ion_mins[i], ion_maxs[i]), bins=200) for block in self.time_blocks] for i in range(len(names))]
    electron_hist_maxs = [max([hist[i][0].max() for i in range(len(self.time_blocks))]) for hist in electron_hists]
    ion_hist_maxs = [max([hist[i][0].max() for i in range(len(self.time_blocks))]) for hist in ion_hists]

    for i in range(len(names)):
        name = names[i]
        label = labels[i]
        function = functions[i]
        for j, time_block in enumerate(self.time_blocks):
            e_counts, e_bins = electron_hists[i][j]
            i_counts, i_bins = ion_hists[i][j]
            plt.title('Electrons, t = ' + str(int(round(time_block.time * 1e12))) + 'ps')
            plt.hist(0.5 * (e_bins[:-1] + e_bins[1:]), e_bins, weights=e_counts)
            if name == 'z':
                plt.plot([1000 * (gun_length - 299792458 * time_block.time), 1000 * (gun_length - 299792458 * time_block.time)], [0, electron_hist_maxs[i]], 'k')
            plt.xlim(electron_mins[i], electron_maxs[i])
            plt.ylim(0, electron_hist_maxs[i])
            plt.xlabel(label)
            os.system(f'mkdir -p {results_folder}/frames/electrons/{name}')
            plt.savefig(f'{results_folder}/frames/electrons/{name}/t_{j}.png', dpi=200)
            plt.clf()
            plt.title('Ions, t = ' + str(int(round(time_block.time * 1e12))) + 'ps')
            plt.hist(0.5 * (i_bins[:-1] + i_bins[1:]), i_bins, weights=i_counts)
            plt.xlim(ion_mins[i], ion_maxs[i])
            plt.ylim(0, ion_hist_maxs[i])
            plt.xlabel('z (mm)' if label == '$z - ct$ (mm)' else label)
            os.system(f'mkdir -p {results_folder}/frames/ions/{name}')
            plt.savefig(f'{results_folder}/frames/ions/{name}/t_{j}.png', dpi=200)
            plt.clf()
        os.system(f'rm {results_folder}/electrons_{name}.mp4')
        os.system(f'rm {results_folder}/ions_{name}.mp4')
        print(f'-> making {name} histogram')
        subprocess.run(f'movie {results_folder}/frames/electrons/{name}/t_%d.png {results_folder}/electrons_{name}.mp4', check=True, shell=True, capture_output=True)
        subprocess.run(f'movie {results_folder}/frames/ions/{name}/t_%d.png {results_folder}/ions_{name}.mp4', check=True, shell=True, capture_output=True)

def correlationMovie(self, gun_length, results_folder='.', smart=False):
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

    if smart:

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

        electron_mins = [min([smart_lower_bound(functions[i](block.electrons, block.time), center=is_center[i]) for block in self.time_blocks]) for i in range(len(names))]
        ion_mins = [min([smart_lower_bound(functions[i](block.ions, 0.0), center=is_center[i]) for block in self.time_blocks]) for i in range(len(names))]
        electron_maxs = [max([smart_upper_bound(functions[i](block.electrons, block.time), center=is_center[i]) for block in self.time_blocks]) for i in range(len(names))]
        ion_maxs = [max([smart_upper_bound(functions[i](block.ions, 0.0), center=is_center[i]) for block in self.time_blocks]) for i in range(len(names))]

    else:

        electron_mins = [min([functions[i](block.electrons, block.time).min() for block in self.time_blocks]) for i in range(len(names))]
        ion_mins = [min([functions[i](block.ions, 0.0).min() for block in self.time_blocks]) for i in range(len(names))]
        electron_maxs = [max([functions[i](block.electrons, block.time).max() for block in self.time_blocks]) for i in range(len(names))]
        ion_maxs = [max([functions[i](block.ions, 0.0).max() for block in self.time_blocks]) for i in range(len(names))]

    for i in range(len(names)):
        name1 = names[i]
        label1 = labels[i]
        function1 = functions[i]
        for j in range(i + 1, len(names)):
            name2 = names[j]
            label2 = labels[j]
            function2 = functions[j]
            for k, time_block in enumerate(self.time_blocks):
                plt.title('Electrons, t = ' + str(int(round(time_block.time * 1e12))) + 'ps')
                plt.scatter(function1(time_block.electrons, time_block.time), function2(time_block.electrons, time_block.time), s=0.5)
                if name1 == 'z':
                    plt.plot([1000 * (gun_length - 299792458 * time_block.time), 1000 * (gun_length - 299792458 * time_block.time)], [electron_mins[j], electron_maxs[j]], 'k')
                if name2 == 'z':
                    plt.plot([electron_mins[i], electron_maxs[i]], [1000 * (gun_length - 299792458 * time_block.time), 1000 * (gun_length - 299792458 * time_block.time)], 'k')
                plt.xlabel(label1)
                plt.ylabel(label2)
                plt.xlim(electron_mins[i], electron_maxs[i])
                plt.ylim(electron_mins[j], electron_maxs[j])
                os.system(f'mkdir -p {results_folder}/frames/electrons/{name1}_{name2}')
                plt.savefig(f'{results_folder}/frames/electrons/{name1}_{name2}/t_{k}.png', dpi=200)
                plt.clf()
                plt.title('Ions, t = ' + str(int(round(time_block.time * 1e12))) + 'ps')
                plt.scatter(function1(time_block.ions, 0.0), function2(time_block.ions, 0.0), s=0.5)
                plt.xlabel('z (mm)' if label1 == '$z - ct$ (mm)' else label1)
                plt.ylabel('z (mm)' if label2 == '$z - ct$ (mm)' else label2)
                plt.xlim(ion_mins[i], ion_maxs[i])
                plt.ylim(ion_mins[j], ion_maxs[j])
                os.system(f'mkdir -p {results_folder}/frames/ions/{name1}_{name2}')
                plt.savefig(f'{results_folder}/frames/ions/{name1}_{name2}/t_{k}.png', dpi=200)
                plt.clf()
            os.system(f'rm {results_folder}/electrons_{name1}_{name2}.mp4')
            os.system(f'rm {results_folder}/ions_{name1}_{name2}.mp4')
            print(f'-> making {name1}-{name2} plot')
            subprocess.run(f'movie {results_folder}/frames/electrons/{name1}_{name2}/t_%d.png {results_folder}/electrons_{name1}_{name2}.mp4', check=True, shell=True, capture_output=True)
            subprocess.run(f'movie {results_folder}/frames/ions/{name1}_{name2}/t_%d.png {results_folder}/ions_{name1}_{name2}.mp4', check=True, shell=True, capture_output=True)
'''
