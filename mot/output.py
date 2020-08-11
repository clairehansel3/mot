import numpy as np

class Block(object):

    def __init__(self, f):
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
        return 'Block\nColumns: ' + ' '.join(self.data.keys()) + '\nParticles: ' + str(self.particles) + '\n'

class PositionBlock(Block):

    def __init__(self, f, position):
        self.position = position
        super().__init__(f)

    def __repr__(self):
        return 'Position Block (z = {:.2f} mm)\nColumns: '.format(self.position * 1000) + ' '.join(self.data.keys()) + '\nParticles: ' + str(self.particles) + '\n'

class TimeBlock(Block):

    def __init__(self, f, time):
        self.time = time
        super().__init__(f)

    def __repr__(self):
        return 'Time Block (t = {:.2f} ps)\nColumns: '.format(self.time * 1e12) + ' '.join(self.data.keys()) + '\nParticles: ' + str(self.particles) + '\n'

def getTimeBlocks(filename):
    with open(filename, 'r') as f:
        f.readline()
        f.readline()
        f.readline()
        for line in f:
            assert line.split()[0] == 'time'
            print(f'reading time block t = {float(line.split()[1]):.2e} s')
            yield TimeBlock(f, float(line.split()[1]))

class Data(object):

    def __init__(self, filename):
        print('--> reading data')
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
                if line.split()[0] == 'position':
                    print(f'reading time block {len(self.position_blocks) + 1}, z = {float(line.split()[1])*1e3:.1f}mm')
                    block = PositionBlock(f, float(line.split()[1]))
                    self.blocks.append(block)
                    self.positions.append(block.position)
                    self.position_blocks.append(block)
                elif line.split()[0] == 'time':
                    print(f'reading time block {len(self.time_blocks) + 1}, t = {float(line.split()[1])*1e12:.1f}ps')
                    block = TimeBlock(f, float(line.split()[1]))
                    self.blocks.append(block)
                    self.times.append(block.time)
                    self.time_blocks.append(block)

    def __repr__(self):
        return f'Data\nBlocks: {len(self.blocks)}\n' + ''.join(block.__repr__() for block in self.blocks)
