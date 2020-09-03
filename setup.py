import platform
import setuptools
import subprocess

if platform.system() == 'Darwin':
    lib = 'libsample.dylib'
else:
    lib = 'libsample.so'

subprocess.run(['mkdir', '-p', 'build'], check=True)
subprocess.run(['cmake', '-S', '.', '-B', 'build'], check=True)
subprocess.run(['cmake', '--build', 'build'], check=True)
subprocess.run(['mv', f'build/{lib}', '.'], check=True)

setuptools.setup(
    name='mot',
    version='0.0.1',
    description='magneto optical novel photocathode simulations',
    url='https://github.com/clairehansel3/mot',
    author='Claire Hansel',
    author_email='clairehansel3@gmail.com',
    license='GPLv3',
    packages=['mot'],
    install_requires=['numpy', 'scipy', 'matplotlib', 'pygmo', 'python-daemon'],
    data_files=[lib]
)
