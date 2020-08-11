import daemon
import functools
import itertools
import multiprocessing
import operator
import os
import pathlib
import subprocess

def run_helper(do_scan, item):
    return do_scan(item[0], item[1], *item[2])

def run(this, do_scan, *parameters, processes=1, detach=True, data_folder_path=None, suffix=None):
    """
    Runs a parameter scan in parallel.

    Inputs:
        this:
            __file__
        do_scan:
            A function which takes a simulation index, a simulation folder path,
            and a number of parameters equal to len(parameters).
        parameters:
            Any number of iterables representing different parameters.
        processes:
            Number of parallel processes to run on.
        detatch:
            Whether to run as a daemon
    """

    script = pathlib.Path(this).resolve(strict=True)
    simulation_name = script.with_suffix('').name
    if suffix is not None:
        simulation_name = simulation_name + '_' + str(suffix)
        script = script.parent / simulation_name
    log = script.with_suffix('.log')
    kill_script = script.with_suffix('.kill')
    data_folder = pathlib.Path(data_folder_path) / simulation_name
    if not data_folder.is_dir():
        data_folder.mkdir()
    if detach:
        if kill_script.is_file():
            raise Exception(f'Simulation {simulation_name} is already running')
        print(f'Running simulation {simulation_name}')
        with open(log, 'w+') as f:
            with daemon.DaemonContext(
                stdout=f,
                stderr=f,
                working_directory=script.parent
            ):
                with open(kill_script, 'w+') as f:
                    f.write(f'#!/bin/bash\nkill -SIGINT {os.getpid()}\n')
                try:
                    subprocess.run(['chmod', '+x', kill_script], check=True)
                    combined_parameters = itertools.product(*parameters)
                    number_of_scans = functools.reduce(operator.mul, (len(parameter) for parameter in parameters), 1)
                    paths = (data_folder / str(index) for index in range(number_of_scans))
                    iter = zip(range(number_of_scans), paths, combined_parameters)
                    if processes > number_of_scans:
                        processes = number_of_scans
                    function = functools.partial(run_helper, do_scan)
                    if processes == 1:
                        for result in map(function, iter):
                            print(f'simulation {result} complete')
                    else:
                        with multiprocessing.Pool(processes) as pool:
                            for result in pool.imap_unordered(function, iter):
                                print(f'simulation {result} complete')
                    kill_script.unlink()
                except:
                    kill_script.unlink()
                    raise
    else:
        print(f'Running simulation {simulation_name}')
        combined_parameters = itertools.product(*parameters)
        number_of_scans = functools.reduce(operator.mul, (len(parameter) for parameter in parameters), 1)
        paths = (data_folder / str(index) for index in range(number_of_scans))
        iter = zip(range(number_of_scans), paths, combined_parameters)
        if processes > number_of_scans:
            processes = number_of_scans
        function = functools.partial(run_helper, do_scan)
        if processes == 1:
            for result in map(function, iter):
                print(f'simulation {result} complete')
        else:
            with multiprocessing.Pool(processes) as pool:
                for result in pool.imap_unordered(function, iter):
                    print(f'simulation {result} complete')
    print(f'Done with simulation {simulation_name}')

def detach_run_parallel(this, functions, processes=1, suffix=None):
    script = pathlib.Path(this).resolve(strict=True)
    simulation_name = script.with_suffix('').name
    if suffix is not None:
        simulation_name = simulation_name + '_' + str(suffix)
        script = script.parent / simulation_name
    log = script.with_suffix('.log')
    kill_script = script.with_suffix('.kill')
    if kill_script.is_file():
        raise Exception(f'Simulation {simulation_name} is already running')
    print(f'Running simulation {simulation_name}')
    with open(log, 'w+') as f:
        with daemon.DaemonContext(
            stdout=f,
            stderr=f,
            working_directory=script.parent
        ):
            with open(kill_script, 'w+') as f:
                f.write(f'#!/bin/bash\nkill -SIGINT {os.getpid()}\n')
            try:
                if processes > len(functions):
                    processes = len(functions)
                if processes == 1:
                    for function in functions:
                        function()
                else:
                    with multiprocessing.Pool(processes) as pool:
                        results = []
                        for function in functions:
                            results.append(pool.apply_async(function))
                        for result in results:
                            result.get()
                kill_script.unlink()
            except:
                kill_script.unlink()
                raise

def detach_run(this, main, suffix=None):
    script = pathlib.Path(this).resolve(strict=True)
    simulation_name = script.with_suffix('').name
    if suffix is not None:
        simulation_name = simulation_name + '_' + str(suffix)
        script = script.parent / simulation_name
    log = script.with_suffix('.log')
    kill_script = script.with_suffix('.kill')
    if kill_script.is_file():
        raise Exception(f'Simulation {simulation_name} is already running')
    print(f'Running simulation {simulation_name}')
    with open(log, 'w+') as f:
        with daemon.DaemonContext(
            stdout=f,
            stderr=f,
            working_directory=script.parent
        ):
            with open(kill_script, 'w+') as f:
                f.write(f'#!/bin/bash\nkill -SIGINT {os.getpid()}\n')
            try:
                main()
                kill_script.unlink()
            except:
                kill_script.unlink()
                raise
