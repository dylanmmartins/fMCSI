# -*- coding: utf-8 -*-
"""
Run MCMC spike inference following Pnevmatikakis et al., calling
Matlab as a subprocess.

Written DMM, March 2026
"""


import os
import time
import platform
import subprocess
import numpy as np
import scipy.io
from pathlib import Path



def run_matlab_pnevMCMC(dff, fs=30.0, tau=0.5, n_sweeps=1000, true_spikes=None, sparsity_scale=0.001):

    if dff.ndim == 1:
        dff = dff[np.newaxis, :]
    n_cells, n_frames = dff.shape
    
    if n_sweeps == 'auto':
        n_sweeps_val = 500
    else:
        n_sweeps_val = int(n_sweeps)

    input_mat = 'mcmc_input.mat'
    output_mat = 'mcmc_output.mat'
    wrapper_script = 'run_pnev_wrapper.m'
    
    scipy.io.savemat(input_mat, {
        'dff': dff.astype(np.float64),
        'fs': float(fs),
        'tau': float(tau),
        'n_sweeps': n_sweeps_val,
        'sparsity_scale': float(sparsity_scale)
    })
    

    matlab_code = f"""
    cvx_clear
    cvx_setup

    try
        addpath(genpath(pwd));
        if ispc
            addpath(genpath('C:\\Users\\dmartins\\Documents\\MATLAB\\cvx'));
        elseif isunix
            addpath(genpath('/home/dylan/Documents/MATLAB/cvx'));
        end
        
        load('{input_mat}');
        [n_cells, n_frames] = size(dff);
        
        params.Nsamples = n_sweeps;
        params.B = floor(n_sweeps / 2); 
        params.p = 2; 
        params.f = fs;
        
        all_spikes = cell(n_cells, 1);
        all_probs = zeros(n_cells, n_frames);
        model_traces = zeros(n_cells, n_frames);
        
        set(0, 'DefaultFigureVisible', 'off');
        fprintf('Running MCMC on %d cells...\\n', n_cells);
        
        for i = 1:n_cells

            y = double(dff(i, :))';
            
            try
                res = cont_ca_sampler(y, params);
            
                samples = res.ss;
                n_post = length(samples);
                
                prob_trace = zeros(1, n_frames);
                for s = 1:n_post
                    st = samples{{s}};
                    if ~isempty(st)
                        idx = round(st);
                        idx = idx(idx >= 1 & idx <= n_frames);
                        prob_trace(idx) = prob_trace(idx) + 1;
                    end
                end
                all_probs(i, :) = prob_trace / max(1, n_post);
                
                if ~isempty(samples)
                    all_spikes{{i}} = samples{{end}};
                end
                
                temp_trace = make_mean_sample(res, y);
                model_traces(i, :) = temp_trace(:)'; % Ensure row vector
                
            catch ME
                fprintf('Error on cell %d: %s\\n', i, ME.message);
            end
        end
        
        save('{output_mat}', 'all_spikes', 'all_probs', 'model_traces');
        exit(0);
        
    catch ME
        fprintf('Global MATLAB Error: %s\\n', ME.message);
        exit(1);
    end
    """
    
    with open(wrapper_script, 'w') as f:
        f.write(matlab_code)
        
    print(f"Calling MATLAB subprocess (sweeps={n_sweeps_val})...")
    t0 = time.time()
    
    if platform.system() == "Windows":
        matlab_exe = Path(r"C:\Program Files\MATLAB\R2025b\bin\matlab.exe")
    elif platform.system() == "Linux":
        matlab_exe = '/usr/local/MATLAB/R2025b/bin/matlab'
        
    if not os.path.exists(matlab_exe):
        matlab_exe = 'matlab'

    cmd = f"\"{matlab_exe}\" -batch \"run_pnev_wrapper\""
    
    try:
        subprocess.run(cmd, shell=True, check=True)
    except subprocess.CalledProcessError:
        print("MATLAB execution failed. Please ensure 'matlab' is in your PATH and 'cont_ca_sampler' is available.")

        return [np.array([]) for _ in range(n_cells)], np.zeros_like(dff), np.zeros_like(dff), np.zeros(n_cells)

    print(f"MATLAB finished in {time.time() - t0:.2f}s")
    
    if not os.path.exists(output_mat):
        print("Error: Output MAT file not found.")
        return [np.array([]) for _ in range(n_cells)], np.zeros_like(dff), np.zeros_like(dff), np.zeros(n_cells)
        
    res = scipy.io.loadmat(output_mat)
    
    all_spikes_raw = res['all_spikes']
    all_probs = res['all_probs']
    model_traces = res['model_traces']
    
    final_spikes = []
    for i in range(n_cells):

        spks = all_spikes_raw[i][0]
        if spks.size > 0:

            times = spks.flatten() / fs
            final_spikes.append(times)
        else:
            final_spikes.append(np.array([]))
            
    sweeps_per_cell = np.full(n_cells, n_sweeps_val, dtype=np.int32)
        
    return final_spikes, model_traces, all_probs, sweeps_per_cell
