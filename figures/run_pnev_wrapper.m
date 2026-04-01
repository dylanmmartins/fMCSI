% Wrapper used to run Matlab MCMC spike inference following Pnevmatikakis
% et al.
%
% Written DMM, March 2026

cvx_clear
cvx_setup

try
    addpath(genpath(pwd));
    if ispc
        addpath(genpath('C:\Users\dmartins\Documents\MATLAB\cvx'));
    elseif isunix
        addpath(genpath('/home/dylan/Documents/MATLAB/cvx'));
    end
    
    load('mcmc_input.mat');
    [n_cells, n_frames] = size(dff);
    
    params.Nsamples = n_sweeps;
    params.B = floor(n_sweeps / 2); 
    params.p = 2; 
    params.f = fs;
    
    all_spikes = cell(n_cells, 1);
    all_probs = zeros(n_cells, n_frames);
    model_traces = zeros(n_cells, n_frames);
    
    set(0, 'DefaultFigureVisible', 'off');
    fprintf('Running MCMC on %d cells...\n', n_cells);
    
    for i = 1:n_cells
        y = double(dff(i, :))';
        
        try
            res = cont_ca_sampler(y, params);
          
            samples = res.ss;
            n_post = length(samples);
            
            prob_trace = zeros(1, n_frames);
            for s = 1:n_post
                st = samples{s};
                if ~isempty(st)
                    idx = round(st);
                    idx = idx(idx >= 1 & idx <= n_frames);
                    prob_trace(idx) = prob_trace(idx) + 1;
                end
            end
            all_probs(i, :) = prob_trace / max(1, n_post);
            
            if ~isempty(samples)
                all_spikes{i} = samples{end};
            end

            temp_trace = make_mean_sample(res, y);
            model_traces(i, :) = temp_trace(:)';
            
        catch ME
            fprintf('Error on cell %d: %s\n', i, ME.message);
        end
    end
    
    save('mcmc_output.mat', 'all_spikes', 'all_probs', 'model_traces');
    exit(0);
    
catch ME
    fprintf('Global MATLAB Error: %s\n', ME.message);
    exit(1);
end
    