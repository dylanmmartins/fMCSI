# -*- coding: utf-8 -*-
"""
Spike operations for additiona, removal, and replacement.

add_spike
    Add a given spike to the existing spike train.
remove_spike
    Remove a given spike from the existing spike train.
replace_spike
    Replace a given spike with a new one in the existing spike train.

Written Feb 2026, DMM
"""


import numpy as np
from numba import njit



@njit(cache=True, fastmath=True)
def add_spike(spike_train, n_spikes, old_calcium, old_ll, ef_h, ef_d, ef_nh, ef_nd,
              tau, obs_calcium, time_to_add, dt, A, check_only=False):
    """
    Add a given spike to the existing spike train using a pre-allocated array.

    If the pre-allocated array is full, it will be re-allocated to double its
    size. The new spike is appended to the end of the active elements.

    Parameters:
    spike_train:     1D np.array, pre-allocated spike train
    n_spikes:        int, current number of spikes in spike_train
    old_calcium:     1D np.array, current noiseless calcium trace
    old_ll:          float, current value of the log-likelihood function
    ef_h:            1D np.array, exponential rise kernel (normalised)
    ef_d:            1D np.array, exponential decay kernel (normalised)
    ef_nh:           1D np.array, cumsum of ef_h**2
    ef_nd:           1D np.array, cumsum of ef_d**2
    tau:             array-like of length 2, continuous time rise and decay time constants
    obs_calcium:     1D np.array, observed fluorescence trace
    time_to_add:     float, time of the spike to be added
    dt:              float, time-bin width
    A:               float, spike amplitude
    check_only:      bool, if True, returns delta_ll without modifying arrays

    Returns:
    spike_train:     1D np.array, new or modified vector of spike times
    new_n_spikes:    int, the new number of spikes
    new_calcium:     1D np.array, new noiseless calcium trace
    new_ll:          float, new value of the log-likelihood function
    """

    tau_h, tau_d = tau[0], tau[1]

    wk_h = A * np.exp((time_to_add - dt * np.ceil(time_to_add / dt)) / tau_h)
    wk_d = A * np.exp((time_to_add - dt * np.ceil(time_to_add / dt)) / tau_d)
    t_floor = int(np.floor(time_to_add))

    if check_only:

        end_idx_h = min(len(ef_h) + t_floor, len(old_calcium))
        tmp_len_h = end_idx_h - t_floor
        
        dot_h = 0.0
        for k in range(tmp_len_h):
            obs_val = obs_calcium[t_floor + k]
            if not np.isnan(obs_val):
                res_val = obs_val - old_calcium[t_floor + k]
                dot_h += res_val * (wk_h * ef_h[k])

        delta_ll_h = 2.0 * dot_h - (wk_h ** 2 * ef_nh[tmp_len_h - 1])

        end_idx_d = min(len(ef_d) + t_floor, len(old_calcium))
        tmp_len_d = end_idx_d - t_floor
        
        dot_d = 0.0
        for k in range(tmp_len_d):
            obs_val = obs_calcium[t_floor + k]
            if not np.isnan(obs_val):
                res_val = obs_val - old_calcium[t_floor + k]

                if k < tmp_len_h:
                    res_val -= wk_h * ef_h[k]
                dot_d += res_val * (wk_d * ef_d[k])
                
        delta_ll_d = 2.0 * dot_d - (wk_d ** 2 * ef_nd[tmp_len_d - 1])
        
        total_delta_ll = delta_ll_h + delta_ll_d
        
        return spike_train, n_spikes, old_calcium, total_delta_ll

    if n_spikes == len(spike_train): # reallocate if full
        new_capacity = max(len(spike_train) * 2, 1)
        new_spike_train = np.empty(new_capacity, dtype=spike_train.dtype)
        new_spike_train[:n_spikes] = spike_train[:n_spikes]
        spike_train = new_spike_train

    spike_train[n_spikes] = time_to_add
    new_n_spikes = n_spikes + 1

    end_idx_d = min(len(ef_d) + t_floor, len(old_calcium))
    tmp_len_d = end_idx_d - t_floor
    
    obstemp = obs_calcium[t_floor:end_idx_d]
    old_ca_tmp = old_calcium[t_floor:end_idx_d]
    
    new_ca_tmp = old_ca_tmp.copy()

    if np.any(ef_h):
        end_idx_h = min(len(ef_h) + t_floor, len(old_calcium))
        tmp_len_h = end_idx_h - t_floor
        wef_h = (wk_h * ef_h[:tmp_len_h]).astype(np.float32)
        new_ca_tmp[:tmp_len_h] += wef_h

    wef_d = (wk_d * ef_d[:tmp_len_d]).astype(np.float32)
    new_ca_tmp += wef_d

    sq_err_new = np.sum((new_ca_tmp - obstemp)[~np.isnan(new_ca_tmp - obstemp)] ** 2)
    sq_err_old = np.sum((old_ca_tmp - obstemp)[~np.isnan(old_ca_tmp - obstemp)] ** 2)
    
    new_ll = old_ll - sq_err_new + sq_err_old

    old_calcium[t_floor:end_idx_d] = new_ca_tmp
    
    return spike_train, new_n_spikes, old_calcium, new_ll





@njit(cache=True, fastmath=True)
def remove_spike(spike_train, n_spikes, old_calcium, old_ll, ef_h, ef_d, ef_nh, ef_nd,
                 tau, obs_calcium, time_to_remove, indx, dt, A, check_only=False):
    """
    Remove a given spike from the existing spike train using an in-place,
    O(1) unordered removal (swap with the last element).

    Parameters:
    spike_train:     1D np.array, pre-allocated spike train
    n_spikes:        int, current number of spikes in spike_train
    old_calcium:     1D np.array, current noiseless calcium trace
    old_ll:          float, current value of the log-likelihood function
    ef_h:            1D np.array, exponential rise kernel (normalised)
    ef_d:            1D np.array, exponential decay kernel (normalised)
    ef_nh:           1D np.array, cumsum of ef_h**2
    ef_nd:           1D np.array, cumsum of ef_d**2
    tau:             array-like of length 2, continuous time rise and decay time constants
    obs_calcium:     1D np.array, observed fluorescence trace
    time_to_remove:  float, time of the spike to be removed
    indx:            int, 0-based index where the spike to be removed is located
    dt:              float, time-bin width
    A:               float, spike amplitude
    check_only:      bool, if True, returns delta_ll without modifying arrays

    Returns:
    new_n_spikes:    int, the new number of spikes
    new_calcium:     1D np.array, new noiseless calcium trace
    new_ll:          float, new value of the log-likelihood function
    """

    tau_h, tau_d = tau[0], tau[1]

    wk_h = A * np.exp((time_to_remove - dt * np.ceil(time_to_remove / dt)) / tau_h)
    wk_d = A * np.exp((time_to_remove - dt * np.ceil(time_to_remove / dt)) / tau_d)
    t_floor = int(np.floor(time_to_remove))

    if check_only:

        end_idx_h = min(len(ef_h) + t_floor, len(old_calcium))
        tmp_len_h = end_idx_h - t_floor
        
        dot_h = 0.0
        for k in range(tmp_len_h):
            obs_val = obs_calcium[t_floor + k]
            if not np.isnan(obs_val):
                res_val = obs_val - old_calcium[t_floor + k]
                dot_h += res_val * (wk_h * ef_h[k])
        
        delta_ll_h = -2.0 * dot_h - (wk_h ** 2 * ef_nh[tmp_len_h - 1])

        end_idx_d = min(len(ef_d) + t_floor, len(old_calcium))
        tmp_len_d = end_idx_d - t_floor
        
        dot_d = 0.0
        for k in range(tmp_len_d):
            
            obs_val = obs_calcium[t_floor + k]
            
            if not np.isnan(obs_val):
                res_val = obs_val - old_calcium[t_floor + k]

                if k < tmp_len_h:
                    res_val += wk_h * ef_h[k]
                
                dot_d += res_val * (wk_d * ef_d[k])
                
        delta_ll_d = -2.0 * dot_d - (wk_d ** 2 * ef_nd[tmp_len_d - 1])
        
        return n_spikes, old_calcium, delta_ll_h + delta_ll_d

    spike_train[indx] = spike_train[n_spikes - 1]
    new_n_spikes = n_spikes - 1

    end_idx_d = min(len(ef_d) + t_floor, len(old_calcium))
    tmp_len_d = end_idx_d - t_floor
    
    obstemp = obs_calcium[t_floor:end_idx_d]
    old_ca_tmp = old_calcium[t_floor:end_idx_d]
    
    new_ca_tmp = old_ca_tmp.copy()

    if np.any(ef_h):
        end_idx_h = min(len(ef_h) + t_floor, len(old_calcium))
        tmp_len_h = end_idx_h - t_floor
        wef_h = (wk_h * ef_h[:tmp_len_h]).astype(np.float32)
        new_ca_tmp[:tmp_len_h] -= wef_h

    wef_d = (wk_d * ef_d[:tmp_len_d]).astype(np.float32)
    new_ca_tmp -= wef_d

    sq_err_new = np.sum((new_ca_tmp - obstemp)[~np.isnan(new_ca_tmp - obstemp)] ** 2)
    sq_err_old = np.sum((old_ca_tmp - obstemp)[~np.isnan(old_ca_tmp - obstemp)] ** 2)
    new_ll = old_ll - sq_err_new + sq_err_old

    old_calcium[t_floor:end_idx_d] = new_ca_tmp
    
    return new_n_spikes, old_calcium, new_ll





@njit(cache=True, fastmath=True)
def replace_spike(spike_train, old_calcium, old_ll, ef_h, ef_d, ef_nh, ef_nd, tau,
                  obs_calcium, time_to_remove, indx, time_to_add, dt, A, check_only=False):
    """
    Replace a given spike with a new one in the existing spike train, in-place.

    Parameters:
    spike_train:     1D np.array, current spike train (modified in-place)
    old_calcium:     1D np.array, current noiseless calcium trace
    old_ll:          float, current value of the log-likelihood function
    ef_h:            1D np.array, exponential rise kernel (normalised)
    ef_d:            1D np.array, exponential decay kernel (normalised)
    ef_nh:           1D np.array, cumsum of ef_h**2
    ef_nd:           1D np.array, cumsum of ef_d**2
    tau:             array-like of length 2, continuous time rise and decay time constants
    obs_calcium:     1D np.array, observed fluorescence trace
    time_to_remove:  float, time of the spike to be removed
    indx:            int, 0-based index where the spike to be removed is in the vector
    time_to_add:     float, time of the spike to be added
    dt:              float, time-bin width
    A:               float, spike amplitude

    Returns:
    new_calcium:     1D np.array, new noiseless calcium trace
    new_ll:          float, new value of the log-likelihood function
    """

    tau_h, tau_d = tau[0], tau[1]

    wk_hr = A * np.exp((time_to_remove - dt * np.ceil(time_to_remove / dt)) / tau_h)
    wk_dr = A * np.exp((time_to_remove - dt * np.ceil(time_to_remove / dt)) / tau_d)
    t_rem_floor = int(np.floor(time_to_remove))

    wk_ha = A * np.exp((time_to_add - dt * np.ceil(time_to_add / dt)) / tau_h)
    wk_da = A * np.exp((time_to_add - dt * np.ceil(time_to_add / dt)) / tau_d)
    t_add_floor = int(np.floor(time_to_add))

    min_t = int(np.floor(min(time_to_remove, time_to_add)))

    end_idx_d = min(len(ef_d) + min_t, len(old_calcium))
    
    obstemp = obs_calcium[min_t:end_idx_d]
    old_ca_tmp = old_calcium[min_t:end_idx_d]
    
    new_ca_tmp = old_ca_tmp.copy()

    if np.any(ef_h):
        end_idx_h = min(len(ef_h) + min_t, len(old_calcium))
        tmp_len_h = end_idx_h - min_t

        if t_rem_floor == t_add_floor:
            wef_h = (wk_hr - wk_ha) * ef_h[:tmp_len_h]
        elif t_rem_floor > t_add_floor:
            diff_t = t_rem_floor - t_add_floor
            pad_len = min(tmp_len_h, diff_t)
            ef_len = max(0, tmp_len_h - pad_len)

            shifted_ef = np.empty(tmp_len_h, dtype=ef_h.dtype)
            shifted_ef[:pad_len] = 0.0
            shifted_ef[pad_len:] = ef_h[:ef_len]
            wef_h = wk_hr * shifted_ef - wk_ha * ef_h[:tmp_len_h]
        else:
            diff_t = t_add_floor - t_rem_floor
            pad_len = min(tmp_len_h, diff_t)
            ef_len = max(0, tmp_len_h - pad_len)

            shifted_ef = np.empty(tmp_len_h, dtype=ef_h.dtype)
            shifted_ef[:pad_len] = 0.0
            shifted_ef[pad_len:] = ef_h[:ef_len]
            wef_h = wk_hr * ef_h[:tmp_len_h] - wk_ha * shifted_ef

        new_ca_tmp[:tmp_len_h] -= wef_h.astype(np.float32)

    tmp_len_d = end_idx_d - min_t

    if t_rem_floor == t_add_floor:
        wef_d = (wk_dr - wk_da) * ef_d[:tmp_len_d]
    elif t_rem_floor > t_add_floor:
        diff_t = t_rem_floor - t_add_floor
        pad_len = min(tmp_len_d, diff_t)
        ef_len = max(0, tmp_len_d - pad_len)

        shifted_ef = np.empty(tmp_len_d, dtype=ef_d.dtype)
        shifted_ef[:pad_len] = 0.0
        shifted_ef[pad_len:] = ef_d[:ef_len]
        wef_d = wk_dr * shifted_ef - wk_da * ef_d[:tmp_len_d]
    else:
        diff_t = t_add_floor - t_rem_floor
        pad_len = min(tmp_len_d, diff_t)
        ef_len = max(0, tmp_len_d - pad_len)

        shifted_ef = np.empty(tmp_len_d, dtype=ef_d.dtype)
        shifted_ef[:pad_len] = 0.0
        shifted_ef[pad_len:] = ef_d[:ef_len]
        wef_d = wk_dr * ef_d[:tmp_len_d] - wk_da * shifted_ef

    new_ca_tmp -= wef_d.astype(np.float32)

    sq_err_new = np.sum((new_ca_tmp - obstemp)[~np.isnan(new_ca_tmp - obstemp)] ** 2)
    sq_err_old = np.sum((old_ca_tmp - obstemp)[~np.isnan(old_ca_tmp - obstemp)] ** 2)

    delta_ll = -sq_err_new + sq_err_old

    if check_only:

        return old_calcium, delta_ll
    
    else:
        old_calcium[min_t:end_idx_d] = new_ca_tmp
        spike_train[indx] = time_to_add

        return old_calcium, old_ll + delta_ll


