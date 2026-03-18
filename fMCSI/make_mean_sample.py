# -*- coding: utf-8 -*-
"""
Construct the mean calcium trace from posterior samples.

Written Feb 2026, DMM
"""

import numpy as np
import numba as nb


@nb.njit(cache=True, fastmath=True)
def _iir_filter(x, alpha):
    """1-pole causal IIR: y[n] = x[n] + alpha*y[n-1]."""
    y = np.empty_like(x)
    y[0] = x[0]
    for i in range(1, len(x)):
        y[i] = x[i] + alpha * y[i - 1]
    return y


@nb.njit(cache=True, fastmath=True)
def _compute_single_trace(ss_arr, T, tau0, tau1, am_val, cb_val, cin_val, dt):
    """
    Compute the reconstructed calcium trace for one MCMC sample.

    Replaces per-sample scipy.signal.lfilter calls with the JIT'd _iir_filter
    and bins spike times in the same loop that computes the kernel weights.
    """
    gr0 = np.float32(np.exp(-dt / tau0)) if tau0 > 0.0 else np.float32(0.0)
    gr1 = np.float32(np.exp(-dt / tau1))
    diff_gr = gr1 - gr0   # float32

    ge = np.empty(T, dtype=np.float32)

    ge[0] = np.float32(1.0)
    for k in range(1, T):
        ge[k] = ge[k - 1] * gr1

    s_1 = np.zeros(T, dtype=np.float32)
    s_2 = np.zeros(T, dtype=np.float32)

    for j in range(len(ss_arr)):

        st = ss_arr[j]
        ceil_st = np.ceil(st / dt)
        idx = int(ceil_st) - 1

        if idx < 0:
            idx = 0
        elif idx >= T:
            idx = T - 1

        offset = st - dt * ceil_st
        if gr0 > 0.0:
            s_1[idx] += np.float32(np.exp(offset / tau0))
        s_2[idx] += np.float32(np.exp(offset / tau1))

    G1sp = _iir_filter(s_1, gr0) if gr0 > 0.0 else np.zeros(T, dtype=np.float32)
    G2sp = _iir_filter(s_2, gr1)
    Gs   = (-G1sp + G2sp) / diff_gr   # float32

    am_f  = np.float32(am_val)
    cb_f  = np.float32(cb_val)
    cin_f = np.float32(cin_val)

    trace = np.empty(T, dtype=np.float32)
    for k in range(T):
        trace[k] = cb_f + am_f * Gs[k] + cin_f * ge[k]

    return trace


def make_mean_sample(SAMPLES, Y):
    """
    Construct the mean calcium trace from posterior samples.

    Parameters:
    SAMPLES: dict, output of cont_ca_sampler
    Y:       1D np.array, observed fluorescence trace

    Returns:
    c_m:     1D np.array, mean reconstructed calcium trace

    Acceleration: the per-sample reconstruction (binning + IIR filtering +
    trace assembly) is delegated to the JIT'd _compute_single_trace kernel,
    cutting the Python-loop overhead to just N iterations of lightweight
    scalar indexing.
    """
    T = len(Y)
    N = len(SAMPLES['ns'])
    P = SAMPLES['params']

    f_val = P.get('f', 1.0)
    dt    = 1.0 / f_val
    g_val = np.array(P['g']).flatten()

    if 'g' not in SAMPLES:
        SAMPLES['g'] = np.tile(g_val, (N, 1))

    marg = 1 if len(np.atleast_1d(SAMPLES['Cb'])) == 2 else 0

    C_sum    = np.zeros(T, dtype=np.float32)
    Cin_flat = np.array(SAMPLES['Cin']).flatten()

    for rep in range(N):
        tau = np.atleast_1d(SAMPLES['g'][rep, :])

        ss      = np.atleast_1d(SAMPLES['ss'][rep]).astype(np.float64)
        am_val  = float(SAMPLES['Am'][rep])

        if marg:
            cb_val  = float(SAMPLES['Cb'][0])
            cin_val = float(Cin_flat[0])
        else:
            cb_val  = float(np.atleast_1d(SAMPLES['Cb'])[rep])
            cin_val = float(Cin_flat[rep] if len(Cin_flat) > rep else Cin_flat[-1])

        trace = _compute_single_trace(
            ss, T, float(tau[0]), float(tau[1]),
            am_val, cb_val, cin_val, dt,
        )
        C_sum += trace

    return C_sum / N
