#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Scaling and sensitivity benchmarks

To run inference
    $ python figure2.py --mode test --data-dir /path/to/results

To create figure:
    $ python figure2.py --mode plot --data-dir /path/to/results

Written DMM, March 2026
"""

import argparse
import os
import subprocess
import time
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from scipy.signal import find_peaks
from oasis.functions import deconvolve

import fMCSI
import fMCSI.helpers as helpers
from run_pnev_MCMC import run_matlab_pnevMCMC
from simulation_helpers import generate_synthetic_data

mpl.rcParams['axes.spines.top']  = False
mpl.rcParams['axes.spines.right'] = False
mpl.rcParams['pdf.fonttype'] = 42
mpl.rcParams['ps.fonttype']  = 42
mpl.rcParams['font.size']    = 7

np.random.seed(3)

BETA = 0.5
USE_STRICT_ACCURACY = True

COLORS = {
    'fMCSI':       '#4C72B0',
    'MATLAB':      '#DD8452',
    'OASIS':       '#55A868',
    'CASCADE_GPU': '#8172B3',
    'CASCADE_CPU': '#B39DDB',
}


def _run_cascade_inference(dff, fs, data_dir, prefix, device='gpu'):

    script = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                          'run_cascade_subprocess.py')
    input_path  = os.path.join(data_dir, f'{prefix}_input.npz')
    output_path = os.path.join(data_dir, f'{prefix}_output.npz')

    np.savez(input_path, dff=dff.astype(np.float32), fs=np.float32(fs))
    subprocess.run(
        ['conda', 'run', '-n', 'cascade', 'python', script,
         '--mode', 'inference', '--input', input_path, '--output', output_path,
         '--device', device],
        check=True
    )
    result = np.load(output_path, allow_pickle=True)
    cascade_probs  = result['cascade_probs']
    cascade_spikes = list(result['cascade_spikes'])
    cascade_time   = float(result['cascade_time'])
    return cascade_probs, cascade_spikes, cascade_time


def _metrics(true_spk, pred_spk, true_ev, fs_):
    prec,   rec,   f1   = fMCSI.compute_accuracy_strict(true_spk, pred_spk, tolerance=0.1)
    prec_w, rec_w, f1_w = helpers.compute_accuracy_window(true_spk, pred_spk)
    prec_e, rec_e, f1_e = helpers.compute_accuracy_window(true_ev,  pred_spk)
    cosmic = helpers.compute_cosmic(true_spk, pred_spk, fs_)
    return (np.mean(prec),   np.mean(rec),   np.mean(f1),
            np.mean(prec_w), np.mean(rec_w), np.mean(f1_w),
            np.mean(prec_e), np.mean(rec_e), np.mean(f1_e),
            np.mean(cosmic))


def _row(exp, model, tau_, fs_, time_, m, sweeps=0, n_cells=None, duration=None,
         mean_kurtosis=None, **extra):
    p, r, f, pw, rw, fw, pe, re_, fe, cos = m
    d = {
        'Experiment': exp, 'Model': model, 'Tau': tau_, 'Fs': fs_,
        'Time': time_, 'Sweeps': sweeps,
        'F1': f, 'Precision': p, 'Recall': r,
        'F1_window': fw, 'Precision_window': pw, 'Recall_window': rw,
        'F1_event': fe, 'Precision_event': pe, 'Recall_event': re_,
        'COSMIC': cos,
    }
    if n_cells is not None:
        d['N_Cells'] = n_cells
    if duration is not None:
        d['Duration'] = duration
    if mean_kurtosis is not None:
        d['Mean_Kurtosis'] = mean_kurtosis
    d.update(extra)
    return d

def _save_records(records, path):
    if not records:
        np.savez(path)
        return
    keys = list(dict.fromkeys(k for r in records for k in r))
    out = {}
    for k in keys:
        vals = [r.get(k, None) for r in records]
        if any(isinstance(v, str) for v in vals if v is not None):
            out[k] = np.array([str(v) if v is not None else '' for v in vals], dtype=object)
        else:
            out[k] = np.array([float(v) if v is not None else np.nan for v in vals],
                               dtype=np.float64)
    np.savez(path, **out)


def _load_records(path):

    d = np.load(path, allow_pickle=True)
    return {k: d[k] for k in d.files}


def _tbl_len(tbl):
    return len(next(iter(tbl.values()))) if tbl else 0


def _tbl_filter(tbl, col, val):
    mask = tbl[col] == val
    return {k: v[mask] for k, v in tbl.items()}


def _tbl_sort(tbl, col):
    idx = np.argsort(tbl[col].astype(float))
    return {k: v[idx] for k, v in tbl.items()}


def _tbl_concat(tbls):
    tbls = [t for t in tbls if t]
    if not tbls:
        return {}
    all_keys = list(dict.fromkeys(k for t in tbls for k in t))
    result = {}
    for k in all_keys:
        parts = []
        for t in tbls:
            if k in t:
                parts.append(t[k])
            else:
                n = _tbl_len(t)
                ref = next((t2[k] for t2 in tbls if k in t2), None)
                if ref is not None and ref.dtype == object:
                    parts.append(np.array([''] * n, dtype=object))
                else:
                    parts.append(np.full(n, np.nan))
        if any(p.dtype == object for p in parts):
            result[k] = np.concatenate([p.astype(object) for p in parts])
        else:
            result[k] = np.concatenate([p.astype(np.float64) for p in parts])
    return result


def _oasis_spikes(dff, fs, tau, n_cells):
    diff = np.diff(dff, axis=1)
    sigmas = np.median(np.abs(diff), axis=1) / (0.6745 * np.sqrt(2))
    sigmas = np.maximum(sigmas, 1e-9)
    spikes, calcium = [], []
    for i in range(n_cells):
        g = np.exp(-1 / (fs * tau))
        c, s, _, _, _ = deconvolve(dff[i], g=(g,), sn=sigmas[i], penalty=1)
        spikes.append(np.where(s > 0.2 * sigmas[i])[0] / fs)
        calcium.append(c)
    return spikes, calcium


def benchmark_sweeps(data_dir, run_oasis=True, run_matlab=True, run_mine=True,
                     run_cascade=True):

    n_cells     = 50
    duration    = 600
    fs          = 30.0
    tau         = 1.2
    sweeps_list = [10, 50, 100, 250, 500, 1000, 2000, 3000]

    print(f'Generating synthetic data (n_cells={n_cells}, duration={duration}s)...')
    dff, true_spikes, _, _, _, _ = generate_synthetic_data(
        n_cells=n_cells, fs=fs, duration=duration, tau=tau
    )
    n_frames   = dff.shape[1]
    true_events = [helpers.make_event_ground_truth(s, tau) for s in true_spikes]

    results   = []
    npz_spikes = {'true': true_spikes}
    npz_calcium = {}

    partial_path = os.path.join(data_dir, 'benchmark_sweeps_partial.npz')

    if run_oasis:
        print('\nRunning OASIS (baseline)...')
        t0 = time.time()
        oas_spk, oas_cal = _oasis_spikes(dff, fs, tau, n_cells)
        time_oasis = time.time() - t0
        results.append(_row('Sweeps', 'OASIS', tau, fs, time_oasis,
                            _metrics(true_spikes, oas_spk, true_events, fs),
                            sweeps=0, n_cells=n_cells, duration=duration,
                            Samples_per_sec=np.nan))
        npz_spikes['oasis'] = oas_spk
        npz_calcium['oasis'] = np.array(oas_cal)
        print(f'  OASIS: {time_oasis:.1f}s  F1={results[-1]["F1"]:.3f}')
        _save_records(results, partial_path)

    if run_cascade:
        for _dev, _model in [('gpu', 'CASCADE_GPU'), ('cpu', 'CASCADE_CPU')]:
            print(f'\nRunning CASCADE (subprocess, {_dev.upper()}, baseline)...')
            _, cascade_spikes, time_cascade = _run_cascade_inference(
                dff, fs, data_dir, f'bench_sweeps_cascade_baseline_{_dev}', device=_dev
            )
            results.append(_row('Sweeps', _model, tau, fs, time_cascade,
                                _metrics(true_spikes, cascade_spikes, true_events, fs),
                                sweeps=0, n_cells=n_cells, duration=duration,
                                Samples_per_sec=np.nan))
            npz_spikes[f'cascade_{_dev}'] = cascade_spikes
            print(f'  CASCADE ({_dev.upper()}): {time_cascade:.1f}s  F1={results[-1]["F1"]:.3f}')
            _save_records(results, partial_path)

    print(f'\n--- Varying sweeps: {sweeps_list} ---')
    for s in sweeps_list:
        print(f'  sweeps={s}...')
        if run_mine:
            try:
                t0 = time.time()
                burn_in = int(s * 0.25)
                params  = {'f': fs, 'p': 2, 'Nsamples': s - burn_in, 'B': burn_in, 'auto_stop': False}
                res = fMCSI.run_deconv(dff, params=params, benchmark=True)
                elapsed = time.time() - t0
                sps = (s * n_cells * n_frames) / elapsed
                results.append(_row('Sweeps', 'fMCSI', tau, fs, elapsed,
                                    _metrics(true_spikes, res['optim_spikes'], true_events, fs),
                                    sweeps=s, n_cells=n_cells, duration=duration,
                                    Samples_per_sec=sps))
                npz_spikes['my_method'] = res['optim_spikes']
                print(f'    fMCSI: {elapsed:.1f}s  F1={results[-1]["F1"]:.3f}')
            except Exception as exc:
                print(f'    fMCSI failed: {exc}')

        if run_matlab:
            try:
                t0 = time.time()
                trad_spikes, _, _, _ = run_matlab_pnevMCMC(dff, fs=fs, tau=tau, n_sweeps=s)
                elapsed = time.time() - t0
                sps = (s * n_cells * n_frames) / elapsed
                results.append(_row('Sweeps', 'MATLAB', tau, fs, elapsed,
                                    _metrics(true_spikes, trad_spikes, true_events, fs),
                                    sweeps=s, n_cells=n_cells, duration=duration,
                                    Samples_per_sec=sps))
                npz_spikes['trad_mcmc'] = trad_spikes
                print(f'    MATLAB: {elapsed:.1f}s  F1={results[-1]["F1"]:.3f}')
            except Exception as exc:
                print(f'    MATLAB failed: {exc}')

        _save_records(results, partial_path)

    npz_save = {'dff': dff, 'fs': fs, 'tau': tau}
    for k, v in npz_spikes.items():
        npz_save[f'spikes_{k}'] = np.array(v, dtype=object)
    for k, v in npz_calcium.items():
        npz_save[f'calcium_{k}'] = v
    np.savez(os.path.join(data_dir, 'benchmark_sweeps_traces.npz'), **npz_save)
    return


def benchmark_scalability(data_dir, run_oasis=True, run_matlab=True, run_mine=True,
                           run_cascade=True):

    fs  = 30.0
    tau = 1.2
    cell_counts    = [50, 200, 500, 1000, 2000, 3000]
    fixed_duration = 300.0
    durations      = [300, 1800, 3600, 7200]
    fixed_cells    = 100

    results = []
    partial_path = os.path.join(data_dir, 'benchmark_scalability_partial.npz')

    print('\n--- Cell-count scaling ---')
    for n_cells in cell_counts:
        print(f'  n_cells={n_cells}...')
        try:
            dff, true_spikes, _, _, _, _ = generate_synthetic_data(
                n_cells=n_cells, fs=fs, duration=fixed_duration, tau=tau
            )
        except MemoryError:
            print(f'  Skipping n_cells={n_cells}: MemoryError')
            continue
        n_frames = dff.shape[1]

        if run_mine:
            try:
                t0 = time.time()
                res = fMCSI.run_deconv(dff, params={'f': fs, 'p': 2, 'auto_stop': True},
                                       benchmark=True)
                t_my = time.time() - t0
                sps  = (np.mean(res['optim_nsamples']) * n_cells * n_frames) / t_my
                results.append({'Experiment': 'Cell_Scaling', 'Model': 'fMCSI',
                                'N_Cells': n_cells, 'Time': t_my, 'Samples_per_sec': sps,
                                'Duration': fixed_duration, 'Frames': n_frames})
                print(f'    fMCSI: {t_my:.1f}s')
            except Exception as exc:
                print(f'    fMCSI failed: {exc}')

        if run_matlab:
            try:
                t0 = time.time()
                _, _, _, sweeps = run_matlab_pnevMCMC(dff, fs=fs, tau=tau, n_sweeps='auto')
                t_trad = time.time() - t0
                sps    = (np.mean(sweeps) * n_cells * n_frames) / t_trad
                results.append({'Experiment': 'Cell_Scaling', 'Model': 'MATLAB',
                                'N_Cells': n_cells, 'Time': t_trad, 'Samples_per_sec': sps,
                                'Duration': fixed_duration, 'Frames': n_frames})
                print(f'    MATLAB: {t_trad:.1f}s')
            except Exception as exc:
                print(f'    MATLAB failed: {exc}')

        if run_oasis:
            try:
                t0 = time.time()
                _oasis_spikes(dff, fs, tau, n_cells)
                t_oasis = time.time() - t0
                results.append({'Experiment': 'Cell_Scaling', 'Model': 'OASIS',
                                'N_Cells': n_cells, 'Time': t_oasis,
                                'Samples_per_sec': np.nan,
                                'Duration': fixed_duration, 'Frames': n_frames})
                print(f'    OASIS: {t_oasis:.1f}s')
            except Exception as exc:
                print(f'    OASIS failed: {exc}')

        if run_cascade:
            for _dev, _model in [('gpu', 'CASCADE_GPU'), ('cpu', 'CASCADE_CPU')]:
                try:
                    _, _, t_cascade = _run_cascade_inference(
                        dff, fs, data_dir, f'bench_scale_cells_{n_cells}_{_dev}', device=_dev)
                    results.append({'Experiment': 'Cell_Scaling', 'Model': _model,
                                    'N_Cells': n_cells, 'Time': t_cascade,
                                    'Samples_per_sec': np.nan,
                                    'Duration': fixed_duration, 'Frames': n_frames})
                    print(f'    CASCADE ({_dev.upper()}): {t_cascade:.1f}s')
                except Exception as exc:
                    print(f'    CASCADE ({_dev.upper()}) failed: {exc}')

        _save_records(results, partial_path)

    print('\n--- Duration scaling ---')
    for dur in durations:
        print(f'  duration={dur}s...')
        try:
            dff, true_spikes, _, _, _, _ = generate_synthetic_data(
                n_cells=fixed_cells, fs=fs, duration=dur, tau=tau
            )
        except MemoryError:
            print(f'  Skipping duration={dur}: MemoryError')
            continue
        n_frames = dff.shape[1]

        if run_mine:
            try:
                t0 = time.time()
                res = fMCSI.run_deconv(dff, params={'f': fs, 'p': 2, 'auto_stop': True},
                                       benchmark=True)
                t_my = time.time() - t0
                sps  = (np.mean(res['optim_nsamples']) * fixed_cells * n_frames) / t_my
                results.append({'Experiment': 'Duration_Scaling', 'Model': 'fMCSI',
                                'Duration': dur, 'Time': t_my, 'Samples_per_sec': sps,
                                'N_Cells': fixed_cells, 'Frames': n_frames})
                print(f'    fMCSI: {t_my:.1f}s')
            except Exception as exc:
                print(f'    fMCSI failed: {exc}')

        if run_matlab:
            try:
                t0 = time.time()
                _, _, _, sweeps = run_matlab_pnevMCMC(dff, fs=fs, tau=tau, n_sweeps='auto')
                t_trad = time.time() - t0
                sps    = (np.mean(sweeps) * fixed_cells * n_frames) / t_trad
                results.append({'Experiment': 'Duration_Scaling', 'Model': 'MATLAB',
                                'Duration': dur, 'Time': t_trad, 'Samples_per_sec': sps,
                                'N_Cells': fixed_cells, 'Frames': n_frames})
                print(f'    MATLAB: {t_trad:.1f}s')
            except Exception as exc:
                print(f'    MATLAB failed: {exc}')

        if run_oasis:
            try:
                t0 = time.time()
                _oasis_spikes(dff, fs, tau, fixed_cells)
                t_oasis = time.time() - t0
                results.append({'Experiment': 'Duration_Scaling', 'Model': 'OASIS',
                                'Duration': dur, 'Time': t_oasis,
                                'Samples_per_sec': np.nan,
                                'N_Cells': fixed_cells, 'Frames': n_frames})
                print(f'    OASIS: {t_oasis:.1f}s')
            except Exception as exc:
                print(f'    OASIS failed: {exc}')

        if run_cascade:
            for _dev, _model in [('gpu', 'CASCADE_GPU'), ('cpu', 'CASCADE_CPU')]:
                try:
                    _, _, t_cascade = _run_cascade_inference(
                        dff, fs, data_dir, f'bench_scale_dur_{dur}_{_dev}', device=_dev)
                    results.append({'Experiment': 'Duration_Scaling', 'Model': _model,
                                    'Duration': dur, 'Time': t_cascade,
                                    'Samples_per_sec': np.nan,
                                    'N_Cells': fixed_cells, 'Frames': n_frames})
                    print(f'    CASCADE ({_dev.upper()}): {t_cascade:.1f}s')
                except Exception as exc:
                    print(f'    CASCADE ({_dev.upper()}) failed: {exc}')

        _save_records(results, partial_path)

    return


def benchmark_params(data_dir, run_oasis=True, run_matlab=True, run_mine=True,
                     run_cascade=True):
    
    n_cells  = 50
    duration = 300
    tau_values = [0.2, 0.5, 0.8, 1.2, 2.0]
    fixed_fs   = 30.0
    fs_values  = [7.5, 10, 20, 30, 50, 100]
    fixed_tau  = 1.2

    results = []
    partial_path = os.path.join(data_dir, 'benchmark_params_partial.npz')

    print('\n--- Tau sensitivity ---')
    for tau in tau_values:
        print(f'  tau={tau}s...')
        try:
            dff, true_spikes, _, _, _, _ = generate_synthetic_data(
                n_cells=n_cells, fs=fixed_fs, duration=duration, tau=tau
            )
            true_events = [helpers.make_event_ground_truth(s, tau) for s in true_spikes]

            if run_mine:
                t0  = time.time()
                res = fMCSI.run_deconv(dff, params={'f': fixed_fs, 'p': 2, 'auto_stop': True},
                                       benchmark=True)
                t_my = time.time() - t0
                results.append(_row('Tau_Sensitivity', 'fMCSI', tau, fixed_fs, t_my,
                                    _metrics(true_spikes, res['optim_spikes'], true_events, fixed_fs),
                                    sweeps=np.mean(res['optim_nsamples']),
                                    n_cells=n_cells, duration=duration))
                print(f'    fMCSI: F1={results[-1]["F1"]:.3f}')

            if run_matlab:
                t0 = time.time()
                trad_spikes, _, _, sweeps = run_matlab_pnevMCMC(
                    dff, fs=fixed_fs, tau=tau, n_sweeps='auto')
                t_trad = time.time() - t0
                results.append(_row('Tau_Sensitivity', 'MATLAB', tau, fixed_fs, t_trad,
                                    _metrics(true_spikes, trad_spikes, true_events, fixed_fs),
                                    sweeps=np.mean(sweeps),
                                    n_cells=n_cells, duration=duration))
                print(f'    MATLAB: F1={results[-1]["F1"]:.3f}')

            if run_oasis:
                t0 = time.time()
                oas_spk, _ = _oasis_spikes(dff, fixed_fs, tau, n_cells)
                t_oasis = time.time() - t0
                results.append(_row('Tau_Sensitivity', 'OASIS', tau, fixed_fs, t_oasis,
                                    _metrics(true_spikes, oas_spk, true_events, fixed_fs),
                                    n_cells=n_cells, duration=duration))
                print(f'    OASIS: F1={results[-1]["F1"]:.3f}')

            if run_cascade:
                for _dev, _model in [('gpu', 'CASCADE_GPU'), ('cpu', 'CASCADE_CPU')]:
                    _, cascade_spikes, t_cascade = _run_cascade_inference(
                        dff, fixed_fs, data_dir, f'bench_tau_{tau}_{_dev}', device=_dev)
                    results.append(_row('Tau_Sensitivity', _model, tau, fixed_fs, t_cascade,
                                        _metrics(true_spikes, cascade_spikes, true_events, fixed_fs),
                                        n_cells=n_cells, duration=duration))
                    print(f'    CASCADE ({_dev.upper()}): F1={results[-1]["F1"]:.3f}')

        except Exception as exc:
            print(f'  Failed for tau={tau}: {exc}')
        _save_records(results, partial_path)

    print('\n--- Frame-rate sensitivity ---')
    for fs in fs_values:
        print(f'  fs={fs}Hz...')
        try:
            dff, true_spikes, _, _, _, _ = generate_synthetic_data(
                n_cells=n_cells, fs=fs, duration=duration, tau=fixed_tau
            )
            true_events = [helpers.make_event_ground_truth(s, fixed_tau) for s in true_spikes]

            if run_mine:
                t0  = time.time()
                res = fMCSI.run_deconv(dff, params={'f': fs, 'p': 2, 'auto_stop': True},
                                       benchmark=True)
                t_my = time.time() - t0
                results.append(_row('Fs_Sensitivity', 'fMCSI', fixed_tau, fs, t_my,
                                    _metrics(true_spikes, res['optim_spikes'], true_events, fs),
                                    sweeps=np.mean(res['optim_nsamples']),
                                    n_cells=n_cells, duration=duration))
                print(f'    fMCSI: F1={results[-1]["F1"]:.3f}')

            if run_matlab:
                t0 = time.time()
                trad_spikes, _, _, sweeps = run_matlab_pnevMCMC(
                    dff, fs=fs, tau=fixed_tau, n_sweeps='auto')
                t_trad = time.time() - t0
                results.append(_row('Fs_Sensitivity', 'MATLAB', fixed_tau, fs, t_trad,
                                    _metrics(true_spikes, trad_spikes, true_events, fs),
                                    sweeps=np.mean(sweeps),
                                    n_cells=n_cells, duration=duration))
                print(f'    MATLAB: F1={results[-1]["F1"]:.3f}')

            if run_oasis:
                t0 = time.time()
                oas_spk, _ = _oasis_spikes(dff, fs, fixed_tau, n_cells)
                t_oasis = time.time() - t0
                results.append(_row('Fs_Sensitivity', 'OASIS', fixed_tau, fs, t_oasis,
                                    _metrics(true_spikes, oas_spk, true_events, fs),
                                    n_cells=n_cells, duration=duration))
                print(f'    OASIS: F1={results[-1]["F1"]:.3f}')

            if run_cascade:
                for _dev, _model in [('gpu', 'CASCADE_GPU'), ('cpu', 'CASCADE_CPU')]:
                    _, cascade_spikes, t_cascade = _run_cascade_inference(
                        dff, fs, data_dir, f'bench_fs_{fs}_{_dev}', device=_dev)
                    results.append(_row('Fs_Sensitivity', _model, fixed_tau, fs, t_cascade,
                                        _metrics(true_spikes, cascade_spikes, true_events, fs),
                                        n_cells=n_cells, duration=duration))
                    print(f'    CASCADE ({_dev.upper()}): F1={results[-1]["F1"]:.3f}')

        except Exception as exc:
            print(f'  Failed for fs={fs}: {exc}')
        _save_records(results, partial_path)

    return


def benchmark_kurtosis(data_dir, run_oasis=True, run_matlab=True, run_mine=True,
                       run_cascade=True):

    n_cells  = 50
    duration = 300
    fs       = 30.0
    tau      = 1.2
    kurtosis_ranges = [
        (1.0,  10.0),
        (3.0,  20.0),
        (6.0,  30.0),
        (10.0, 50.0),
        (20.0, 100.0),
    ]

    results = []
    partial_path = os.path.join(data_dir, 'benchmark_kurtosis_partial.npz')

    for min_k, max_k in kurtosis_ranges:
        label = f'{min_k}-{max_k}'
        print(f'  Kurtosis range {label}...')
        try:
            dff, true_spikes, _, _, _, actual_kurt = generate_synthetic_data(
                n_cells=n_cells, fs=fs, duration=duration, tau=tau,
                target_kurtosis_range=(min_k, max_k)
            )
            mean_kurt   = float(np.mean(actual_kurt))
            true_events = [helpers.make_event_ground_truth(s, tau) for s in true_spikes]

            base = {'Experiment': 'Kurtosis_Sensitivity', 'Kurtosis_Range': label,
                    'N_Cells': n_cells, 'Duration': duration}

            def km(pred_spk):
                return _metrics(true_spikes, pred_spk, true_events, fs)

            if run_mine:
                t0  = time.time()
                res = fMCSI.run_deconv(dff, params={'f': fs, 'p': 2, 'auto_stop': True},
                                       benchmark=True)
                t_my = time.time() - t0
                results.append({**base, 'Model': 'fMCSI', 'Time': t_my,
                                 'Mean_Kurtosis': mean_kurt, **dict(zip(
                                     ['F1','Precision','Recall',
                                      'F1_window','Precision_window','Recall_window',
                                      'F1_event','Precision_event','Recall_event','COSMIC'],
                                     km(res['optim_spikes'])))})
                print(f'    fMCSI: F1={results[-1]["F1"]:.3f}')

            if run_matlab:
                t0 = time.time()
                trad_spikes, _, _, _ = run_matlab_pnevMCMC(dff, fs=fs, tau=tau, n_sweeps='auto')
                t_trad = time.time() - t0
                results.append({**base, 'Model': 'MATLAB', 'Time': t_trad,
                                 'Mean_Kurtosis': mean_kurt, **dict(zip(
                                     ['F1','Precision','Recall',
                                      'F1_window','Precision_window','Recall_window',
                                      'F1_event','Precision_event','Recall_event','COSMIC'],
                                     km(trad_spikes)))})
                print(f'    MATLAB: F1={results[-1]["F1"]:.3f}')

            if run_oasis:
                t0 = time.time()
                oas_spk, _ = _oasis_spikes(dff, fs, tau, n_cells)
                t_oasis = time.time() - t0
                results.append({**base, 'Model': 'OASIS', 'Time': t_oasis,
                                 'Mean_Kurtosis': mean_kurt, **dict(zip(
                                     ['F1','Precision','Recall',
                                      'F1_window','Precision_window','Recall_window',
                                      'F1_event','Precision_event','Recall_event','COSMIC'],
                                     km(oas_spk)))})
                print(f'    OASIS: F1={results[-1]["F1"]:.3f}')

            if run_cascade:
                for _dev, _model in [('gpu', 'CASCADE_GPU'), ('cpu', 'CASCADE_CPU')]:
                    _, cascade_spikes, t_cascade = _run_cascade_inference(
                        dff, fs, data_dir, f'bench_kurt_{label}_{_dev}', device=_dev)
                    results.append({**base, 'Model': _model, 'Time': t_cascade,
                                     'Mean_Kurtosis': mean_kurt, **dict(zip(
                                         ['F1','Precision','Recall',
                                          'F1_window','Precision_window','Recall_window',
                                          'F1_event','Precision_event','Recall_event','COSMIC'],
                                         km(cascade_spikes)))})
                    print(f'    CASCADE ({_dev.upper()}): F1={results[-1]["F1"]:.3f}')

        except Exception as exc:
            print(f'  Failed for kurtosis {label}: {exc}')
        _save_records(results, partial_path)

    return


def benchmark_firing_rate_sensitivity(data_dir, run_oasis=True, run_matlab=True,
                                      run_mine=True, run_cascade=True):
    
    n_cells  = 250
    duration = 300
    fs       = 30.0
    tau      = 1.2

    print(f'Generating synthetic data (n_cells={n_cells}, duration={duration}s)...')
    dff, true_spikes, _, _, firing_rates, _ = generate_synthetic_data(
        n_cells=n_cells, fs=fs, duration=duration, tau=tau
    )
    true_events = [helpers.make_event_ground_truth(s, tau) for s in true_spikes]
    npz_spikes  = {'true': true_spikes}
    npz_calcium = {}

    all_results  = []
    partial_path = os.path.join(data_dir, 'firing_rate_sensitivity_partial.npz')

    def per_cell(model, i, pred_spk_i, time_i):
        prec,   rec,   f1   = fMCSI.compute_accuracy_strict([true_spikes[i]], [pred_spk_i])
        prec_w, rec_w, f1_w = helpers.compute_accuracy_window([true_spikes[i]], [pred_spk_i])
        prec_e, rec_e, f1_e = helpers.compute_accuracy_window([true_events[i]], [pred_spk_i])
        cosmic = helpers.compute_cosmic([true_spikes[i]], [pred_spk_i], fs)
        return {
            'model': model, 'cell_id': i, 'firing_rate': float(firing_rates[i]),
            'precision': prec[0], 'recall': rec[0], 'f1': f1[0],
            'precision_window': prec_w[0], 'recall_window': rec_w[0], 'f1_window': f1_w[0],
            'precision_event': prec_e[0], 'recall_event': rec_e[0], 'f1_event': f1_e[0],
            'cosmic': cosmic[0], 'time': time_i,
        }

    if run_mine:
        print('\nRunning fMCSI...')
        try:
            t0  = time.time()
            res = fMCSI.run_deconv(dff, params={'f': fs, 'p': 2, 'auto_stop': True},
                                   benchmark=True)
            total_time = time.time() - t0
            for i in range(n_cells):
                all_results.append(per_cell('fMCSI', i, res['optim_spikes'][i],
                                             res['optim_times_per_cell'][i]))
            npz_spikes['my_method'] = res['optim_spikes']
            print(f'  Finished in {total_time:.1f}s')
        except Exception as exc:
            print(f'  fMCSI failed: {exc}')
        _save_records(all_results, partial_path)

    if run_matlab:
        print('\nRunning MATLAB...')
        trad_spikes_all = []
        for i in range(n_cells):
            print(f'  Processing cell {i+1}/{n_cells}...', end='\r')
            try:
                t0 = time.time()
                trad_spk, _, _, _ = run_matlab_pnevMCMC(
                    dff[i:i+1], fs=fs, tau=tau, n_sweeps='auto')
                time_taken = time.time() - t0
                trad_spikes_all.append(trad_spk[0])
                all_results.append(per_cell('MATLAB', i, trad_spk[0], time_taken))
            except Exception as exc:
                print(f'\n  MATLAB failed on cell {i}: {exc}')
                trad_spikes_all.append(np.array([]))
        print('\n  Finished.')
        npz_spikes['trad_mcmc'] = trad_spikes_all
        _save_records(all_results, partial_path)

    if run_oasis:
        print('\nRunning OASIS...')
        try:
            t0 = time.time()
            oas_spk, oas_cal = _oasis_spikes(dff, fs, tau, n_cells)
            total_time = time.time() - t0
            for i in range(n_cells):
                all_results.append(per_cell('OASIS', i, oas_spk[i], np.nan))
            npz_spikes['oasis']  = oas_spk
            npz_calcium['oasis'] = np.array(oas_cal)
            print(f'  Finished in {total_time:.1f}s')
        except Exception as exc:
            print(f'  OASIS failed: {exc}')
        _save_records(all_results, partial_path)

    if run_cascade:
        for _dev, _model in [('gpu', 'CASCADE_GPU'), ('cpu', 'CASCADE_CPU')]:
            print(f'\nRunning CASCADE (subprocess, {_dev.upper()})...')
            try:
                _, cascade_spikes, t_cascade = _run_cascade_inference(
                    dff, fs, data_dir, f'bench_firerate_cascade_{_dev}', device=_dev)
                for i in range(n_cells):
                    all_results.append(per_cell(_model, i, cascade_spikes[i], np.nan))
                npz_spikes[f'cascade_{_dev}'] = cascade_spikes
                print(f'  CASCADE ({_dev.upper()}) finished in {t_cascade:.1f}s')
            except Exception as exc:
                print(f'  CASCADE ({_dev.upper()}) failed: {exc}')
            _save_records(all_results, partial_path)

    if all_results:
        npz_save = {'dff': dff, 'fs': fs, 'tau': tau,
                    'firing_rates': np.array(firing_rates)}
        for k, v in npz_spikes.items():
            npz_save[f'spikes_{k}'] = np.array(v, dtype=object)
        for k, v in npz_calcium.items():
            npz_save[f'calcium_{k}'] = v
        np.savez(os.path.join(data_dir, 'firing_rate_sensitivity_traces.npz'), **npz_save)

    return

def run_test(data_dir='.', run_fmcsi=True, run_matlab=True,
             run_oasis=True, run_cascade=True):
    """Run all benchmark suites and write partial JSON files to data_dir."""
    os.makedirs(data_dir, exist_ok=True)
    kw = dict(run_oasis=run_oasis, run_matlab=run_matlab,
              run_mine=run_fmcsi, run_cascade=run_cascade)
    print('=== Sweeps benchmark ===')
    benchmark_sweeps(data_dir, **kw)
    print('\n=== Scalability benchmark ===')
    benchmark_scalability(data_dir, **kw)
    print('\n=== Parameter sensitivity benchmark ===')
    benchmark_params(data_dir, **kw)
    print('\n=== Kurtosis sensitivity benchmark ===')
    benchmark_kurtosis(data_dir, **kw)
    print('\n=== Firing-rate sensitivity benchmark ===')
    benchmark_firing_rate_sensitivity(data_dir, **kw)
    print('\nTest mode complete.')


def _fbeta(prec, rec):
    p  = np.asarray(prec, dtype=float)
    r  = np.asarray(rec,  dtype=float)
    b2 = BETA ** 2
    denom = b2 * p + r
    with np.errstate(divide='ignore', invalid='ignore'):
        return np.where(denom > 0, (1 + b2) * p * r / denom, 0.0)


def _fit_scaling(x, y):
    if len(x) < 3:
        return np.nan, np.nan, 'N/A'
    x, y = np.array(x, float), np.array(y, float)
    def r2(y_pred):
        ss_res = np.sum((y - y_pred) ** 2)
        ss_tot = np.sum((y - np.mean(y)) ** 2)
        return 0.0 if ss_tot < 1e-9 else 1 - ss_res / ss_tot
    try:
        r2_lin  = r2(np.poly1d(np.polyfit(x, y, 1))(x))
    except Exception:
        r2_lin  = -np.inf
    try:
        r2_poly = r2(np.poly1d(np.polyfit(x, y, 2))(x))
    except Exception:
        r2_poly = -np.inf
    conclusion = 'Linear' if r2_lin >= r2_poly - 0.02 else 'Polynomial'
    return r2_lin, r2_poly, conclusion


def _set_three_ticks_x(ax):
    all_x = [v for line in ax.get_lines() for v in line.get_xdata()]
    if len(all_x) < 2:
        return
    lo, hi = float(np.nanmin(all_x)), float(np.nanmax(all_x))
    if lo == hi:
        return
    ax.set_xticks([lo, (lo + hi) / 2, hi])
    ax.set_xticklabels([f'{v:.3g}' for v in [lo, (lo + hi) / 2, hi]])


def _filter_cascade_shared_x(cascade_sub, full_tbl, xcol):
    non_cascade = np.array([not str(m).startswith('CASCADE') for m in full_tbl['Model']])
    vals = full_tbl[xcol][non_cascade].astype(float)
    other_x = set(vals[~np.isnan(vals)].tolist())
    mask = np.isin(cascade_sub[xcol].astype(float), list(other_x))
    return {k: v[mask] for k, v in cascade_sub.items()}


def plot_figure(data_dir='.'):

    partial_files = [
        'benchmark_sweeps_partial.npz',
        'benchmark_scalability_partial.npz',
        'benchmark_params_partial.npz',
        'benchmark_kurtosis_partial.npz',
        'firing_rate_sensitivity_partial.npz',
    ]

    all_records = []
    for fname in partial_files:
        fpath = os.path.join(data_dir, fname)
        if os.path.exists(fpath):
            try:
                tbl = _load_records(fpath)
                all_records.append(tbl)
                print(f'Loaded {fpath}  ({_tbl_len(tbl)} rows)')
            except Exception as exc:
                print(f'Error reading {fpath}: {exc}')

    if not all_records:
        raise RuntimeError(f'No partial benchmark files found in {data_dir}. '
                           'Run --mode test first.')

    combined = _tbl_concat(all_records)

    f1_col   = 'F1'        if USE_STRICT_ACCURACY else 'F1_window'
    prec_col = 'Precision' if USE_STRICT_ACCURACY else 'Precision_window'
    rec_col  = 'Recall'    if USE_STRICT_ACCURACY else 'Recall_window'

    scaling_stats = []

    mosaic = [
        ['sweeps',  'cells',   'duration', 'legend'],
        ['tau_p',   'tau_r',   'fs_p',     'fs_r'  ],
        ['kurt_f',  'kurt_p',  'kurt_r',   'kurt_cosmic'],
    ]
    fig, axes = plt.subplot_mosaic(mosaic, figsize=(7, 4.5), dpi=300)

    for model in ['fMCSI', 'MATLAB']:
        m_rows = _tbl_filter(combined, 'Model', model)
        if _tbl_len(m_rows) == 0:
            continue
        subset = _tbl_sort(_tbl_filter(m_rows, 'Experiment', 'Sweeps'), 'Sweeps')
        if _tbl_len(subset) > 0:
            axes['sweeps'].plot(subset['Sweeps'], subset['Time'] / 60.0,
                                '.-', label=model, color=COLORS.get(model, 'k'))
            r2l, r2p, c = _fit_scaling(subset['Sweeps'], subset['Time'])
            scaling_stats.append({'Experiment': 'Sweeps', 'Model': model,
                                   'Variable': 'Sweeps', 'Lin_R2': r2l,
                                   'Poly_R2': r2p, 'Conclusion': c})
    axes['sweeps'].set_xlabel('# sweeps')
    axes['sweeps'].set_ylabel('compute time (min)')

    for model in ['MATLAB', 'OASIS', 'CASCADE_GPU', 'CASCADE_CPU', 'fMCSI']:
        m_rows = _tbl_filter(combined, 'Model', model)
        if _tbl_len(m_rows) == 0:
            continue
        subset = _tbl_sort(_tbl_filter(m_rows, 'Experiment', 'Cell_Scaling'), 'N_Cells')
        if model.startswith('CASCADE'):
            subset = _filter_cascade_shared_x(
                subset, _tbl_filter(combined, 'Experiment', 'Cell_Scaling'), 'N_Cells')
        if _tbl_len(subset) > 0:
            axes['cells'].plot(subset['N_Cells'], subset['Time'] / 3600.,
                               '.-', label=model, color=COLORS.get(model, 'k'))
            r2l, r2p, c = _fit_scaling(subset['N_Cells'], subset['Time'])
            scaling_stats.append({'Experiment': 'Cell_Scaling', 'Model': model,
                                   'Variable': 'N_Cells', 'Lin_R2': r2l,
                                   'Poly_R2': r2p, 'Conclusion': c})
    axes['cells'].set_xlabel('# cells')
    axes['cells'].set_ylabel('compute time (hours)')

    for model in ['MATLAB', 'OASIS', 'CASCADE_GPU', 'CASCADE_CPU', 'fMCSI']:
        m_rows = _tbl_filter(combined, 'Model', model)
        if _tbl_len(m_rows) == 0:
            continue
        subset = _tbl_sort(_tbl_filter(m_rows, 'Experiment', 'Duration_Scaling'), 'Duration')
        if model.startswith('CASCADE'):
            subset = _filter_cascade_shared_x(
                subset, _tbl_filter(combined, 'Experiment', 'Duration_Scaling'), 'Duration')
        if _tbl_len(subset) > 0:
            axes['duration'].plot(subset['Duration'] / 60., subset['Time'] / 3600.,
                                  '.-', label=model, color=COLORS.get(model, 'k'))
            r2l, r2p, c = _fit_scaling(subset['Duration'], subset['Time'])
            scaling_stats.append({'Experiment': 'Duration_Scaling', 'Model': model,
                                   'Variable': 'Duration', 'Lin_R2': r2l,
                                   'Poly_R2': r2p, 'Conclusion': c})
    axes['duration'].set_xlabel('recording duration (min)')
    axes['duration'].set_ylabel('compute time (hours)')
    _set_three_ticks_x(axes['duration'])

    for model in ['MATLAB', 'CASCADE_GPU', 'CASCADE_CPU', 'OASIS', 'fMCSI']:
        m_rows = _tbl_filter(combined, 'Model', model)
        if _tbl_len(m_rows) == 0:
            continue
        for exp, xcol, ax_p, ax_r in [
            ('Tau_Sensitivity', 'Tau', 'tau_p', 'tau_r'),
            ('Fs_Sensitivity',  'Fs',  'fs_p',  'fs_r'),
        ]:
            subset = _tbl_sort(_tbl_filter(m_rows, 'Experiment', exp), xcol)
            if model.startswith('CASCADE'):
                subset = _filter_cascade_shared_x(
                    subset, _tbl_filter(combined, 'Experiment', exp), xcol)
            if _tbl_len(subset) > 0:
                axes[ax_p].plot(subset[xcol], subset[prec_col], '.-',
                                color=COLORS.get(model, 'k'))
                axes[ax_r].plot(subset[xcol], subset[rec_col],  '.-',
                                color=COLORS.get(model, 'k'))

    for ax_key, xlabel, ylabel in [
        ('tau_p', 'tau (s)', 'Precision'), ('tau_r', 'tau (s)', 'Recall'),
        ('fs_p',  'Hz',      'precision'), ('fs_r',  'Hz',      'recall'),
    ]:
        axes[ax_key].set_xlabel(xlabel)
        axes[ax_key].set_ylabel(ylabel)
        axes[ax_key].set_ylim(-0.05, 1.05)
        _set_three_ticks_x(axes[ax_key])

    for model in ['MATLAB', 'CASCADE_GPU', 'CASCADE_CPU', 'OASIS', 'fMCSI']:
        m_rows = _tbl_filter(combined, 'Model', model)
        if _tbl_len(m_rows) == 0:
            continue
        subset = _tbl_sort(_tbl_filter(m_rows, 'Experiment', 'Kurtosis_Sensitivity'), 'Mean_Kurtosis')
        if model.startswith('CASCADE'):
            subset = _filter_cascade_shared_x(
                subset, _tbl_filter(combined, 'Experiment', 'Kurtosis_Sensitivity'), 'Mean_Kurtosis')
        if _tbl_len(subset) > 0:
            fb = _fbeta(subset[prec_col], subset[rec_col])
            axes['kurt_f'].plot(subset['Mean_Kurtosis'], fb, '.-',
                                color=COLORS.get(model, 'k'))
            axes['kurt_p'].plot(subset['Mean_Kurtosis'], subset[prec_col], '.-',
                                color=COLORS.get(model, 'k'))
            axes['kurt_r'].plot(subset['Mean_Kurtosis'], subset[rec_col],  '.-',
                                color=COLORS.get(model, 'k'))
            axes['kurt_cosmic'].plot(subset['Mean_Kurtosis'], subset['COSMIC'], '.-',
                                     color=COLORS.get(model, 'k'))

    for ax_key, ylabel in [
        ('kurt_f',      r'$F_\beta$'), ('kurt_p', 'precision'),
        ('kurt_r',      'recall'),     ('kurt_cosmic', 'CosMIC score'),
    ]:
        axes[ax_key].set_xlabel('mean kurtosis')
        axes[ax_key].set_ylabel(ylabel)
        axes[ax_key].set_ylim(-0.05, 1.05)
        _set_three_ticks_x(axes[ax_key])

    ax_leg = axes['legend']
    ax_leg.axis('off')
    _legend_labels = {
        'fMCSI': 'fMCSI', 'MATLAB': 'MATLAB', 'OASIS': 'OASIS',
        'CASCADE_GPU': 'CASCADE (GPU)', 'CASCADE_CPU': 'CASCADE (CPU)',
    }
    handles = [plt.Line2D([0], [0], color=COLORS[m], marker='.', linestyle='-',
                          label=_legend_labels[m])
               for m in ['fMCSI', 'MATLAB', 'OASIS', 'CASCADE_GPU', 'CASCADE_CPU']]
    ax_leg.legend(handles=handles, loc='center', frameon=False, fontsize=9)

    plt.tight_layout()

    print('\nScaling statistics:')
    print(f'{"Experiment":<22} {"Model":<12} {"Variable":<10} '
          f'{"Lin R²":<8} {"Poly R²":<8} {"Fit":<12}')
    print('-' * 80)
    for s in scaling_stats:
        print(f'{s["Experiment"]:<22} {s["Model"]:<12} {s["Variable"]:<10} '
              f'{s["Lin_R2"]:<8.4f} {s["Poly_R2"]:<8.4f} {s["Conclusion"]:<12}')

    for ext in ('png', 'svg'):
        out = os.path.join(data_dir, f'figure2.{ext}')
        fig.savefig(out, dpi=300, bbox_inches='tight')
        print(f'Saved -> {out}')
    plt.close(fig)


if __name__ == '__main__':

    parser = argparse.ArgumentParser(
        description='Figure 2: scaling and sensitivity benchmarks'
    )
    parser.add_argument('--mode', required=True, choices=['test', 'plot'],
                        help='"test" runs benchmarks; "plot" generates the figure')
    parser.add_argument('--data-dir', default='.',
                        help='Directory for reading/writing result files (default: .)')
    parser.add_argument('--no-fmcsi',   action='store_true', help='Skip fMCSI')
    parser.add_argument('--no-matlab',  action='store_true', help='Skip MATLAB')
    parser.add_argument('--no-oasis',   action='store_true', help='Skip OASIS')
    parser.add_argument('--no-cascade', action='store_true', help='Skip CASCADE')
    args = parser.parse_args()

    if args.mode == 'test':
        run_test(
            data_dir    = args.data_dir,
            run_fmcsi   = not args.no_fmcsi,
            run_matlab  = not args.no_matlab,
            run_oasis   = not args.no_oasis,
            run_cascade = not args.no_cascade,
        )
    else:
        plot_figure(data_dir=args.data_dir)
