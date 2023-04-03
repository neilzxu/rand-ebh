from typing import Any, Callable, List, Optional, Union

import copy
import hashlib
from itertools import product
import json
import os
import pickle
import random
import re

import dill
import multiprocess as mp
import numpy as np
import pandas as pd
from tqdm import tqdm

from alg import get_alg
from data import get_data_method
from pvalue import get_pvalue
from utils import build_registry

SALT = 222
HASH_SIZE = 32


def load_results(out_dir: str):
    result_paths = [
        filename for filename in os.listdir(out_dir)
        if re.fullmatch(r'.*\.pkl', filename)
    ]
    results = []
    for filename in result_paths:
        results.append(dill.load(open(os.path.join(out_dir, filename), 'rb')))
    return results


def setup_df(results, make_datapoints):
    datapoints = make_datapoints(results)
    df = pd.DataFrame.from_records(datapoints)
    return df


def generate_path(data_spec, alg_spec):

    def make_name(key, value):
        if isinstance(value, float):
            return f'{key}={value:.3f}'
        else:
            return f'{key}={value}'

    msg = json.dumps([data_spec, alg_spec])
    # bytes is called on an iterable
    hasher = hashlib.blake2b(salt=bytes([SALT]), digest_size=HASH_SIZE)
    hasher.update(msg.encode('utf-8'))
    return str(hasher.hexdigest())


def run_omt(alg_fn, alg_spec, pvalue_list):
    state = alg_fn(**(alg_spec['kwargs']))
    rejset = state.run_fdr(pvalue_list).astype(bool)
    return rejset, state


def save_result_to_disk(result, final_path):
    with open(final_path, 'wb') as out_f:
        # pickle.dump(result, out_f)
        dill.dump(result, out_f)


def exec_exp(data_spec,
             alg_spec,
             load_data: Union[str, Any],
             save_result,
             name=None,
             out_dir='comp_exp',
             pool=None):
    if isinstance(load_data, str):
        with open(load_data, 'rb') as in_f:
            data, alternates = pickle.load(in_f)
    else:
        data, alternates = load_data

    path = generate_path(data_spec, alg_spec)

    final_path = f'{out_dir}/{path}.pkl'
    instances = []
    rejsets = []
    trials = data_spec['kwargs']['trials']
    alg_fn = get_alg(alg_spec['method'])
    pvalues = get_pvalue(alg_spec['pvalue']['method'])(
        x=data, **(alg_spec['pvalue']['kwargs']))

    if 'SEEDS' not in alg_spec['kwargs']:
        args_list = [(alg_fn, alg_spec, pvalue) for pvalue in pvalues]
    else:
        seeds = alg_spec['kwargs']['SEEDS']
        assert len(set(seeds)) == len(seeds)
        assert len(seeds) == trials
        base_alg_kwargs = {
            k: v
            for k, v in alg_spec['kwargs'].items() if k != 'SEEDS'
        }
        args_list = [(alg_fn, {
            'method': alg_spec['method'],
            'kwargs': {
                **base_alg_kwargs
            },
            'seed': seed
        }, pvalue) for seed, pvalue in zip(seeds, pvalues)]

    if pool is None:
        rejsets, instances = zip(*[run_omt(*args) for args in args_list])
    else:
        res = list(
            tqdm(pool.imap(lambda args: run_omt(*args), args_list),
                 total=trials,
                 desc="Running trials",
                 leave=False))
        rejsets, instances = zip(*res)

    result = {
        'name': name,
        'alg_spec': alg_spec,
        'data_spec': data_spec,
        'pvalues': pvalues,
        'alternates': alternates,
        'rejsets': np.stack(rejsets),
        'instances': instances
    }
    if save_result:
        save_result_to_disk(result, final_path)
    return result


ExpFn = Callable[[int, str], None]

_EXP_DISPATCH, register_experiment, get_experiment = build_registry()


def get_data_filename(data_spec):
    return f'{generate_path(data_spec, {})}.pkl'


def make_data(data_spec, data_path, load=True):
    if os.path.exists(data_path) and load:
        with open(data_path, 'rb') as in_f:
            data = pickle.load(in_f)
    else:
        data = get_data_method(data_spec['method'])(**(data_spec['kwargs']))
        with open(data_path, 'wb') as out_f:
            pickle.dump(data, out_f)
    if not load:
        data = None
    return data


def run_exp(out_dir: str,
            data_specs: List[Any],
            alg_specs: Optional[List[Any]] = None,
            alg_names: Optional[List[str]] = None,
            data_alg_specs: Optional[List[Any]] = None,
            data_alg_names: Optional[List[str]] = None,
            processes: int = 20,
            cache_data: bool = True,
            save_result: bool = True,
            long_exp: bool = False) -> List[Any]:
    """Run experiments corresponding to product of configs for algs and data.

    Concurrent processing with processes sized thread pool and saves
    results of each trial in hash pathname inside out_dir.
    """

    OUT_DIR = out_dir
    if not os.path.exists(OUT_DIR):
        os.makedirs(OUT_DIR)
    print(f'Running experiment for directory: {out_dir}')

    data_dir = os.path.join(OUT_DIR, 'data')
    if not os.path.exists(data_dir):
        os.makedirs(data_dir)

    # Check data existences
    data_paths = [
        os.path.join(data_dir, get_data_filename(data_spec))
        for data_spec in data_specs
    ]
    if data_alg_specs is not None:
        flat_specs, flat_names = data_alg_specs, data_alg_names
    else:
        flat_specs, flat_names = list(
            product(range(len(data_specs)),
                    alg_specs)), alg_names * len(data_specs)

    # Hash all args and check if hashing failed to get paths for saving the
    # results of each trial.
    hashes = [
        generate_path(data_specs[data_idx], alg_spec)
        for data_idx, alg_spec in flat_specs
    ]
    # Check if hashing failed
    assert len(set(hashes)) == len(hashes)

    # Only run experiments with no results saved yet.
    if save_result:
        runnable_idxs = [
            idx for idx, hash_str in enumerate(hashes)
            if not os.path.exists(os.path.join(out_dir, f'{hash_str}.pkl'))
        ]
    else:
        runnable_idxs = list(range(len(hashes)))

    # Save data objects in memory if caching data
    if cache_data and len(runnable_idxs) > 0:
        data_args = list(zip(data_specs, data_paths))
    else:
        # Only make data that hasn't been made yet (or load/make all data)
        needed_data_idxs = {
            flat_specs[idx][0]
            for idx in runnable_idxs
            if not os.path.exists(data_paths[flat_specs[idx][0]])
        }
        data_args = [(data_specs[i], data_paths[i]) for i in needed_data_idxs]

    with mp.get_context("spawn").Pool(processes) as p:
        # Make each data
        data_cache = list(
            tqdm(p.imap(
                lambda data_pair: make_data(*data_pair, load=cache_data),
                data_args),
                 total=len(data_args),
                 desc=f"Data generation (cache_data: {cache_data})"))

        # Build and run experiments that need to be run
        runnable_kwargs = [{
            'data_spec':
            data_specs[flat_specs[idx][0]],
            'alg_spec':
            flat_specs[idx][1],
            'load_data':
            (data_cache if cache_data else data_paths)[flat_specs[idx][0]],
            'name':
            flat_names[idx],
            'out_dir':
            out_dir
        } for idx in runnable_idxs]
    with mp.get_context("spawn").Pool(processes) as p:
        if long_exp:
            res = [
                exec_exp(**kwargs, pool=p, save_result=save_result)
                for kwargs in tqdm(runnable_kwargs, desc="Experimentation")
            ]
        else:
            permuted_kwargs = copy.copy(runnable_kwargs)
            random.seed(322)
            random.shuffle(permuted_kwargs)
            res = list(
                tqdm(p.imap(
                    lambda kwargs: exec_exp(**kwargs, save_result=save_result),
                    permuted_kwargs),
                     total=len(runnable_kwargs),
                     desc="Experimentation"))
        return res
