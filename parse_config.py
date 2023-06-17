# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.


import os
import logging
import pdb
from pathlib import Path
from functools import reduce
from operator import getitem
from datetime import datetime
# import pdb; pdb.set_trace()
from logger import setup_logging
from utils import read_json, write_json
import time
import inspect


class ConfigParser:
    def __init__(self, args, options='', timestamp=True, test=False, eval_mode=None):
        # parse default and custom cli options
        for opt in options:
            args.add_argument(*opt.flags, default=None, type=opt.type)
        args = args.parse_args()
        self.args = args
        if args.device:
            os.environ["CUDA_VISIBLE_DEVICES"] = args.device
        if args.resume is None:
            msg_no_cfg = "Configuration file need to be specified. Add '-c config.json', for example."
            assert args.config is not None, msg_no_cfg
            self.cfg_fname = Path(args.config)
            config = read_json(self.cfg_fname)
            self.resume = None
        else:
            self.resume = Path(args.resume)
            resume_cfg_fname = self.resume.parent / 'config.json'
            # Use absolute path if there are errors here
            if eval_mode == "epic":
                resume_cfg_fname = Path('configs/eval/epic.json')
            if eval_mode == "charades":
                resume_cfg_fname = Path('configs/eval/charades.json')

            config = read_json(resume_cfg_fname)
            if args.config is not None:
                config.update(read_json(Path(args.config)))

        # load config file and apply custom cli options
        self._config = _update_config(config, options, args)

        # set save_dir where trained model and log will be saved.
        save_dir = Path(self.config['trainer']['save_dir'])
        # timestamp = datetime.now().strftime(r'%m%d_%H%M%S') if timestamp else ''
        timestamp = datetime.now().strftime(r'%m%d_%H:%M:%S') if timestamp else ''
        print('Starting timestamp is {}'.format(timestamp))
        if 'SLURM_JOBID' in os.environ:
            #current_model_save_dir = timestamp + '_{}'.format(os.environ['JOB_NAME'])
            current_model_save_dir = os.environ['SLURM_JOBID']
        else:
            current_model_save_dir = timestamp

        exper_name = self.config['name']

        self._save_dir = save_dir / 'models' /  current_model_save_dir
        self._web_log_dir = save_dir / 'web' /  current_model_save_dir
        self._log_dir = save_dir / 'log' /  current_model_save_dir
        self._tf_dir = save_dir / 'tf' / current_model_save_dir

        if not test:
            self.save_dir.mkdir(parents=True, exist_ok=True)
            self.log_dir.mkdir(parents=True, exist_ok=True)
            self._tf_dir.mkdir(parents=True, exist_ok=True)

        # if set, remove all previous experiments with the current config
        if vars(args).get("purge_exp_dir", False):
            for dirpath in (self._save_dir, self._log_dir, self._web_log_dir):
                config_dir = dirpath.parent
                existing = list(config_dir.glob("*"))
                print(f"purging {len(existing)} directories from config_dir...")
                tic = time.time()
                os.system(f"rm -rf {config_dir}")
                print(f"Finished purge in {time.time() - tic:.3f}s")

        # save updated config file to the checkpoint dir
        if not test:
            if args.rank == 0:
                write_json(self.config, self.save_dir / 'config.json')

            # configure logging module
            setup_logging(self.log_dir)
            self.log_levels = {
                0: logging.WARNING,
                1: logging.INFO,
                2: logging.DEBUG
            }

    def initialize(self, name, module,  *args, index=None, **kwargs):
        """
        finds a function handle with the name given as 'type' in config, and returns the
        instance initialized with corresponding keyword args given as 'args'.
        """
        if index is None:
            module_name = self[name]['type']
            module_args = dict(self[name]['args'])
            assert all([k not in module_args for k in kwargs]), 'Overwriting kwargs given in config file is not allowed'
            module_args.update(kwargs)
        else:
            module_name = self[name][index]['type']
            module_args = dict(self[name][index]['args'])

        # if parameter not in config subdict, then check if it's in global config.
        signature = inspect.signature(getattr(module, module_name).__init__)
        print(module_name)
        for param in signature.parameters.keys():
            if param not in module_args and param in self.config:
                module_args[param] = self[param]
            if module_name == 'FrozenInTime' and param == 'args':
                module_args[param] = self.args
            if module_name == 'MultiDistTextVideoDataLoader' and param == 'args':
                module_args[param] = self.args

        return getattr(module, module_name)(*args, **module_args)

    def __getitem__(self, name):
        return self.config[name]

    def get_logger(self, name, verbosity=2):
        msg_verbosity = 'verbosity option {} is invalid. Valid options are {}.'.format(verbosity,
                                                                                       self.log_levels.keys())
        assert verbosity in self.log_levels, msg_verbosity
        logger = logging.getLogger(name)
        logger.setLevel(self.log_levels[verbosity])
        return logger

    # setting read-only attributes
    @property
    def config(self):
        return self._config

    @property
    def save_dir(self):
        return self._save_dir

    @property
    def log_dir(self):
        return self._log_dir

    @property
    def tf_dir(self):
        return self._tf_dir

# helper functions used to update config dict with custom cli options
def _update_config(config, options, args):
    for opt in options:
        value = getattr(args, _get_opt_name(opt.flags))
        if value is not None:
            _set_by_path(config, opt.target, value)
    return config


def _get_opt_name(flags):
    for flg in flags:
        if flg.startswith('--'):
            return flg.replace('--', '')
    return flags[0].replace('--', '')


def _set_by_path(tree, keys, value):
    """Set a value in a nested object in tree by sequence of keys."""
    _get_by_path(tree, keys[:-1])[keys[-1]] = value


def _get_by_path(tree, keys):
    """Access a nested object in tree by sequence of keys."""
    return reduce(getitem, keys, tree)
