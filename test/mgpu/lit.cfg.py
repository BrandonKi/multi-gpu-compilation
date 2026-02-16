# check-mgpu suite: only multigpu/mgpu dialect tests.

import os

_config_dir = os.path.dirname(os.path.abspath(__file__))
_site_cfg = os.path.join(_config_dir, '..', 'lit.site.cfg.py')
lit_config.load_config(config, _site_cfg)

config.test_source_root = _config_dir
config.test_exec_root = _config_dir

config.name = 'polygeist-mgpu'
