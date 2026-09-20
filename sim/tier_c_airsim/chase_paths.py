"""Put the repo's chase_* packages on sys.path so the Tier-C scripts import
the ONE shared contract (chase_gym / chase_train / chase_eval) without the
caller having to set PYTHONPATH -- the tier_c venv does not pip-install the
ament packages, and a forgotten PYTHONPATH is a silent `No module named
chase_gym`. Import this module BEFORE any chase_* import.

Layout: this file lives in sim/tier_c_airsim; the packages are ament-python
dirs two levels up under workspace/src/<pkg>/<pkg>.
"""
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
_SRC = os.path.join(_HERE, '..', '..', 'workspace', 'src')
for _pkg in ('chase_gym', 'chase_train', 'chase_eval'):
    _p = os.path.abspath(os.path.join(_SRC, _pkg))
    if os.path.isdir(_p) and _p not in sys.path:
        sys.path.insert(0, _p)
