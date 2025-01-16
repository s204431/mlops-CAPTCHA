import os
from pathlib import Path

_ROOT = Path(os.path.dirname(__file__)).parent.parent.absolute()  # root of project
if os.path.exists('/.dockerenv'):
    _ROOT = '/'
