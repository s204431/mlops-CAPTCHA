import os
from pathlib import Path

# root of project
_ROOT = Path(os.path.dirname(__file__)).parent.parent.absolute()

# in the case that the code is running in a docker container
if os.path.exists("/.dockerenv"):
    _ROOT = "/"
