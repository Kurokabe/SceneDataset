# read version from installed package
from importlib.metadata import version

__version__ = version("scenedataset")

from .scenedataset import SceneDataset
