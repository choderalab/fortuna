"""
fortuna
Methodologies for adaptive sampling
"""

# Add imports here
from .bandits import *
from .choice import *

# Handle versioneer
from ._version import get_versions
versions = get_versions()
__version__ = versions['version']
__git_revision__ = versions['full-revisionid']
del get_versions, versions
