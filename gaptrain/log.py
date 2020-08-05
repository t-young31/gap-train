import logging
import coloredlogs

import os

try:
    ll = os.environ['GT_LOG_LEVEL']

except KeyError:
    # Default to debugging log level
    ll = 'DEBUG'

logging.basicConfig(level=getattr(logging, ll),
                    format='%(name)-12s: %(levelname)-8s %(message)s')
logger = logging.getLogger(__name__)

# Try and use colourful logs
try:
    coloredlogs.install(level=getattr(logging, ll), logger=logger)
except ImportError:
    pass
