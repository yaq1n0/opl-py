"""opl-py: A typed Python SDK for OpenPowerlifting data.

This project uses data from the OpenPowerlifting project.
You may download a copy of the data at https://data.openpowerlifting.org.

OpenPowerlifting data is contributed to the public domain.
"""

__version__ = "0.1.0"

from opl.core.client import OPL
from opl.core.enums import Equipment, Event, Sex

__all__ = ["OPL", "Equipment", "Event", "Sex"]
