REGISTRY = {}

from .basic_controller import BasicMAC
from .n_controller import NMAC
from .cql_basic_controller import CQLBasicMAC

REGISTRY["basic_mac"] = BasicMAC
REGISTRY["cql_basic_mac"] = CQLBasicMAC
REGISTRY["n_mac"] = NMAC