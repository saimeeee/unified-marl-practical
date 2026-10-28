REGISTRY = {}

from .rnn_agent import RNNAgent
from .grnn_agent import GRNNAgent
from .rnn_ns_agent import RNNNSAgent
from .n_rnn_agent import NRNNAgent

REGISTRY["rnn"] = RNNAgent
REGISTRY["grnn"] = GRNNAgent
REGISTRY["rnn_ns"] = RNNNSAgent
REGISTRY["n_rnn"] = NRNNAgent