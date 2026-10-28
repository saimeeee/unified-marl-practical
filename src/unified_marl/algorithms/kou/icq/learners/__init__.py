from .q_learner import QLearner
from .offpg_learner import OffPGLearner
from .bc_learner import BCLearner
from .bcmapt_learner import BCMAPTLearner
from .cpl_learner import CPLLearner
from .dppo_learner import DPPOLearner
from .madpo_learner import MADPOLearner
from .madpo_aux_learner import MADPOauxLearner
from .cql_learner import CQLearner
from .iql_learner import IQLLearner


REGISTRY = {}

REGISTRY["q_learner"] = QLearner
REGISTRY["offpg_learner"] = OffPGLearner
REGISTRY["bc_learner"] = BCLearner
REGISTRY["bcmapt_learner"] = BCMAPTLearner
REGISTRY["cpl_learner"] = CPLLearner
REGISTRY["dppo_learner"] = DPPOLearner
REGISTRY["madpo_learner"] = MADPOLearner
REGISTRY['madpo_aux_learner'] = MADPOauxLearner
REGISTRY["cql_learner"] = CQLearner
REGISTRY["iql_learner"] = IQLLearner
