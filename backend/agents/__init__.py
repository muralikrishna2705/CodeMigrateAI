from .base import AgentResult, BaseAgent, ReflectionResult
from .analyzer_agent import AnalyzerAgent
from .critic_agent import CriticAgent
from .deep_analyzer_agent import DeepAnalyzerAgent
from .fixer_agent import FixerAgent
from .migrator_agent import MigratorAgent
from .orchestrator_agent import OrchestratorAgent
from .planner_agent import PlannerAgent
from .reflector_agent import ReflectorAgent
from .retriever_agent import RetrieverAgent
from .validator_agent import ValidatorAgent

# Import the runtime agents too, so any import that touches the `agents` package
# (including a bare `from agents.base import BaseAgent`) triggers full
# AgentMeta auto-registration — this is safe because `.base` above has
# already fully finished importing by this point, so the runtime modules'
# `from agents.base import BaseAgent` resolve against a complete module. The
# DispatcherAgent (dynamic routing) and RuntimeValidatorAgent (external service
# validation) run as real graph nodes; Provider/Runtime are infra, not agents.
import runtime.agent_dispatcher  # noqa: F401
import runtime.agent_observer  # noqa: F401
import runtime.agent_validator  # noqa: F401

__all__ = [
    "BaseAgent",
    "AgentResult",
    "ReflectionResult",
    "AnalyzerAgent",
    "CriticAgent",
    "DeepAnalyzerAgent",
    "FixerAgent",
    "MigratorAgent",
    "OrchestratorAgent",
    "PlannerAgent",
    "ReflectorAgent",
    "RetrieverAgent",
    "ValidatorAgent",
]
