'''One module per built-in coding-agent profile.'''

from llm_compat.agents.profiles.aider import PROFILE as AIDER
from llm_compat.agents.profiles.cline import PROFILE as CLINE
from llm_compat.agents.profiles.codex import PROFILE as CODEX
from llm_compat.agents.profiles.opencode import PROFILE as OPENCODE
from llm_compat.agents.profiles.roo_code import PROFILE as ROO_CODE

PROFILES = (AIDER, CLINE, CODEX, OPENCODE, ROO_CODE)

__all__ = ['PROFILES']