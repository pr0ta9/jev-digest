"""jev-digest: purpose-driven passage digest for agents.

Tools: digest (search the web), browse (read known URLs) and read_documents (read local files).
search -> fetch -> passages -> Jev judges relevance, scope and topic per passage -> Jev marks passages that repeat others
-> original passages grouped by source. Every step is a decision over options the code enumerated; nothing is generated.
"""

from .pipeline import run_digest  # noqa: F401

__version__ = "0.1.0"
