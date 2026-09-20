"""Tools layer. Importing the package registers every tool.

The loop only touches registry.execute / registry.schemas /
registry.approval_for / registry.ToolContext.
"""

from . import files, registry, shell, web

