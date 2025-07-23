# logger_context.py
"""
Context variable for tracking call IDs across async operations.
This module provides a thread-safe way to propagate call IDs through
the entire request lifecycle without explicitly passing them around.
"""

from contextvars import ContextVar
call_id_var: ContextVar[str] = ContextVar("call_id", default="N/A")
