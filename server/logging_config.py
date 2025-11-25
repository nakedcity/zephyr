import logging
import sys
from contextvars import ContextVar
from typing import Optional

# Context variable to store trace_id per request
trace_id_var: ContextVar[Optional[str]] = ContextVar('trace_id', default=None)


class TraceIDFormatter(logging.Formatter):
    """Custom formatter that includes trace_id in log messages."""
    
    def format(self, record):
        trace_id = trace_id_var.get()
        if trace_id:
            record.trace_id = f"[trace_id={trace_id}]"
        else:
            record.trace_id = ""
        return super().format(record)


def setup_logging():
    """Configure logging with trace ID support."""
    # Create formatter
    formatter = TraceIDFormatter(
        fmt='%(asctime)s [%(levelname)s] %(trace_id)s %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    # Configure root logger
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(formatter)
    
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    logger.addHandler(handler)
    
    return logger


def set_trace_id(trace_id: str):
    """Set the trace_id for the current context."""
    trace_id_var.set(trace_id)


def get_trace_id() -> Optional[str]:
    """Get the trace_id for the current context."""
    return trace_id_var.get()
