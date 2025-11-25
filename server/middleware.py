import uuid
from fastapi import Request
from starlette.middleware.base import BaseHTTPMiddleware
from server.logging_config import set_trace_id


class TraceIDMiddleware(BaseHTTPMiddleware):
    """Middleware to generate or extract trace_id for each request."""
    
    async def dispatch(self, request: Request, call_next):
        # Try to get trace_id from header, otherwise generate new one
        trace_id = request.headers.get('X-Trace-ID')
        if not trace_id:
            trace_id = str(uuid.uuid4())[:8]  # Short trace ID (first 8 chars)
        
        # Set trace_id in context
        set_trace_id(trace_id)
        
        # Add trace_id to response headers for client tracking
        response = await call_next(request)
        response.headers['X-Trace-ID'] = trace_id
        
        return response
