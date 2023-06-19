import os
from functools import wraps

from dotenv import load_dotenv
from fastapi.security.api_key import APIKeyHeader
from fastapi import Security, HTTPException
from httpx import Request
from starlette.status import HTTP_403_FORBIDDEN, HTTP_401_UNAUTHORIZED

load_dotenv()
# API KEY
api_key_header = APIKeyHeader(name="access_token", auto_error=False)


class Handler:
    def __init__(self, successor=None):
        self.successor = successor

    def handle_request(self, request):
        pass


class APIKeyHandler(Handler):
    def handle_request(self, request):
        api_key = request.headers.get(api_key_header.name)
        if api_key != os.getenv("API_KEY"):
            raise HTTPException(
                status_code=HTTP_403_FORBIDDEN, detail="Could not validate API KEY"
            )

        if self.successor:
            self.successor.handle_request(request)


# class ChainOfResponsibilityMiddleware:
#     def __init__(self, app):
#         self.app = app
#         self.handler = APIKeyHandler()
#
#     async def __call__(self, request, response):
#         self.handler.handle_request(request)
#         return await self.app(request, response)


def auth_observer(auth_key):
    def decorator(func):
        @wraps(func)
        async def wrapper(*args, **kwargs):
            request = next(arg for arg in args if isinstance(arg, Request))
            headers = request.headers
            if "authorization" in headers and headers["authorization"] == auth_key:
                return await func(*args, **kwargs)
            else:
                raise HTTPException(
                    status_code=HTTP_401_UNAUTHORIZED,
                    detail="Unauthorized",
                    headers={"WWW-Authenticate": "Bearer"},
                )

        return wrapper

    return decorator


async def get_api_key(api_key_header: str = Security(api_key_header)):
    """
    Validates API KEY
    """
    if api_key_header == os.getenv("API_KEY"):
        return api_key_header
    else:
        raise HTTPException(
            status_code=HTTP_403_FORBIDDEN, detail="Could not validate API KEY"
        )
