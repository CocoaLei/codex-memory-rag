from .schemas import ContextRequest


def read_context(retriever, request: ContextRequest) -> dict:
    return retriever.read_context(request)
