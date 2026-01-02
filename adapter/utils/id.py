import uuid


def gen_id(prefix: str):
    return f"{prefix}-{uuid.uuid4()}"
