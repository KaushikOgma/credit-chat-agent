# auth.py

async def socket_auth(token: str):
    # Simple token check
    VALID_TOKEN = "simple_token"
    
    if token == VALID_TOKEN:
        return True  # Authenticated
    else:
        return False  # Unauthorized
