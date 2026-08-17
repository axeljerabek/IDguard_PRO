from functools import wraps
from flask import request, Response

def check_auth(username, password):
    return username == 'admin' and password == 'changethispasswd12'

def authenticate():
    return Response(
        'Zugriff verweigert. Bitte Zugangsdaten eingeben.\n', 401,
        {'WWW-Authenticate': 'Basic realm="IDguard PRO Login"'}
    )

def requires_auth(f):
    @wraps(f)
    def decorated(*args, **kwargs):
        auth = request.authorization
        if not auth or not check_auth(auth.username, auth.password):
            return authenticate()
        return f(*args, **kwargs)
    return decorated
