"""Tiny routing table web app (framework-free)."""

ROUTES = {}


def route(path):
    def deco(fn):
        ROUTES[path] = fn
        return fn
    return deco


@route("/")
def index():
    return {"status": 200, "body": "welcome"}


@route("/version")
def version():
    return {"status": 200, "body": "1.0.0"}


def handle(path):
    fn = ROUTES.get(path)
    if fn is None:
        return {"status": 404, "body": "not found"}
    return fn()
