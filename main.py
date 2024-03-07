from langevals.server import app as fastapi_app

from grevillea import Grevillea


handler = Grevillea(fastapi_app)


def main(request):
    return handler(request)
