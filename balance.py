import argparse
import concurrent.futures
import datetime
import gettext
import itertools
import time
import threading
import urllib.parse

import flask
import gunicorn.app.base
import requests

import config

parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('--netloc', nargs='+', help='netloc of api server node')
parser.add_argument('--max-workers', type=int)
args, gunicorn_argv = parser.parse_known_args()

app = flask.Flask(__name__)

netlocs = config.netlocs
if args.netloc is not None:
    netlocs = args.netloc
if args.max_workers is None:
    max_workers = len(netlocs) * 4
else:
    max_workers = args.max_workers
executor = concurrent.futures.ProcessPoolExecutor(len(netlocs) * 4)
urls = [urllib.parse.urlunparse(urllib.parse.ParseResult('http', netloc, 'inference', '', '', '')) for netloc in netlocs]
urls = itertools.cycle(urls)

cache_lock = threading.Lock()
cache = {}

session = requests.Session()


def request(url, data):
    response = session.post(url, json=data)
    return response.json()


@app.route('/inference', methods=['POST'])
def inference():
    """ API that return your model predictions when E.SUN calls this API. """
    server_timestamp = time.time()
    start = datetime.datetime.fromtimestamp(server_timestamp)
    print(flask.request.remote_addr)

    data: dict = flask.request.get_json(force=True)
    esun_uuid = data['esun_uuid']

    with cache_lock:
        try:
            future = cache[esun_uuid]
        except KeyError:
            future = cache[esun_uuid] = executor.submit(request, next(urls), data)

    response = future.result()

    retry = data['retry']

    if retry == 0:
        del cache[esun_uuid]

    print(flask.request.remote_addr, datetime.datetime.now() - start)
    return flask.jsonify(response)


class App(gunicorn.app.base.BaseApplication):
    def __init__(self, app, options=None, argv=None):
        self.options = options or {}
        self.argv = argv or []
        self.application = app
        super().__init__()

    def load_config(self):
        config = {}
        for key, value in self.options.items():
            if key in self.cfg.settings and value is not None:
                config[key] = value
        parser = self.cfg.parser()
        args, remaining_argv = parser.parse_known_args(self.argv)
        if remaining_argv:
            self.byebye(remaining_argv)
        args = vars(args)
        remaining_argv = []
        for key, value in args.items():
            if key in self.cfg.settings and value is not None:
                config[key] = value
        print('gunicorn config:')
        print(config)
        for key, value in config.items():
            self.cfg.set(key.lower(), value)

    def load(self):
        return self.application

    def byebye(self, argv):
        msg = gettext.gettext('unrecognized arguments: %s')
        message = msg % ' '.join(argv)
        args = {'prog': parser.prog, 'message': message}
        parser.exit(2, gettext.gettext('%(prog)s: error: %(message)s\n') % args)


options = config.balancer_options

print('main args:')
print(vars(args))
App(app, options, gunicorn_argv).run()
