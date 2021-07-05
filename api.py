import argparse
import base64
import concurrent.futures
import datetime
import gettext
import hashlib
import json
import os
import pathlib
import threading
import time

import cv2
import flask
import gunicorn.app.base
import numpy
import torch

import classifiers
import config

parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('--primary', nargs='+', required=True, help='primary checkpoints to ensemble')
parser.add_argument('--primary-threshold', default=0.28, type=float, help=' ')
parser.add_argument('--prelim', nargs='+', help='prelim checkpoints to ensemble')
parser.add_argument('--prelim-threshold', default=0.7, type=float, help=' ')
parser.add_argument('--max-workers', default=1, type=int, help='The maximum number of processes that can be used to execute the predict function calls')
parser.add_argument('--training-data-dic')
parser.add_argument('--data', default='data', help='directory to save requests and responses')
parser.add_argument('--captain-email')
parser.add_argument('--salt')
args, gunicorn_argv = parser.parse_known_args()

prelim = None

training_data_dic = None
if args.training_data_dic is not None:
    training_data_dic = pathlib.Path(args.training_data_dic).read_text().split('\n')
    training_data_dic = list(filter(lambda x: x != '', training_data_dic)) + ['isnull']

pcs = []
if args.prelim is not None:
    for pckpt in args.prelim:
        state_dict = torch.load(pckpt, map_location='cpu')
        classifier = getattr(classifiers, state_dict['name'])()
        classifier.load_state_dict(state_dict)
        pcs.append(classifier)

if len(pcs) > 0:
    prelim = classifiers.EnsembleClassifier(pcs, method=None, threshold=args.prelim_threshold)

cs = []
for ckpt in args.primary:
    state_dict = torch.load(ckpt, map_location='cpu')
    classifier = getattr(classifiers, state_dict['name'])()
    classifier.load_state_dict(state_dict)
    cs.append(classifier)

classifier = classifiers.EnsembleClassifier(cs, method=None, threshold=args.primary_threshold, prelim=prelim, training_data_dic=training_data_dic)

executor = concurrent.futures.ProcessPoolExecutor(args.max_workers)
lock = threading.Lock()
cache = {}

app = flask.Flask(__name__)

CAPTAIN_EMAIL = config.CAPTAIN_EMAIL
SALT = config.SALT
if args.captain_email is not None:
    CAPTAIN_EMAIL = args.captain_email
if args.salt is not None:
    SALT = args.salt.encode()

data_directory = pathlib.Path(args.data)
request_directory = data_directory / 'request'
response_directory = data_directory / 'response'

request_directory.mkdir(parents=True, exist_ok=True)
response_directory.mkdir(parents=True, exist_ok=True)


def generate_server_uuid(input_string: str) -> str:
    """ Create your own server_uuid.

    @param:
        input_string (str): information to be encoded as server_uuid
    @returns:
        server_uuid (str): your unique server_uuid
    """
    s = hashlib.sha256()
    data = input_string.encode("utf-8") + SALT
    s.update(data)
    server_uuid = s.hexdigest()
    return server_uuid


def base64_to_binary_for_cv2(image_64_encoded: str) -> numpy.ndarray:
    """ Convert base64 to numpy.ndarray for cv2.

    @param:
        image_64_encode(str): image that encoded in base64 string format.
    @returns:
        image(numpy.ndarray): an image.
    """
    img_base64_binary = image_64_encoded.encode("utf-8")
    img_binary = base64.b64decode(img_base64_binary)
    image: numpy.ndarray = cv2.imdecode(numpy.frombuffer(img_binary, numpy.uint8), cv2.IMREAD_COLOR)
    return image


def _check_datatype_to_string(prediction):
    """ Check if your prediction is in str type or not.
        If not, then raise error.

    @param:
        prediction: your prediction
    @returns:
        True or raise TypeError.
    """
    if isinstance(prediction, str):
        return True
    raise TypeError('Prediction is not in string type.')


def predict(image_64_encoded: str) -> str:
    """ Predict your model result.

    @param:
        image (numpy.ndarray): an image.
    @returns:
        prediction (str): a word.
    """

    image = base64_to_binary_for_cv2(image_64_encoded)

    # PUT YOUR MODEL INFERENCING CODE HERE
    prediction = '陳'

    prediction = classifier.classify(image)
    if training_data_dic is not None and prediction not in training_data_dic:
        prediction = 'isnull'
    print(os.getpid(), prediction)

    if _check_datatype_to_string(prediction):
        return prediction


@app.route('/inference', methods=['POST'])
def inference():
    """ API that return your model predictions when E.SUN calls this API. """
    server_timestamp = time.time()
    start = datetime.datetime.fromtimestamp(server_timestamp)
    print(flask.request.remote_addr)

    data: dict = flask.request.get_json(force=True)
    esun_uuid = data['esun_uuid']

    with lock:
        try:
            future = cache[esun_uuid]
        except KeyError:
            future = cache[esun_uuid] = executor.submit(predict, data['image'])

    answer = future.result()

    t = datetime.datetime.now()
    ts = str(int(t.utcnow().timestamp()))
    server_uuid = generate_server_uuid(CAPTAIN_EMAIL + ts)

    esun_timestamp = data['esun_timestamp']
    retry = data['retry']

    stem = '{}-{}'.format(esun_timestamp, server_timestamp)

    request_directory.mkdir(parents=True, exist_ok=True)
    response_directory.mkdir(parents=True, exist_ok=True)

    with open(request_directory / f'{stem}.json', 'w') as f:
        json.dump(data, f, ensure_ascii=False)

    server_timestamp = time.time()
    response = {'esun_uuid': data['esun_uuid'], 'server_uuid': server_uuid, 'answer': answer, 'server_timestamp': server_timestamp}
    with open(response_directory / f'{stem}.json', 'w') as f:
        json.dump(response, f, ensure_ascii=False)

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


options = config.options

print('main args:')
print(vars(args))
App(app, options, gunicorn_argv).run()
