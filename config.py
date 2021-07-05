# PUT YOUR INFORMATION HERE
CAPTAIN_EMAIL = ''
SALT = b''  # in bytes

netlocs = ['0.0.0.0:42049', '0.0.0.0:22066']
# gunicorn options
# bind, workers, threads, timeout, etc.
balancer_options = {
    'bind': '0.0.0.0:15391',
    'threads': 120,
    'timeout': 360,
}

# gunicorn options
# bind, workers, threads, timeout, etc.
options = {
    'threads': 30,
    'timeout': 360,
}
