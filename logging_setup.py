import logging
from datetime import datetime

def setup_logger(log_file='train.log'):
    logging.basicConfig(
        level=logging.INFO,
        format='[%(asctime)s] %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S',
        filename=log_file,
        filemode='a'
    )
    # Also print to console
    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    formatter = logging.Formatter('[%(asctime)s] %(levelname)s - %(message)s', datefmt='%Y-%m-%d %H:%M:%S')
    console.setFormatter(formatter)
    logging.getLogger('').addHandler(console)
    logging.info("Program execution started...")