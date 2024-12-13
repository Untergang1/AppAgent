import logging


def get_logger(name, file_path=None, stream=False):
    logger = logging.getLogger(name)
    logger.setLevel(level=logging.DEBUG)

    if file_path:
        # path =
        file_handler = logging.FileHandler(file_path)
        file_handler.setLevel(level=logging.DEBUG)
        file_formatter = logging.Formatter(fmt='%(asctime)s - %(name)s - %(levelname)s - %(message)s', datefmt='%H:%M:%S')
        file_handler.setFormatter(file_formatter)
        logger.addHandler(file_handler)

    if stream:
        stream_handler = logging.StreamHandler()
        stream_handler.setLevel(level=logging.DEBUG)
        color_formatter = logging.Formatter(fmt='%(asctime)s - %(name)s - %(levelname)s - %(message)s', datefmt='%H:%M:%S')
        stream_handler.setFormatter(color_formatter)
        logger.addHandler(stream_handler)
    return logger

