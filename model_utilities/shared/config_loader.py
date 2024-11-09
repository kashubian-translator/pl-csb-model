import configparser


def load() -> dict:
    config = configparser.ConfigParser(interpolation=configparser.ExtendedInterpolation())
    config.read("model_utilities/config.ini")
    return config
