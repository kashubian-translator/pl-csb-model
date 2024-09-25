import configparser

def load() -> dict:
    config = configparser.ConfigParser(interpolation=configparser.ExtendedInterpolation())
    config.read("pl-csb-model/config.ini")
    return config
