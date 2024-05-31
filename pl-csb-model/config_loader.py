import configparser

def load() -> dict:
    config = configparser.ConfigParser()
    config.read("config.ini")
    return config
