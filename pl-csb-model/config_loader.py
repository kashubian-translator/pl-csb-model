import configparser

def load() -> dict:
    config = configparser.ConfigParser()
    config.read("pl-csb-model/config.ini")
    return config
