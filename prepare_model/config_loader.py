import configparser

def load_default() -> dict:
    config = configparser.ConfigParser()
    config.read("config.ini")
    return dict(config["DEFAULT"])
