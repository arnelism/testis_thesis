from dotenv import load_dotenv


def load_env():
    load_dotenv("overrides.env")
    load_dotenv("defaults.env")
