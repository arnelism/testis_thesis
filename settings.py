import os

from dotenv import load_dotenv


def load_env():
    overrides = os.environ.get("overrides")
    if overrides is not None:
        if load_dotenv(overrides):
            print(f"Overriding settings with values from {overrides}")

    load_dotenv("defaults.env")

    settingsfile = os.environ.get("settingsfile")
    if settingsfile is not None:
        if load_dotenv(settingsfile):
            print(f"Loading more settings from {settingsfile}")
