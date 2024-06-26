import os

from .deps import MASE_HW_DEPS


def get_modules():
    current_dir = os.path.dirname(os.path.abspath(__file__))
    mods = [
        d
        for d in os.listdir(current_dir)
        if os.path.isdir(os.path.join(current_dir, d))
    ]
    if "__pycache__" in mods:
        mods.remove("__pycache__")
    return mods
