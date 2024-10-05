# from current paths list all the *.prompt file then readthem and return the {file}=content
from glob import glob
import os

cwd = os.path.dirname(os.path.realpath(__file__))


def get_prompt_files():
    from loguru import logger
    
    files = glob(cwd + "/*.prompt")

    out = {os.path.basename(file).split(".")[0]: open(file).read() for file in files}
    logger.info(f"Prompts: {out.keys()}")
    return out


__all__ = ["get_prompt_files"]

for k, v in get_prompt_files().items():
    setattr(__import__(__name__), k, v)
    globals()[k] = v
