import logging
import os


if "train.log" in os.listdir("."):
    os.remove("train.log")

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

# formatter = logging.Formatter('%(levelname)s:%(name)s:%(message)s')
formatter = logging.Formatter("%(message)s")

file_handler = logging.FileHandler("train.log")
file_handler.setFormatter(formatter)

logger.addHandler(file_handler)
