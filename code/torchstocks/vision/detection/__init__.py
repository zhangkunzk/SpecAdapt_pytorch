#!/usr/bin/env python3

import sys
import os

PATH_STR = "/"
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ENGINE_PATH = BASE_DIR.split(PATH_STR)[0:-2]
sys.path.append(PATH_STR.join(ENGINE_PATH))