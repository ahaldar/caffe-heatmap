# tp_mkdir.m

import os

def tp_mkdir(dirName):
    if not os.path.exists(dirName):
        os.makedirs(dirName)
