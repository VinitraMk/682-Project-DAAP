import os
import subprocess
import config as C

for split in ["train", "val"]:
    for dirname in os.listdir(os.path.join(C.DATA_DIR, split)):
        cur_dir = os.path.join(C.DATA_DIR, split, dirname)
        if dirname not in C.CLASS_CODE2NAME:
            continue    
        new_dir = os.path.join(C.DATA_DIR, split, C.CLASS_CODE2NAME[dirname])
        subprocess.call("mv {} {}".format(cur_dir, new_dir), shell=True)

