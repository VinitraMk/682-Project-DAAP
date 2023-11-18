import json

DATA_DIR = "./data/imagenette2"
ATTACK_DIR = "./data/attacked"
ATTACK_CUSTOM_DIR = "./data/attacked_custom"
ATTACK_PLOT_DIR = "./data/plots"
DEFENSE_DIRS = "./data/defense/{}"
UNET_CHECKPOINT = "./data/checkpoints/UNet"

with open("./data/class_code2name.json") as f:
    CLASS_CODE2NAME = json.load(f)
with open("./data/class_id2name.json") as f:
    CLASS_ID2NAME = json.load(f)
for id in list(CLASS_ID2NAME.keys()):
    CLASS_ID2NAME[int(id)] = CLASS_ID2NAME[id]
CLASS_NAME2ID = {v:k for k, v in CLASS_ID2NAME.items()}