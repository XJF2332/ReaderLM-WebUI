import json
import os

config_path = os.path.join("configs", "config.json")
config = {}
if not os.path.exists(config_path):
    with open(config_path, "w") as f:
        json.dump({}, f)
        f.close()
with open(config_path, "r") as f:
    config = json.load(f)
    f.close()


def save_config():
    with open(config_path, "w") as f:
        json.dump(config, f)
        f.close()
