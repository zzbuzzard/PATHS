"""
python mk_folds.py [name]
"""
import os
from os.path import join
import shutil
import json
from copy import deepcopy
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("-n", "--name", required=True, type=str)
parser.add_argument("-f", "--folds", type=int, default=5)

args = parser.parse_args()

root = "../models"
paths = [join(root, args.name + f"_{i}") for i in range(10)]
start = None

for i in paths:
    if os.path.isdir(i):
        start = i
        # print("Basing config off", i)
        break

if start is None:
    print("No starting config found for", args.name)
    quit(1)

confpath = join(start, "config.json")
if not os.path.isfile(confpath):
    print("Config not found at", confpath)
    quit(1)

with open(confpath, "r") as file:
    config = json.loads(file.read())

# if args.folds == -1:
#     if "task" in config and config["task"] == "subtype_classification":
#         args.folds = 10
#     else:
#         args.folds = 5
print("Creating", args.folds, "folds")

if not "root_name" in config:
    print("Adding root_name to config")
    config["root_name"] = args.name
else:
    r = config["root_name"]
    assert r == args.name, f"Root name invalid: {r} != {args.name}"

for i in range(args.folds):
    # print("Creating fold", i)
    goal_conf = deepcopy(config)
    goal_conf["seed"] = i
    cpath = join(paths[i], "config.json")
    if not os.path.isdir(paths[i]) or not os.path.isfile(cpath):
        os.makedirs(paths[i], exist_ok=True)
        with open(cpath, "w") as file:
            file.write(json.dumps(goal_conf, indent=4))
    else:
        with open(cpath, "r") as file:
            c = json.loads(file.read())
        if goal_conf == c:
            pass
            # print(f"Leaving config at i={i} as it's all correct")
        else:
            current_missing = [i for i in goal_conf if i not in c]
            current_extra = [i for i in c if i not in goal_conf]
            current_diff = [i for i in c if c[i] != goal_conf[i]]
            print()
            print(f"Config already exists at i={i}")
            if len(current_missing) > 0:
                print("Current config is missing keys:", current_missing)
            if len(current_extra) > 0:
                print("Current config has extra keys:", current_extra)
            if len(current_diff) > 0:
                print("Keys clash:", current_diff)
                print(" Current keys:", [c[i] for i in current_diff])
                print(" Goal keys:", [goal_conf[i] for i in current_diff])
            print("Overwriting...")
            with open(cpath, "w") as file:
                file.write(json.dumps(goal_conf, indent=4))
