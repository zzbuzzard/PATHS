"""
Given [name]
 creates [ds]_[name] for ds = [KIRC, COADREAD, LUAD, KIRP, BRCA]
"""
import os
from os.path import join
import shutil
import json
from copy import deepcopy
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("-n", "--name", required=True, type=str)

args = parser.parse_args()

names = ["KIRP", "BRCA", "COADREAD", "LUAD", "KIRC"]

root = "../models"
found = None
todo = []
start_name = None
for name in names:
    start = join(root, name.lower() + "_" + args.name + "_0")
    if os.path.isdir(start):
        found = start
        start_name = name.lower()
    else:
        todo.append(name)

if found is None:
    print(args.name, "does not exist for any of the datasets:", names)
    quit(1)

confpath = join(found, "config.json")
if not os.path.isfile(confpath):
    print("Config not found at", confpath)
    quit(1)

with open(confpath, "r") as file:
    config = json.loads(file.read())

# config["root_name"] = args.name

wsi_dir = lambda s: config["wsi_dir"].replace(start_name, s.lower())
csv_path = lambda s: config["csv_path"].replace(start_name, s.lower())
omic_path = lambda s: config["omic_path"].replace(start_name, s.lower())
preprocess_dir = lambda s: config["preprocess_dir"].replace(start_name, s.lower())

for ds in todo:
    print("Creating ds", ds)
    ds = ds.lower()
    goal_conf = deepcopy(config)
    goal_conf["seed"] = 0
    goal_conf["wsi_dir"] = wsi_dir(ds)
    goal_conf["csv_path"] = csv_path(ds)
    if "omic_path" in goal_conf:
        goal_conf["omic_path"] = omic_path(ds)
    if "preprocess_dir" in goal_conf:
        goal_conf["preprocess_dir"] = preprocess_dir(ds)
    goal_conf["root_name"] = f"{ds}_" + args.name

    path = join(root, f"{ds}_" + args.name + "_0")
    cpath = join(path, "config.json")
    if not os.path.isdir(path) or not os.path.isfile(cpath):
        os.makedirs(path, exist_ok=True)
        with open(cpath, "w") as file:
            file.write(json.dumps(goal_conf, indent=4))
    else:
        with open(cpath, "r") as file:
            c = json.loads(file.read())
        if goal_conf == c:
            print(f"Leaving config at ds={ds} as it's all correct")
        else:
            current_missing = [i for i in goal_conf if i not in c]
            current_extra = [i for i in c if i not in goal_conf]
            current_diff = [i for i in c if c[i] != goal_conf[i]]
            print()
            print(f"Config already exists at ds={ds}")
            if len(current_missing) > 0:
                print("Current config is missing keys:", current_missing)
            if len(current_extra) > 0:
                print("Current config has extra keys:", current_extra)
            if len(current_diff) > 0:
                print("Keys clash:", current_diff)
                print(" Current keys:", [c[i] for i in current_diff])
                print(" Goal keys:", [goal_conf[i] for i in current_diff])
            choice = input("Overwrite? (y/n) ").upper()
            if choice == "Y":
                with open(cpath, "w") as file:
                    file.write(json.dumps(goal_conf, indent=4))
            else:
                print("Oke, skipping")
        print()
