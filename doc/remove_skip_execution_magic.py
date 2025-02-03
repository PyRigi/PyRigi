# this script removes the tag 'skip-execution'
# from all tutorial notebooks so these cells
# appear in the online documentation

import glob
import os

path = os.path.join(os.path.dirname(__file__), "userguide/tutorials/")

for filename in glob.iglob(path + "**/*.md", recursive=True):
    with open(filename, "r") as file:
        lines = file.readlines()
    with open(filename, "w") as file:
        for line in lines:
            if "%%skip_execution" not in line:
                file.write(line)
