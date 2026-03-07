
import os

IGNORE_FOLDERS = {"venv"}

def print_tree(start_path, prefix=""):
    files = [f for f in os.listdir(start_path) if f not in IGNORE_FOLDERS]

    for i, name in enumerate(files):
        path = os.path.join(start_path, name)

        connector = "└── " if i == len(files) - 1 else "├── "
        print(prefix + connector + name)

        if os.path.isdir(path):
            extension = "    " if i == len(files) - 1 else "│   "
            print_tree(path, prefix + extension)


print_tree(r"D:\REPO ANALYSIS USING AWS MODEL")