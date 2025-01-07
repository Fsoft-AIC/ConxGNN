import os
import shutil
import fnmatch


def copy_repo_files(dest_dir, current_dir):
    """Copies all files in the current repo to the destination directory,
    excluding those specified in .gitignore."""
    chosen_patterns = [
        "environment",
        "preprocessing",
        "scripts",
        "src",
        ".gitignore",
        "README.md",
        "train.py",
        "utils.py",
    ]

    def chosen_func(_, names):
        ignored = []
        for pattern in chosen_patterns:
            ignored.extend(fnmatch.filter(names, pattern))
        return ignored

    # Create destination directory if it doesn't exist
    os.makedirs(dest_dir, exist_ok=True)

    for item in os.listdir("."):
        if item in chosen_patterns:
            s = os.path.join(".", item)
            d = os.path.join(dest_dir, item)
            if os.path.isdir(s):
                shutil.copytree(s, d)
            else:
                shutil.copy2(s, d)
