import os
import subprocess
import yaml

# Base configuration file
CONFIG_FILE = "configs/meld.yaml"
LOG_BASE = "meld_abla"
ABLATION_FOLDER = "ABLATION"
os.makedirs(ABLATION_FOLDER, exist_ok=True)

with open(CONFIG_FILE, "r") as file:
    config = yaml.safe_load(file)
    wp_wf_values = config.get("wp_wf", [])


# Function to run the train.py script with the given wp_wf values
def run_training(wp_wf, log_file):
    for i in range(1, 6):  # Run 5 times
        command = ["python", "train.py", CONFIG_FILE, "--wp_wf"] + wp_wf
        with open(log_file, "a") as log:
            subprocess.run(command, stdout=log)


# Split wp_wf values by 0 to form branches
branches = []
current_branch = []
for value in wp_wf_values:
    if value == 0:
        if current_branch:
            branches.append(current_branch)
            current_branch = []
    else:
        current_branch.append(str(value))
if current_branch:  # Add the last branch
    branches.append(current_branch)

# Define groups of wp_wf values to run
groups = [
    ["6", "4"],
    ["11", "11"],
    ["7", "4"],
    ["6", "4", "0", "11", "11"],
    ["6", "4", "0", "7", "4"],
    ["11", "11", "0", "7", "4"],
]

# Run each group
for group in groups:
    wp_wf_str = ",".join(group)
    log_file = os.path.join(ABLATION_FOLDER, f"{LOG_BASE}_wpwf_{wp_wf_str}.txt")
    run_training(group, log_file)

print("Training runs completed.")
