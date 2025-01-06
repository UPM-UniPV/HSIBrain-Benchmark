import os
import re

# local dir
root_dir = "/home/guillermo.vazquez/HSIBrain/mlruns"

# old dir
pattern = r"(artifact_(?:uri|location):\s*)file:///home/guille/Documents/HSIBrain_results"
# local machine dir
replacement = r"\1file:///home/guillermo.vazquez/HSIBrain"

file_extension = ".yaml"

def update_artifact_location(directory, pattern, replacement, extension):
    for dirpath, _, filenames in os.walk(directory):
        for filename in filenames:
            if filename.endswith(extension):
                file_path = os.path.join(dirpath, filename)
                with open(file_path, "r") as file:
                    content = file.read()
                
                # Cerca e sostituisci il pattern
                updated_content = re.sub(pattern, replacement, content)
                
                # Salva il file solo se c'è stato un cambiamento
                if updated_content != content:
                    with open(file_path, "w") as file:
                        file.write(updated_content)
                    print(f"Updated: {file_path}")
                else:
                    print(f"No changes in: {file_path}")

update_artifact_location(root_dir, pattern, replacement, file_extension)
