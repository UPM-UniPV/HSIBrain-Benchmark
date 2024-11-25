import os
import re

# Percorso principale da cui partire per cercare i file
root_dir = "/home/domenico/mlruns"

# Pattern da cercare e nuova stringa
pattern = r"(artifact_(?:uri|location):\s*)file:///home/ragusa/ModelExperiments"
replacement = r"\1file:///home/domenico/"

# Estensione dei file da modificare
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

# Esegui lo script
update_artifact_location(root_dir, pattern, replacement, file_extension)
