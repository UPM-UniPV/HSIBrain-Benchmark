import os
import re

# local dir
current_dir = os.getcwd()

# assuming this script is in the same folder of mlruns dir
root_dir = current_dir + "/mlruns"

# old dir
# something like /home/user/HSIBrain
pattern = r"(artifact_(?:uri|location):\s*)file://{re.escape(current_dir)}"
# local machine dir
replacement = r"\1file://{}".format(os.path.expanduser("~"))

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
