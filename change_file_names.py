import os
import re

# Directory containing the files
directory = './Pics_embed'

# Regex to match the pattern: [name]_(trend|season|noise)_embedding_[flag]_[feat].pth
pattern = re.compile(r'^(?P<name>[a-zA-Z0-9]+)_(?P<type>trend|season|noise)_embedding_(?P<flag>[a-zA-Z0-9]+)_(?P<feat>[a-zA-Z0-9]+)\.pth$')

for filename in os.listdir(directory):
    match = pattern.match(filename)
    if match:
        old_path = os.path.join(directory, filename)

        # Extract parts from filename
        name = match.group('name')
        embed_type = match.group('type')
        flag = match.group('flag')
        feat = match.group('feat')

        # Build new filename with "CLIP_" added
        new_filename = f"{name}_CLIP_{embed_type}_embedding_{flag}_{feat}.pth"
        new_path = os.path.join(directory, new_filename)

        # Rename the file
        os.rename(old_path, new_path)
        print(f"Renamed: {filename} -> {new_filename}")
