import pandas as pd
import numpy as np
import os

# Ensure the output directory exists
output_dir = "npz_files"
os.makedirs(output_dir, exist_ok=True)

# Loop over all the specified file names
for xx in range(1, 31):  # Loop from 01 to 30
    for Y in range(1, 10):  # Loop from 1 to 9
        file_name = f".\\preprocessed\\{xx:02d}_sensor{Y}_labelled.csv"
        try:
            # Try to read the file
            df = pd.read_csv(file_name)

            # Filter rows based on the last column and save them
            for condition, suffix in ((1, "F"), (0, "NF")):
                # condition_rows = df[df.iloc[:, -1] == condition]
                condition_rows = df[df.iloc[:, -1] == condition].iloc[:, :-1]
                if not condition_rows.empty:
                    output_file_path = os.path.join(output_dir, f"{xx}_{Y}_1_{suffix}.npz")
                    np.savez(output_file_path, condition_rows.values)

        except FileNotFoundError:
            print(f"File not found: {file_name}")
        except Exception as e:
            print(f"An error occurred while processing {file_name}: {e}")

print("Processing complete.")






