import os
import pandas as pd

def find_global_max_residual(directory="OUTPUT/ANNUALS"):
    max_res = float("-inf")

    for fname in os.listdir(directory):
        if fname.endswith("_annual.csv"):
            fpath = os.path.join(directory, fname)
            try:
                df = pd.read_csv(fpath)
                if "max_residual" in df.columns:
                    local_max = df["max_residual"].max()
                    max_res = max(max_res, local_max)
            except Exception as e:
                print(f"Skipping {fname}: {e}")

    return max_res if max_res != float("-inf") else None

if __name__ == "__main__":
    global_max = find_global_max_residual()
    print("Global max_residual:", global_max)

