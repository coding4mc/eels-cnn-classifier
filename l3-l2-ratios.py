from lib.maths import calculate_l_peak_area_ratios
import hyperspy.api as hs
import numpy as np
import os

# file_name = "fib1a-2025-04-13.pkl"
# file_name = "250410_VCP_E_ni.hspy"
file_name = "fib1b-2025-04-14.hspy"
base_name, _ = os.path.splitext(file_name)

s = hs.load(f"Processed Test Data/{file_name}")

ratio_matrix = calculate_l_peak_area_ratios(
    spectrum=s,
    l2_signal_range=[870.0, 879.0],
    l2_peak_range=7,
    l3_signal_range=[850.0, 860.0],
    l3_peak_range=6
)
print(ratio_matrix.shape)

# np.save(file=f"L3-L2 Ratios/{base_name}.npy", arr=ratio_matrix)

