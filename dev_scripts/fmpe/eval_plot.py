import sys
sys.path.append("../..")

import matplotlib.pyplot as plt

from dingo.gw.result import Result
result = Result(file_name="dev_scripts/fmpe/outdir_GW150914/result/GW150914.hdf5")
result.plot_corner(filename="dev_scripts/fmpe/outdir_GW150914/result/GW150914_corner.pdf")
