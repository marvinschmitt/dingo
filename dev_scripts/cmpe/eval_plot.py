METHOD = "cmpe"

import sys
sys.path.append("../..")

import matplotlib.pyplot as plt

from dingo.gw.result import Result
result = Result(file_name=f"dev_scripts/{METHOD}/outdir_GW150914/result/GW150914.hdf5")
print(result.samples)
params=['luminosity_distance', 'geocent_time', 'a_1', 'a_2']

import corner
corner.corner(result.samples[params], labels=params, show_titles=True)
result.plot_corner(filename=f"dev_scripts/{METHOD}/outdir_GW150914/result/GW150914_corner.pdf", parameters=params)
