METHOD="cmpe"

cd ../..
python -m dingo.gw.pipe.sampling dev_scripts/$METHOD/GW150914.ini
python dev_scripts/$METHOD/eval_plot.py