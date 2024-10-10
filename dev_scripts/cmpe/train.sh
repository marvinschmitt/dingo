METHOD="cmpe"



cd ../..
rm -r dev_scripts/$METHOD/training
python -m dingo.gw.training.train_pipeline --settings dev_scripts/$METHOD/train_settings.yaml --train_dir dev_scripts/$METHOD/training