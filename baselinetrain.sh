#!/usr/bin/bash
. .venv-dave2/bin/activate
python throttle_model-from-h5-38.py -p /p/sdbb/SUT_base_dataset/ -o $SLURM_JOB_ID