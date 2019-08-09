# amr-labels

## Setup & running

When running on the server, first run

``. server_setup.sh``

This should put you into a conda environment. Find a free GPU-ID X (X=0,1,2,...) with `nvidia-smi` and run

``python main.py --gpu X``
