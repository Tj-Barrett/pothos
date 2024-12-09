# pothos

pothos is a molecular dynamics tool for tracking and clustering polymer chain behavior. It evaluates order parameters such as $\langle P_2 \rangle$, and uses a modified DBSCAN to cluster aligned chain segments. Output data can be colored in a variety of ways in OVITO, such as cluster size, bead vector, average cluster vector, etc.

The current full version of pothos is pure python, however it is being rewritten into c++ to improve performance. For now, pothos++ offers both the legendre and verho alignment functions as a compiled function. pothos++ will be updated to eventually replace the python version.

# Installation Instructions


## Prerequesites
For the Python version : Python 3, Scipy, Numpy, Sci-kit Learn, Rich, Numba, Joblib

## Installation
The best way to use pothos is through a virtual enviroment
```
% set up the virtual environment
python -m venv venv
cd venv/bin
source activate
```

Download the source from github and enter it
```
% enter directory
cd /path/to/pothos
```

Install with pip
```
pip install .

or

pip install git+https://github.com/Tj-Barrett/pothos
```

pothos reads lammps dump files in the form
```
dump
```
