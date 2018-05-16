# Overview

This project evaluates the effectiveness of modeling optimal honeypot allocation as a Bayesian Stackleberg Game.  The main code for executing the experiments described in an upcoming paper are described in `experiment_runner.py`.

# Installation

```
pip install pyomo

#Download bonmin
cd $bonmin_dir
cd ThirdParty
cd ASL
./get.ASL
cd ../Blas
./get.Blas
cd ../Lapack
./get.Lapack
cd ../Mumps
./get.Mumps
cd ../..
./configure
make
make install

# update path to point to $bonmin_dir/bin/ so bonmin is on the path
```
