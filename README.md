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
