#!/bin/bash

printf "MPI installation script at %s\n" "$(date)"

my_prefix=$(pwd)/mpi

if [ -d mpi ]; then
    export PATH=${my_prefix}/bin:${PATH}
    printf "Added existing local mpi to path:\n"
    printf "PATH is %s\n" "${PATH}"
else
    printf "%s\n" "Install MPI to prefix %s\n" "${my_prefix}"
    wget https://download.open-mpi.org/release/open-mpi/v4.1/openmpi-4.1.4.tar.\
gz
    tar xf openmpi-4.1.4.tar.gz
    cd openmpi-4.1.4
    ./configure CC=gcc-11 CXX=g++-11    \
                --enable-mpi-fortran=no \
                --prefix=${my_prefix}
    make -j 2
    make install
    cd -
    rm -rf openmpi-4.1.4.tar.gz
    rm -rf openmpi-4.1.4
fi
