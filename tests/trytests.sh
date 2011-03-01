##############################################################################
#
#  try.sh
#
#  This file runs the unit tests and the regression tests.
#  It is intended for my nightly test, so may not be very
#  portable.
#
#  $Id$
#
#  Edinburgh Soft Matter and Statistical Physics Group and
#  Edinburgh Parallel Computing Centre
#
#  Kevin Stratford (kevin@epcc.ed.ac.uk)
#  (c) 2010 The University of Edinburgh
#
##############################################################################
#!/bin/bash

# Test machine is currently martensite, where we need
export PATH=$PATH:/opt/openmpi/bin

# Serial unit tests

cd ../mpi_s
make clean
make libc
make testc

cd ../src
make clean
make lib
cd ../tests
make clean
make do_tests

# Serial regressions

cd ../src
make serial

echo Running autocorrelation test
./Ludwig.exe ../tests/regression/test_auto1_input > /tmp/junk
diff /tmp/junk ../tests/regression/test_auto1_srl_d3q19.ref

echo Running LE1 test
./Ludwig.exe ../tests/regression/test_le1_input > /tmp/junk
diff /tmp/junk ../tests/regression/test_le1_srl_d3q19.ref

echo Running LE2 test
./Ludwig.exe ../tests/regression/test_le2_input > /tmp/junk
diff /tmp/junk ../tests/regression/test_le2_srl_d3q19.ref1

echo Running LE3 test
./Ludwig.exe ../tests/regression/test_le3_input > /tmp/junk
diff /tmp/junk ../tests/regression/test_le3_srl_d3q19.ref

echo Running LE4 test
./Ludwig.exe ../tests/regression/test_le4_input > /tmp/junk
diff /tmp/junk ../tests/regression/test_le4_srl_d3q19.ref

echo Running LE5 test
./Ludwig.exe ../tests/regression/test_le5_input > /tmp/junk
diff /tmp/junk ../tests/regression/test_le5_srl_d3q19.ref

echo Running LE6 test
./Ludwig.exe ../tests/regression/test_le6_input > /tmp/junk
diff /tmp/junk ../tests/regression/test_le6_srl_d3q19.ref

echo Running spin03 serial test
./Ludwig.exe ../tests/regression/test_spin03_input > /tmp/junk
diff /tmp/junk ../tests/regression/test_spin03_d3q19_srl.ref

echo Running spin04 serial test
# CHECK CORRECT REFERENCE FILE
./Ludwig.exe ../tests/regression/test_spin04_input > /tmp/junk
diff /tmp/junk ../tests/regression/test_spin04_d3q19_srl.ref1

echo Running spin solid1 serial test
./Ludwig.exe ../tests/regression/test_spin_solid1_input > /tmp/junk
diff /tmp/junk ../tests/regression/test_spin_solid1_d3q19.ref1

echo Running Yukawa test
cp ../tests/regression/test_yukawa_cds.001-001 ./config.cds.init.001-001
./Ludwig.exe ../tests/regression/test_yukawa_input > /tmp/junk
diff /tmp/junk ../tests/regression/test_yukawa_d3q19.ref1

echo Running Cholesteric normal anchoring test
./Ludwig.exe ../tests/regression/test_chol_normal_input > /tmp/junk
diff /tmp/junk ../tests/regression/test_chol_normal_d3q19.ref1

echo Running Cholesteric planar anchoring test
./Ludwig.exe ../tests/regression/test_chol_planar_input > /tmp/junk
diff /tmp/junk ../tests/regression/test_chol_planar_d3q19.ref1


# Parallel unit tests

cd ../src
make clean
make libmpi
cd ../tests
make clean
make do_tests_mpi

# Parallel regressions

cd ../src
make mpi

echo Running BPI test
mpirun -np 8 ./Ludwig.exe ../tests/regression/test_bp1_input > /tmp/junk
diff /tmp/junk ../tests/regression/test_bp1_par.ref

echo Running BPII test
mpirun -np 8 ./Ludwig.exe ../tests/regression/test_bp2_input > /tmp/junk
diff /tmp/junk ../tests/regression/test_bp2_par.ref

echo Running spin03 parallel test
mpirun -np 8 ./Ludwig.exe ../tests/regression/test_spin03_input > /tmp/junk
diff /tmp/junk ../tests/regression/test_spin03_d3q19_par.ref

echo Running Yukawa parallel test
cp ../tests/regression/test_yukawa_cds.001-001 ./config.cds.init.001-001
mpirun -np 8 ./Ludwig.exe ../tests/regression/test_yukawa_input > /tmp/junk
diff /tmp/junk ../tests/regression/test_yukawa_d3q19.ref8

echo Running Cholesteric normal anchoring parallel test
mpirun -np 8 ./Ludwig.exe ../tests/regression/test_chol_normal_input > /tmp/j8
diff /tmp/j8 ../tests/regression/test_chol_normal_d3q19.ref8

echo Running Cholesteric planar anchoring parallel test
mpirun -np 8 ./Ludwig.exe ../tests/regression/test_chol_planar_input > /tmp/j8
diff /tmp/j8 ../tests/regression/test_chol_planar_d3q19.ref8

echo Running PARALLEL RESTART TEST PART ONE
mpirun -np 8 ./Ludwig.exe ../tests/regression/test_parallel_restart_input1 > /tmp/j9
diff /tmp/j9 ../tests/regression/test_parallel_restart1_d3q19.ref8

echo Running PARALLEL RESTART TEST PART TWO
mpirun -np 8 ./Ludwig.exe ../tests/regression/test_parallel_restart_input2 > /tmp/j9
diff /tmp/j9 ../tests/regression/test_parallel_restart2_d3q19.ref8

make clean

cd ../tests
make clean
