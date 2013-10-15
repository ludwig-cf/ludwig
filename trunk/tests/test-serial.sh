#!/bin/bash

# Serial unit/regression tests
# We start in ../..

cd trunk/mpi_s
make libc
make testc

cd ../src
make lib

cd ../tests
make do_tests
make clean

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

# Clean up all directories

make clean

cd ../mpi_s
make clean
