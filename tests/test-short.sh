#!/bin/bash

# Paralel unit tests

cd trunk/src
make clean
make libmpi
make mpi

cd ../tests
make clean
make tests_mpi

mpirun -np 8 ./test_collision
mpirun -np 8 ./test_colloids
mpirun -np 8 ./test_colloids_halo
mpirun -np 8 ./test_colloid_sums
mpirun -np 8 ./test_coords
mpirun -np 8 ./test_ewald
mpirun -np 8 ./test_halo
mpirun -np 8 ./test_io
mpirun -np 8 ./test_le
mpirun -np 8 ./test_model
mpirun -np 8 ./test_pe
mpirun -np 8 ./test_phi
mpirun -np 8 ./test_phi_ch
mpirun -np 8 ./test_polar_active
mpirun -np 8 ./test_prop
mpirun -np 8 ./test_psi
mpirun -np 8 ./test_psi_sor
mpirun -np 8 ./test_random
mpirun -np 8 ./test_runtime
mpirun -np 8 ./test_site_map
mpirun -np 8 ./test_timer

cd ../src

echo Running BPI test
mpirun -np 8 ./Ludwig.exe ../tests/regression/test_bp1_input > tmp.log
diff tmp.log ../tests/regression/test_bp1_d3q19.ref8

echo Running BPII test
mpirun -np 8 ./Ludwig.exe ../tests/regression/test_bp2_input > tmp.log
diff tmp.log ../tests/regression/test_bp2_d3q19.ref8

echo Running spin03 parallel test
mpirun -np 8 ./Ludwig.exe ../tests/regression/test_spin03_input > tmp.log
diff tmp.log ../tests/regression/test_spin03_d3q19_par.ref

echo Running Yukawa parallel test
cp ../tests/regression/test_yukawa_cds.001-001 ./config.cds.init.001-001
mpirun -np 8 ./Ludwig.exe ../tests/regression/test_yukawa_input > tmp.log
diff tmp.log ../tests/regression/test_yukawa_d3q19.ref8

echo Running Cholesteric normal anchoring parallel test
mpirun -np 8 ./Ludwig.exe ../tests/regression/test_chol_normal_input > tmp.log
diff tmp.log ../tests/regression/test_chol_normal_d3q19.ref8

echo Running Cholesteric planar anchoring parallel test
mpirun -np 8 ./Ludwig.exe ../tests/regression/test_chol_planar_input > /tmp/j8
diff /tmp/j8 ../tests/regression/test_chol_planar_d3q19.ref8

echo Running PARALLEL RESTART TEST PART ONE
mpirun -np 8 ./Ludwig.exe ../tests/regression/test_parallel_restart_input1 > tmp.log
diff tmp.log ../tests/regression/test_parallel_restart1_d3q19.ref8

echo Running PARALLEL RESTART TEST PART TWO
mpirun -np 8 ./Ludwig.exe ../tests/regression/test_parallel_restart_input2 > tmp.log
diff tmp.log ../tests/regression/test_parallel_restart2_d3q19.ref8

echo Running Explicit surface free energy anchoring test normal
mpirun -np 8 ./Ludwig.exe ../tests/regression/test_chol_normal_input > tmp.log
diff tmp.log ../tests/regression/test_chol_normal_d3q19.ref8

echo Running Explicit surface free energy anchoring test planar
mpirun -np 8 ./Ludwig.exe ../tests/regression/test_chol_planar_input > tmp.log
diff tmp.log ../tests/regression/test_chol_planar_d3q19.ref8

rm -rf tmp.log

cd ../src
make clean

