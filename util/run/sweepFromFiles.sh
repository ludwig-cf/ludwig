#!/bin/bash

#set -x
#trap read debug

#module purge
#module load gcc mvapich2 python

# META PARAMETERS
SWEEP_DIR=$PWD
LUDWIG_DIR="/home/jeremie/PHD/ludwig"
UTIL_DIR=$LUDWIG_DIR"/util"


# SIMULATION PARAMETERS 
N_start=100      # Has to be the same as the N_cycles of the relaxing simulation
N_cycles=100
grid=2_2_2
ntasks=8
memory=128000
time=1


# EXTRACTION PARAMETERS
freq=10
freq_vel=10

nstart=$freq
nend=$N_cycles
nint=$freq

# SYSTEM PARAMETERS
size=60_60_60
boundary_walls=0_0_0
periodicity=1_1_1

## PHYSICAL PARAMETERS

# FLUID
viscosity=0.625
fd_advection_scheme_order=1

# FREE ENERGY
phi_A0=-0.002
phi_A1=+0.0018
phi_A2=0.0

phi_B0=0.002
phi_B1=-0.0018
phi_B2=0.0

phi_Kappa0=0.04
phi_Kappa1=-0.036
phi_Kappa2=0.0

psi_A=0.0
psi_B=0.0
psi_Kappa=0.0
psi_beta=0.001

psi0=0.0001

two_symm_oft_mobility_phi=1.0
two_symm_oft_mobility_psi=1.0

two_symm_oft_lambda=0.1


# COLLOID

colloid_init=from_file

colloid_one_a0=8.0
colloid_one_ah=8.0

colloid_one_r=15.0_30.0_30.0
colloid_one_m=1.0_0.0_0.0

colloid_one_isfixedr=0
colloid_one_isfixeds=1

colloid_one_Tj1=0.0
colloid_one_Tj2=0.0


# SWEEPING PARAMETER
# Choose parameter to sweep over:
sweepingParam="colloid_one_Tj1"
sweepingRange=(0.5)
prefix=$sweepingParam""

for param in ${sweepingRange[@]}; do
  cd $SWEEP_DIR
  datafolder=$prefix"_"$param
  simname="LB_"$datafolder

  if [[ -e $datafolder ]];
  then
    echo ""$datafolder" found. Restarting..."
  else 
    echo ""$datafolder" not found"
    exit;
  fi

  declare "${sweepingParam}"=$param

## CREATE INPUT FILE ###
  sed "s/XXXN_startXXX/$N_start/g" $"$UTIL_DIR"/init/inputXXX"" > input;

  sed -i "s/XXXN_cyclesXXX/$N_cycles/g" input; 
  sed -i "s/XXXsizeXXX/$size/g" input;
  sed -i "s/XXXgridXXX/$grid/g" input; 

  sed -i "s/XXXboundary_wallsXXX/$boundary_walls/g" input;
  sed -i "s/XXXperiodicityXXX/$periodicity/g" input;
  sed -i "s/XXXviscosityXXX/$viscosity/g" input;

  sed -i "s/XXXphi_A0XXX/$phi_A0/g" input;
  sed -i "s/XXXphi_A1XXX/$phi_A1/g" input;
  sed -i "s/XXXphi_A2XXX/$phi_A2/g" input;

  sed -i "s/XXXphi_B0XXX/$phi_B0/g" input;
  sed -i "s/XXXphi_B1XXX/$phi_B1/g" input;
  sed -i "s/XXXphi_B2XXX/$phi_B2/g" input;

  sed -i "s/XXXphi_Kappa0XXX/$phi_Kappa0/g" input;
  sed -i "s/XXXphi_Kappa1XXX/$phi_Kappa1/g" input;
  sed -i "s/XXXphi_Kappa2XXX/$phi_Kappa2/g" input;

  sed -i "s/XXXpsi_AXXX/$psi_A/g" input;
  sed -i "s/XXXpsi_BXXX/$psi_B/g" input;
  sed -i "s/XXXpsi_KappaXXX/$psi_Kappa/g" input;
  sed -i "s/XXXpsi_betaXXX/$psi_beta/g" input;
  sed -i "s/XXXpsi0XXX/$psi0/g" input;

  sed -i "s/XXXtwo_symm_oft_mobility_phiXXX/$two_symm_oft_mobility_phi/g" input;
  sed -i "s/XXXtwo_symm_oft_mobility_psiXXX/$two_symm_oft_mobility_psi/g" input;
  sed -i "s/XXXtwo_symm_oft_lambdaXXX/$two_symm_oft_lambda/g" input;

  sed -i "s/XXXfd_advection_scheme_orderXXX/$fd_advection_scheme_order/g" input;

  sed -i "s/XXXcolloid_initXXX/$colloid_init/g" input;
  sed -i "s/XXXcolloid_one_Tj1XXX/$colloid_one_Tj1/g" input;
  sed -i "s/XXXcolloid_one_Tj2XXX/$colloid_one_Tj2/g" input;

  sed -i "s/XXXcolloid_one_a0XXX/$colloid_one_a0/g" input;
  sed -i "s/XXXcolloid_one_ahXXX/$colloid_one_ah/g" input;

  sed -i "s/XXXcolloid_one_rXXX/$colloid_one_r/g" input;
  sed -i "s/XXXcolloid_one_mXXX/$colloid_one_m/g" input;

  sed -i "s/XXXcolloid_one_isfixedrXXX/$colloid_one_isfixedr/g" input;
  sed -i "s/XXXcolloid_one_isfixedsXXX/$colloid_one_isfixeds/g" input;

  sed -i "s/XXXfreqXXX/$freq/g" input;
  sed -i "s/XXXfreq_velXXX/$freq_vel/g" input;


## CREATE LAUNCH FILE ##
 
  sed "s/XXXntasksXXX/$ntasks/g" $"$UTIL_DIR"/init/launchXXX"" > $simname; 
  sed -i "s/XXXmemoryXXX/$memory/g" $simname; 
  sed -i "s/XXXtimeXXX/$time/g" $simname;
  
  cp $simname input  $datafolder
  cp $LUDWIG_DIR"/src/Ludwig.exe" $datafolder

  cd $datafolder

  mpirun -n 8 ./Ludwig.exe
  #sbatch $simname 

done

cd $SWEEP_DIR
rm -rf Ludwig.exe LB_* input config.cds.init.001-001 || true
