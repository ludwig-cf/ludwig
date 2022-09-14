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
N_start=0
N_cycles=100
grid=2_2_2
ntasks=8
memory=128000
time=1


# EXTRACTION PARAMETERS
freq=10
freq_vel=100

nstart=$freq
nend=$N_cycles
nint=$freq

# SYSTEM PARAMETERS
size=60_60_60
boundary_walls=0_0_0
periodicity=1_1_1

## PHYSICAL PARAMETERS

# FLUID
viscosity=0.5
fd_advection_scheme_order=1

# FREE ENERGY
phi_A=-0.0128
phi_B=0.0128
phi_kappa0=0.04
phi_kappa1=-0.004
phi_kappa2=1e-4

psi_A=0.0
psi_B=0.0
psi_kappa=0.0

two_symm_oft_mobility_phi=0.5
two_symm_oft_mobility_psi=0.5

two_symm_oft_lambda=0.1


# VESICLE 

colloid_one_a0=8.0
colloid_one_ah=8.0

colloid_one_r=15.0_30.0_30.0
colloid_one_m=1.0_0.0_0.0

colloid_one_isfixedr=0
colloid_one_isfixeds=1

colloid_one_Tj1=0.0
colloid_one_Tj2=2.0


# SWEEPING PARAMETER
# Choose parameter to sweep over:
sweepingParam="viscosity"
sweepingRange=(0.5 1.0)
prefix=$sweepingParam""

for param in ${sweepingRange[@]}; do
  cd $SWEEP_DIR
  datafolder=$prefix"_"$param
  simname="LB_"$datafolder

  if [[ -e $datafolder ]];
  then
    while true; do
      read -p "Do you wish to overwrite "$datafolder" ?" yn
      case $yn in
          [Yy]* ) rm -r $datafolder; mkdir $datafolder -v; break;;
          [Nn]* ) exit;;
          * ) echo "Please answer yes or no.";;
      esac
    done
  else 
    mkdir $datafolder -v;
  fi

  declare "${sweepingParam}"=$param
  echo "Running simulation with "${sweepingParam}"=$param"

## CREATE INPUT FILE ###
  sed "s/XXXN_startXXX/$N_start/g" $"$UTIL_DIR"/init/inputXXX"" > input;

  sed -i "s/XXXN_cyclesXXX/$N_cycles/g" input; 
  sed -i "s/XXXsizeXXX/$size/g" input;
  sed -i "s/XXXgridXXX/$grid/g" input; 

  sed -i "s/XXXboundary_wallsXXX/$boundary_walls/g" input;
  sed -i "s/XXXperiodicityXXX/$periodicity/g" input;
  sed -i "s/XXXviscosityXXX/$viscosity/g" input;

  sed -i "s/XXXphi_AXXX/$phi_A/g" input;
  sed -i "s/XXXphi_BXXX/$phi_B/g" input;

  sed -i "s/XXXphi_kappa0XXX/$phi_kappa0/g" input;
  sed -i "s/XXXphi_kappa1XXX/$phi_kappa1/g" input;
  sed -i "s/XXXphi_kappa2XXX/$phi_kappa2/g" input;

  sed -i "s/XXXpsi_AXXX/$psi_A/g" input;
  sed -i "s/XXXpsi_BXXX/$psi_B/g" input;
  sed -i "s/XXXpsi_kappaXXX/$psi_kappa/g" input;

  sed -i "s/XXXtwo_symm_oft_mobility_phiXXX/$two_symm_oft_mobility_phi/g" input;
  sed -i "s/XXXtwo_symm_oft_mobility_psiXXX/$two_symm_oft_mobility_psi/g" input;
  sed -i "s/XXXtwo_symm_oft_lambdaXXX/$two_symm_oft_lambda/g" input;

  sed -i "s/XXXfd_advection_scheme_orderXXX/$fd_advection_scheme_order/g" input;

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
