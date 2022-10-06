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
nstart=$freq
nend=$N_cycles
nint=$freq


# SYSTEM PARAMETERS
size=40_40_40
boundary_walls=0_0_0
periodicity=1_1_1

## PHYSICAL PARAMETERS

# FLUID
viscosity=0.5
fd_advection_scheme_order=1

# FREE ENERGY
symmetric_ll_a1=0.1
symmetric_ll_b1=0.0
symmetric_ll_kappa1=0.0

symmetric_ll_a2=0.1
symmetric_ll_b2=0.0
symmetric_ll_kappa2=0.0

symmetric_ll_mobility_phi=1.0
symmetric_ll_mobility_psi=1.0

phi0=0.5
psi0=0.5

cahn_hilliard_options_conserve=0


# VESICLE 
isfixedr=0
XSHIFT=20
YSHIFT=20
ZSHIFT=20

mx=1
my=0
mz=0

phi_subgrid_switch=1
u0=1e-6
delta=3.0
cutoff=6.0

vesicle_radius=10.0
mesh_harmonic_k=1e-2


# MASK / PERMEABILITY
mask_phi_switch=1
mask_psi_switch=1

mask_phi_permeability=0.1
mask_psi_permeability=0.1

phi_interaction_mask=0

mask_std_width=1.0
mask_std_alpha=0.8
mask_alpha_cutoff=1.0


# CHEMICAL REACTION
chemical_reaction_switch=0
chemical_reaction_model="uniform"

kappa=0.0001
kappa1=0.0001
kappam1=0.0001


# EXTERNAL FIELDS
grad_mu_phi=0.0_0.0_0.0
grad_mu_psi=0.0_0.0_0.0



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

  ### CREATE VESICLE INPUT FILE "latticeTrisphere.txt" ### 
  cd $UTIL_DIR

  sed "s/XXXvesicle_radiusXXX/$vesicle_radius/g" init/write_trisphereXXX.py > write_trisphere.py;
  sed -i "s/XXXXSHIFTXXX/$XSHIFT/g" write_trisphere.py;
  sed -i "s/XXXYSHIFTXXX/$YSHIFT/g" write_trisphere.py;
  sed -i "s/XXXZSHIFTXXX/$ZSHIFT/g" write_trisphere.py; 

  sed -i "s/XXXmxXXX/$mx/g" write_trisphere.py
  sed -i "s/XXXmyXXX/$my/g" write_trisphere.py
  sed -i "s/XXXmzXXX/$mz/g" write_trisphere.py

  cp init/rawfiles/trisphere.xyz .
  python3 write_trisphere.py > /dev/null


  ### CREATE  "config.cds.init001-001" with "trisphere_init.c" ###
  sed "s/XXXu0XXX/$u0/g" init/trisphere_initXXX.c > trisphere_init.c;
  sed -i "s/XXXcutoffXXX/$cutoff/g" trisphere_init.c; 
  sed -i "s/XXXdeltaXXX/$delta/g" trisphere_init.c; 
  sed -i "s/XXXisfixedrXXX/$isfixedr/g" trisphere_init.c;

  make clean > /dev/null; make > /dev/null;
  chmod u+x trisphere_init; ./trisphere_init > /dev/null

  cp config.cds.init.001-001 $SWEEP_DIR
  cd $SWEEP_DIR


## CREATE INPUT FILE ###
  sed "s/XXXN_startXXX/$N_start/g" $"$UTIL_DIR"/init/inputXXX"" > input;

  sed -i "s/XXXN_cyclesXXX/$N_cycles/g" input; 
  sed -i "s/XXXsizeXXX/$size/g" input;
  sed -i "s/XXXgridXXX/$grid/g" input; 

  sed -i "s/XXXboundary_wallsXXX/$boundary_walls/g" input;
  sed -i "s/XXXperiodicityXXX/$periodicity/g" input;
  sed -i "s/XXXviscosityXXX/$viscosity/g" input;

  sed -i "s/XXXfd_advection_scheme_orderXXX/$fd_advection_scheme_order/g" input;
  sed -i "s/XXXfreqXXX/$freq/g" input;

  sed -i "s/XXXsymmetric_ll_a1XXX/$symmetric_ll_a1/g" input;
  sed -i "s/XXXsymmetric_ll_b1XXX/$symmetric_ll_b1/g" input;
  sed -i "s/XXXsymmetric_ll_kappa1XXX/$symmetric_ll_kappa1/g" input;

  sed -i "s/XXXsymmetric_ll_a2XXX/$symmetric_ll_a2/g" input;
  sed -i "s/XXXsymmetric_ll_b2XXX/$symmetric_ll_b2/g" input;
  sed -i "s/XXXsymmetric_ll_kappa2XXX/$symmetric_ll_kappa2/g" input;

  sed -i "s/XXXsymmetric_ll_mobility_phiXXX/$symmetric_ll_mobility_phi/g" input;
  sed -i "s/XXXsymmetric_ll_mobility_psiXXX/$symmetric_ll_mobility_psi/g" input;

  sed -i "s/XXXphi0XXX/$phi0/g" input;
  sed -i "s/XXXpsi0XXX/$psi0/g" input;
  sed -i "s/XXXcahn_hilliard_options_conserveXXX/$cahn_hilliard_options_conserve/g" input;

  sed -i "s/XXXphi_subgrid_switchXXX/$phi_subgrid_switch/g" input; 
  sed -i "s/XXXvesicle_radiusXXX/$vesicle_radius/g" input; 
  sed -i "s/XXXmesh_harmonic_kXXX/$mesh_harmonic_k/g" input; 
 
  sed -i "s/XXXmask_phi_switchXXX/$mask_phi_switch/g" input; 
  sed -i "s/XXXmask_psi_switchXXX/$mask_psi_switch/g" input; 

  sed -i "s/XXXmask_phi_permeabilityXXX/$mask_phi_permeability/g" input; 
  sed -i "s/XXXmask_psi_permeabilityXXX/$mask_phi_permeability/g" input; 

  sed -i "s/XXXphi_interaction_maskXXX/$phi_interaction_mask/g" input; 

  sed -i "s/XXXmask_std_widthXXX/$mask_std_width/g" input; 
  sed -i "s/XXXmask_std_alphaXXX/$mask_std_alpha/g" input; 
  sed -i "s/XXXmask_alpha_cutoffXXX/$mask_alpha_cutoff/g" input; 

  sed -i "s/XXXchemical_reaction_switchXXX/$chemical_reaction_switch/g" input; 
  sed -i "s/XXXchemical_reaction_modelXXX/$chemical_reaction_model/g" input; 

  sed -i "s/XXXkappaXXX/$kappa/g" input; 
  sed -i "s/XXXkappa1XXX/$kappa1/g" input; 
  sed -i "s/XXXkappam1XXX/$kappam1/g" input; 

  sed -i "s/XXXgrad_mu_phiXXX/$grad_mu_phi/g" input; 
  sed -i "s/XXXgrad_mu_psiXXX/$grad_mu_psi/g" input; 


## CREATE LAUNCH FILE ##
 
  sed "s/XXXntasksXXX/$ntasks/g" $"$UTIL_DIR"/init/launchXXX"" > $simname; 
  sed -i "s/XXXmemoryXXX/$memory/g" $simname; 
  sed -i "s/XXXtimeXXX/$time/g" $simname;
  
  cp $simname input config.cds.init.001-001 $datafolder
  cp $LUDWIG_DIR"/src/Ludwig.exe" $datafolder

  cd $datafolder

  mpirun -n 8 ./Ludwig.exe
  #sbatch $simname 

done

cd $SWEEP_DIR
rm -rf Ludwig.exe trisphere.xyz latticeTrisphere.txt LB_* latticeTrisphere2.txt write_trisphere.py write_trisphere2.txt input trisphere_init.c config.cds.init.001-001 || true
