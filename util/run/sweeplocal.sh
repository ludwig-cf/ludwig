#!/bin/bash

#set -x
#trap read debug

module purge
module load gcc mvapich2 python

# META PARAMETERS
SWEEP_DIR=$PWD
LUDWIG_DIR="/home/jeremie/PhD/ludwig"
UTIL_DIR=$LUDWIG_DIR"/util"


# SIMULATION PARAMETERS 
N_start=0
N_cycles=100000
grid=3_3_3
ntasks=27
memory=64000
time=1


# EXTRACTION PARAMETERS
freq=5000
freq_write=5000
nstart=$freq
nend=$N_cycles
nint=$freq


# SYSTEM PARAMETERS
size=60_60_60
boundary_walls=0_0_0
periodicity=1_1_1
colloid_init="from_file"

## PHYSICAL PARAMETERS

# FLUID
viscosity=0.5
fd_advection_scheme_order=1

# FREE ENERGY
symmetric_ll_a1=0.01
symmetric_ll_b1=0.0
symmetric_ll_kappa1=0.00000001

symmetric_ll_a2=0.01
symmetric_ll_b2=0.0
symmetric_ll_kappa2=0.00000001

symmetric_ll_mobility_phi=1.0
symmetric_ll_mobility_psi=1.0

phi0=0.0
psi0=0.0

cahn_hilliard_options_conserve=0


# VESICLE 
isfixedr=0
isfixedr_centre=0
XSHIFT=30
YSHIFT=30
ZSHIFT=30

mx=-1
my=0
mz=0

phi_subgrid_switch=1
u0=1e-4
delta=8
cutoff=4.0

vesicle_radius=8.0
mesh_harmonic_k=1e-2


# MASK / PERMEABILITY
mask_phi_switch=1
mask_psi_switch=0

mask_phi_permeability=0.0
mask_psi_permeability=0.0

phi_interaction_mask=1
phi_interaction_external_only=0

mask_std_width=1.0
mask_std_alpha=0.5
mask_alpha_cutoff=2.0


# CHEMICAL REACTION
chemical_reaction_switch=1
chemical_reaction_model="psi<->phi"

kappa=0.0
kappa1=1e-1
kappam1=1e-1


# EXTERNAL FIELDS
grad_mu_phi=0.0_0.0_0.0
grad_mu_psi=0.0_1e-4_0.0

add_tangential_force=0
tangential_force_magnitude=1e-3

# SWEEPING PARAMETER
# Choose parameter to sweep over:
sweepingParam="vesicle_template"
sweepingRange=("fullerene" "trisphere")

sweepingParam2="add_tangential_force"
sweepingRange2=(0 1)

prefix=$sweepingParam
prefix2=$sweepingParam2

for param in ${sweepingRange[@]}; do
  cd $SWEEP_DIR
  datafolder=$prefix"_"$param

  mkdir $datafolder 
  declare "${sweepingParam}"=$param
  echo "Running simulation with "${sweepingParam}"=$param"

  for param2 in ${sweepingRange2[@]}; do
    cd $SWEEP_DIR
    cd $datafolder
    datafolder2=$prefix2"_"$param2
    simname="LB_"$datafolder$datafolder2
    mkdir $datafolder2
    declare "${sweepingParam2}"=$param2;

    echo "Running simulation with "${sweepingParam2}"=$param2"
    
    writescriptXXX="write_"$vesicle_template"XXX.py"
    writescript="write_"$vesicle_template".py"
    configscriptXXX=$vesicle_template"_initXXX.c"
    configscript=$vesicle_template"_init.c"
    exec=$vesicle_template"_init"

    echo "Using "$vesicle_template "as template for vesicle"
    
    ### CREATE VESICLE INPUT FILE "latticeTrisphere.txt" ### 
    cd $UTIL_DIR
    sed "s/XXXvesicle_radiusXXX/$vesicle_radius/g" init/$writescriptXXX > $writescript;
    sed -i "s/XXXXSHIFTXXX/$XSHIFT/g" $writescript;
    sed -i "s/XXXYSHIFTXXX/$YSHIFT/g" $writescript;
    sed -i "s/XXXZSHIFTXXX/$ZSHIFT/g" $writescript; 

    sed -i "s/XXXmxXXX/$mx/g" $writescript
    sed -i "s/XXXmyXXX/$my/g" $writescript
    sed -i "s/XXXmzXXX/$mz/g" $writescript

    cp init/rawfiles/$vesicle_template.xyz .
    python3 $writescript > /dev/null
    

    ### CREATE  "config.cds.init001-001" with "trisphere_init.c" ###
    sed "s/XXXu0XXX/$u0/g" init/$configscriptXXX > $configscript;
    sed -i "s/XXXcutoffXXX/$cutoff/g" $configscript; 
    sed -i "s/XXXdeltaXXX/$delta/g" $configscript; 
    sed -i "s/XXXisfixedrXXX/$isfixedr/g" $configscript;
    sed -i "s/XXXisfixedr_centreXXX/$isfixedr_centre/g" $configscript;
  
    make clean > /dev/null; make > /dev/null;
    chmod u+x $exec; ./$exec > /dev/null

    cp config.cds.init.001-001 $SWEEP_DIR
    cd $SWEEP_DIR

    ## CREATE INPUT FILE ###
    sed "s/XXXN_startXXX/$N_start/g" $"$UTIL_DIR"/init/inputXXX"" > input;

    sed -i "s/XXXN_cyclesXXX/$N_cycles/g" input; 
    sed -i "s/XXXcolloid_initXXX/$colloid_init/g" input; 
    sed -i "s/XXXsizeXXX/$size/g" input;
    sed -i "s/XXXgridXXX/$grid/g" input; 

    sed -i "s/XXXboundary_wallsXXX/$boundary_walls/g" input;
    sed -i "s/XXXperiodicityXXX/$periodicity/g" input;
    sed -i "s/XXXviscosityXXX/$viscosity/g" input;

    sed -i "s/XXXfd_advection_scheme_orderXXX/$fd_advection_scheme_order/g" input;
    sed -i "s/XXXfreqXXX/$freq/g" input;
    sed -i "s/XXXfreq_writeXXX/$freq_write/g" input;

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
    sed -i "s/XXXphi_interaction_external_onlyXXX/$phi_interaction_external_only/g" input; 

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

    sed -i "s/XXXadd_tangential_forceXXX/$add_tangential_force/g" input; 
    sed -i "s/XXXtangential_force_magnitudeXXX/$tangential_force_magnitude/g" input; 


    ## CREATE LAUNCH FILE ##
 
    sed "s/XXXntasksXXX/$ntasks/g" $"$UTIL_DIR"/init/launchXXX"" > $simname; 
    sed -i "s/XXXmemoryXXX/$memory/g" $simname; 
    sed -i "s/XXXtimeXXX/$time/g" $simname;
  
    cp $simname input config.cds.init.001-001 $datafolder"/"$datafolder2
    cp $LUDWIG_DIR"/src/Ludwig.exe" $datafolder"/"$datafolder2

    cd $datafolder"/"$datafolder2

    #mpirun -n 8 ./Ludwig.exe
    #sbatch $simname 
  done
done

cd $SWEEP_DIR
rm -rf Ludwig.exe trisphere.xyz latticeTrisphere.txt LB_* latticeTrisphere2.txt write_trisphere.py write_trisphere2.txt input trisphere_init.c config.cds.init.001-001 || true
