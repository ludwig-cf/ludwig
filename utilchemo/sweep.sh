#!/bin/bash

#set -x
#trap read debug

module purge
module load gcc mvapich2 python

# META PARAMETERS
BASEDIR=$PWD
SERVERUTILDIR=~/utilchemo/
UTILDIR=$BASEDIR/../util/

icosphere_small_l=0.1821
icosphere_large_l=0.2060

# SIMULATION PARAMETERS 
N_start=0
N_cycles=600000
size=80_80_80
grid=3_3_3
ntasks=27
memory=128000
time=1

# EXTRACTION PARAMETERS
freq=10000
freqconfig=100000
nstart=$freq
nend=$N_cycles
nint=$freq

# PHYSICAL PARAMETERS
conserve=2
mobility=1.0
A=0.00625

# INTERACTION PARAMETERS
phi_subgrid_on=0
u0=1e-5
cutoff=3.0
delta=1.0

# VESICLE PARAMETERS
RADIUS=12.0
PHIPROD=0.01
XSHIFT=40
YSHIFT=40
ZSHIFT=40
r0=$RADIUS
k0=1e-2
k2=1e-2
k3=1e-2

# SECOND VESICLE ?
TWOVESICLES=0
XSHIFT2=40
YSHIFT2=40
ZSHIFT2=40

# SWEEPING PARAMETER
# Choose parameter to sweep over:
sweepingParam="PHIPROD"
sweepingRange=(0.001 0.005 0.01 0.05 0.1)
prefix=$sweepingParam""

for param in ${sweepingRange[@]}; do
  cd $BASEDIR
  datafolder=$prefix"_"$param
  simname="LB_"$datafolder
  mkdir $datafolder -v 
  
  declare "${sweepingParam}"=$param
  echo "Beginning simulation with "${sweepingParam}"=$param"

  ### CREATE VESICLE INPUT FILE "latticeHexasphere.txt" ### 
  sed "s/XXXRADIUSXXX/$RADIUS/g" $"$SERVERUTILDIR"write_hexasphereXXX.py"" > write_hexasphere.py; sed -i "s/XXXXSHIFTXXX/$XSHIFT/g" write_hexasphere.py; sed -i "s/XXXYSHIFTXXX/$YSHIFT/g" write_hexasphere.py; sed -i "s/XXXZSHIFTXXX/$ZSHIFT/g" write_hexasphere.py; sed -i "s/XXXphi_productionXXX/$PHIPROD/g" write_hexasphere.py;
  cp $SERVERUTILDIR"hexasphere.xyz" .
  python3 write_hexasphere.py > r0s
  r2=$(sed '2q;d' r0s); r3=$(sed '1q;d' r0s);

  if [ $TWOVESICLES == 1 ]
  then
    sed "s/XXXRADIUSXXX/$RADIUS/g" $"$SERVERUTILDIR"write_hexasphere2XXX.py"" > write_hexasphere2.py; sed -i "s/XXXXSHIFTXXX/$XSHIFT2/g" write_hexasphere2.py; sed -i "s/XXXYSHIFTXXX/$YSHIFT2/g" write_hexasphere2.py; sed -i "s/XXXZSHIFTXXX/$ZSHIFT2/g" write_hexasphere2.py; sed -i "s/XXXphi_productionXXX/$PHIPROD/g" write_hexasphere2.py;
    python3 write_hexasphere2.py
    cat latticeHexasphere2.txt >> latticeHexasphere.txt
  fi


  ### CREATE  "config.cds.init001-001" with "multi_poly_init.c" ###
  sed "s/XXXu0XXX/$u0/g" $"$SERVERUTILDIR"multi_poly_initXXX.c"" > multi_poly_init.c; sed -i "s/XXXcutoffXXX/$cutoff/g" multi_poly_init.c; sed -i "s/XXXdeltaXXX/$delta/g" multi_poly_init.c; sed -i "s/XXXtwo_vesiclesXXX/$TWOVESICLES/g" multi_poly_init.c; 

  cp latticeHexasphere.txt $UTILDIR
  cp multi_poly_init.c $UTILDIR
  cd $UTILDIR

  make clean > /dev/null; make > /dev/null;
  chmod u+x multi_poly_init; ./multi_poly_init > /dev/null

  cp config.cds.init.001-001 $BASEDIR
  cd $BASEDIR


## CREATE INPUT FILE ###
  sed "s/XXXN_startXXX/$N_start/g" $"$SERVERUTILDIR"inputXXX"" > input; sed -i "s/XXXN_cyclesXXX/$N_cycles/g" input; sed -i "s/XXXsizeXXX/$size/g" input; sed -i "s/XXXgridXXX/$grid/g" input; sed -i "s/XXXmobilityXXX/$mobility/g" input; sed -i "s/XXXphi_subgrid_onXXX/$phi_subgrid_on/g" input; sed -i "s/XXXcahn_hilliard_options_conserveXXX/$conserve/g" input; sed -i "s/XXXbond_harmonic_r0XXX/$r3/g" input; sed -i "s/XXXbond_harmonic2_r0XXX/$r2/g" input; sed -i "s/XXXbond_harmonic3_r0XXX/$r0/g" input; sed -i "s/XXXbond_harmonic_kXXX/$k0/g" input; sed -i "s/XXXbond_harmonic2_kXXX/$k2/g" input; sed -i "s/XXXbond_harmonic3_kXXX/$k3/g" input; sed -i "s/XXXfreqXXX/$freq/g" input; sed -i "s/XXXfreqconfigXXX/$freqconfig/g" input; sed -i "s/XXXradius_vesicleXXX/$RADIUS/g" input; sed -i "s/XXXAXXX/$A/g" input;

## CREATE LAUNCH FILE ##
 
  sed "s/XXXntasksXXX/$ntasks/g" $SERVERUTILDIR"launchXXX" > $simname; sed -i "s/XXXmemoryXXX/$memory/g" $simname; sed -i "s/XXXtimeXXX/$time/g" $simname;
  
  
  cp $simname ../src/Ludwig.exe input config.cds.init.001-001 $datafolder
  cd $datafolder

  sbatch $simname 

done

cd $BASEDIR
rm -rf r0s hexasphere.xyz latticeHexasphere.txt LB_* latticeHexasphere2.txt write_hexasphere.py write_hexasphere2.txt input multi_poly_init.c config.cds.init.001-001 || true
