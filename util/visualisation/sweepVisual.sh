#!/bin/bash

#set -x
#trap read debug

# META PARAMETERS
SWEEP_DIR=$PWD"/"
UTIL_DIR=~/PHD/ludwig/util/

# SIMULATION PARAMETERS 
N_start=0
N_cycles=100000

# EXTRACTION PARAMETERS
freq=1000
nstart=$freq
nend=$N_cycles
nint=$freq
nfiles=$(($N_cycles / $freq))

# VIDEO PARAMETERS
spatial_res=300
length_slice=30

# SWEEPING PARAMETER
# Must match with datafolders
sweepingParam="mask_phi_permeability"
sweepingRange=(0.0 1e-5 1e-4 1e-3 1e-2 1e-1)

prefix=$sweepingParam

printf "\n Checking total file numbers... \n \n"
for param in ${sweepingRange[@]}; do
  datafolder=$prefix"_"$param
  cd $datafolder
  
  if [[ "$(ls phi*.vtk | wc -l)" -ne "$nfiles" || "$(ls col* | wc -l)" -ne "$nfiles" || "$(ls dircol* | wc -l)" -ne "$nfiles" || "$(ls mask*.vtk | wc -l)" -ne "$nfiles" ]]; then
    echo "Number of files not matching in "$datafolder
  fi
  else
    echo "Number of files is matching in "$datafolder
  cd $SWEEP_DIR
done

cd $SWEEP_DIR

while true; do
    read -p "Do you wish to continue ? " yn
    case $yn in
        [Yy]* ) break;;
        [Nn]* ) exit;;
        * ) echo "Please answer yes or no.";;
    esac
done

printf "\n Proceeding... \n \n"

for param in ${sweepingRange[@]}; do
  datafolder=$prefix"_"$param
   
  cp $UTIL_DIR"/visual/save_video.py" $datafolder

  cd $datafolder
  printf "Creating video in "$datafolder
  python3 save_video.py $nstart $nend $nint $spatial_res $length_slice
  rm save_video.py  

  cd $SWEEP_DIR
done
