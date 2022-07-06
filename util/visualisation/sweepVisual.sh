#!/bin/bash

#set -x
#trap read debug

# META PARAMETERS
BASEDIR=$PWD"/"
UTILCHEMODIR=~/PHD/utilchemo/
UTILDIR=~/PHD/ludwig/util/

# SIMULATION PARAMETERS 
N_start=0
N_cycles=100000

# EXTRACTION PARAMETERS
freq=10000
nstart=$freq
nend=$N_cycles
nint=$freq
nfiles=$(($N_cycles / $freq))
nconfigfiles=$(($N_cycles / $freqconfig))

# VIDEO PARAMETERS
spatial_res=300
length_slice=30

# SWEEPING PARAMETER
# Must match with datafolders
sweepingParam="PHIPROD"
sweepingRange=(0.001 0.005 0.01 0.05 0.1)

prefix=$sweepingParam

printf "\n CHECKING NUMBER OF FILES \n \n"
for param in ${sweepingRange[@]}; do
  datafolder=$prefix"_"$param
  cd $datafolder
  
  if [[ "$(ls phi*.vtk | wc -l)" -ne "$nfiles" || "$(ls col* | wc -l)" -ne "$nfiles" || "$(ls dircol* | wc -l)" -ne "$nfiles" || "$(ls mobility*.vtk | wc -l)" -ne "$nfiles" ]]; then
    echo "Number of files not matching in "$datafolder
  fi
  cd $BASEDIR
done
cd $BASEDIR
printf "\n EXTRACTING \n \n"


while true; do
    read -p "Do you wish to continue ? " yn
    case $yn in
        [Yy]* ) break;;
        [Nn]* ) exit;;
        * ) echo "Please answer yes or no.";;
    esac
done

printf "\n PROCEEDING \n \n"

for param in ${sweepingRange[@]}; do
  datafolder=$prefix"_"$param
   
  cp $UTILCHEMODIR"save_video.py" $datafolder

  cd $datafolder
  printf "Creating video in "$datafolder
  python3 save_video.py $nstart $nend $nint $spatial_res $length_slice
  rm save_video.py  

  cd $BASEDIR
done
