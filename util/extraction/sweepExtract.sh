#!/bin/bash

#set -x
#trap read debug

# META PARAMETERS
SWEEP_DIR=$PWD"/"
UTIL_DIR=/home/jeremie/PHD/ludwig/util/

# SIMULATION PARAMETERS 
N_start=0
N_cycles=100

# EXTRACTION PARAMETERS
freq=10
nstart=$freq
nend=$N_cycles
nint=$freq
nfiles=$(($N_cycles / $freq))

# SWEEPING PARAMETER
# Must match with datafolders
sweepingParam="viscosity"
sweepingRange=(0.5 1.0)

prefix=$sweepingParam

printf "\n CHECKING NUMBER OF FILES \n \n"
for param in ${sweepingRange[@]}; do
  datafolder=$prefix"_"$param
  cd $datafolder
  
  if [[ "$(ls phi*-001 | wc -l)" -ne "$nfiles" || "$(ls config.cds0* | wc -l)" -ne "$nfiles" || "$(ls vel*-001 | wc -l)" -ne "$nfiles" ]]; 
  then
    echo "Number of files not matching in "$datafolder
  else 
    echo "Number of files matching in "$datafolder
  fi
  cd $SWEEP_DIR
done

while true; do
    read -p "Do you wish to continue ? " yn
    case $yn in
        [Yy]* ) break;;
        [Nn]* ) exit;;
        * ) echo "Please answer yes or no.";;
    esac
done

cd $SWEEP_DIR
printf "\n EXTRACTING \n \n"

for param in ${sweepingRange[@]}; do
  datafolder=$prefix"_"$param
   
  cp $UTIL_DIR"extract" $datafolder
  cp $UTIL_DIR"extract_colloids" $datafolder
  cp $UTIL_DIR"extraction/extract.py" $datafolder
  cp $UTIL_DIR"extraction/make_dircols.py" $datafolder
  cd $datafolder

  python3 extract.py -pmvc --nstart $nstart --nend $nend --nint $nint
  python3 make_dircols.py $nstart $nend $nint
  rm extract.py extract extract_colloids make_dircols.py

  cd $SWEEP_DIR
done
