#!/bin/bash

#set -x
#trap read debug

# META PARAMETERS
BASEDIR=$PWD"/"
UTILCHEMODIR=~/PHD/utilchemo/
UTILDIR=~/PHD/ludwig/util/

# SIMULATION PARAMETERS 
N_start=0
N_cycles=600000

# EXTRACTION PARAMETERS
freq=10000
freqconfig=100000
nstart=$freq
nend=$N_cycles
nint=$freq
nfiles=$(($N_cycles / $freq))
nconfigfiles=$(($N_cycles / $freqconfig))

# SWEEPING PARAMETER
# Must match with datafolders
sweepingParam="PHIPROD"
sweepingRange=(0.001 0.005 0.01 0.05 0.1)

prefix=$sweepingParam

printf "\n CHECKING NUMBER OF FILES \n \n"
for param in ${sweepingRange[@]}; do
  datafolder=$prefix"_"$param
  cd $datafolder
  
  if [[ "$(ls phi*-001 | wc -l)" -ne "$nfiles" || "$(ls config.cds0* | wc -l)" -ne "$nfiles" || "$(ls vel*-001 | wc -l)" -ne "$nfiles" || "$(ls dist*001 | wc -l)" -ne "$nconfigfiles" ]]; then
    echo "Number of files not matching in "$datafolder
  fi
  cd $BASEDIR
done

while true; do
    read -p "Do you wish to continue ? " yn
    case $yn in
        [Yy]* ) break;;
        [Nn]* ) exit;;
        * ) echo "Please answer yes or no.";;
    esac
done

cd $BASEDIR
printf "\n EXTRACTING \n \n"

for param in ${sweepingRange[@]}; do
  datafolder=$prefix"_"$param
   
  cp $UTILDIR"extract" $datafolder
  cp $UTILDIR"extract_colloids" $datafolder
  cp $UTILCHEMODIR"extract.py" $datafolder
  cp $UTILCHEMODIR"make_dircols.py" $datafolder
  cd $datafolder

  python3 extract.py -pmvc --nstart $nstart --nend $nend --nint $nint
  python3 make_dircols.py $nstart $nend $nint
  rm extract.py extract extract_colloids make_dircols.py

  cd $BASEDIR
done
