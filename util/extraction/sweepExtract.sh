#!/bin/bash

#set -x
#trap read debug

# META PARAMETERS
SWEEP_DIR=$PWD"/"
UTIL_DIR=/home/jeremie/PhD/ludwig/util/

# SIMULATION PARAMETERS 
N_start=0
N_cycles=100000

# EXTRACTION PARAMETERS
freq=5000
nstart=$freq
nend=$N_cycles
nint=$freq
nfiles=$(($N_cycles / $freq))

first_folder="vesicle_radius"
first_parameters=(6.0 7.0 8.0 9.0 10.0)

second_folder="phi_interaction_external"
second_parameters=(1 0)

target_folder_list=()

if [ ${#first_parameters[@]} -eq 0 ]; then
  if [ ${#second_parameters[@]} -eq 0 ]; then
    target_folder=$first_folder"/"$second_folder
    target_folder_list+=($target_folder)
  else
    for second_parameter in ${second_parameters[@]}; do
      target_folder=$first_folder"/"$second_folder"_"$second_parameter
      target_folder_list+=($target_folder)
    done
  fi
else 
  if [ ${#second_parameters[@]} -eq 0 ]; then
    for first_parameter in ${first_parameters[@]}; do
      target_folder=$first_folder"_"$first_parameter"/"$second_folder
      target_folder_list+=($target_folder)
    done
  else
    for first_parameter in ${first_parameters[@]}; do
      for second_parameter in ${second_parameters[@]}; do
        target_folder=$first_folder"_"$first_parameter"/"$second_folder"_"$second_parameter
        target_folder_list+=($target_folder)
      done
    done
  fi
fi

printf "\n CHECKING NUMBER OF FILES \n \n"
for target_folder in ${target_folder_list[@]}; do
  cd $target_folder
  
  if [[ "$(ls phi*-001 | wc -l)" -ne "$nfiles" || "$(ls config.cds0* | wc -l)" -ne "$nfiles" || "$(ls vel*-001 | wc -l)" -ne "$nfiles" ]]; 
  then
    echo "Number of files not matching in "$target_folder
  else 
    echo "Number of files matching in "$target_folder
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

for target_folder in ${target_folder_list[@]}; do 
  cp $UTIL_DIR"extract" $target_folder
  cp $UTIL_DIR"extract_colloids" $target_folder
  cp $UTIL_DIR"extraction/extract.py" $target_folder
  cp $UTIL_DIR"extraction/make_dircols.py" $target_folder
  cd $target_folder

  python3 extract.py -pmvc --nstart $nstart --nend $nend --nint $nint
  python3 make_dircols.py $nstart $nend $nint
  rm extract.py extract extract_colloids make_dircols.py

  cd $SWEEP_DIR
done
