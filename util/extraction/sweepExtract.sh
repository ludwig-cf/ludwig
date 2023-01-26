#!/bin/bash

#set -x
#trap read debug

# META PARAMETERS
SWEEP_DIR=$PWD"/"
UTIL_DIR=/home/jeremie/PhD/ludwig/util/

# SIMULATION PARAMETERS 
N_start=0
N_cycles=110000

# EXTRACTION PARAMETERS
freq=10000
nstart=$freq
nend=$N_cycles
nint=$freq
nfiles=$(($N_cycles / $freq))

zeroth_folders=(51cube_reorientation)
first_folder="vesicle_template"
first_parameters=(icosphere)

second_folder="boundary_walls"
second_parameters=(0_1_0)

target_folder_list=()

for zeroth_folder in ${zeroth_folders[@]}; do
  if [ ${#first_parameters[@]} -eq 0 ]; then
    if [ ${#second_parameters[@]} -eq 0 ]; then
      target_folder=$zeroth_folder"/"$first_folder"/"$second_folder
      target_folder_list+=($target_folder)
    else
      for second_parameter in ${second_parameters[@]}; do
        target_folder=$zeroth_folder"/"$first_folder"/"$second_folder"_"$second_parameter
        target_folder_list+=($target_folder)
      done
    fi
  else 
    if [ ${#second_parameters[@]} -eq 0 ]; then
      for first_parameter in ${first_parameters[@]}; do
        target_folder=$zeroth_folder"/"$first_folder"_"$first_parameter"/"$second_folder
        target_folder_list+=($target_folder)
      done
    else
      for first_parameter in ${first_parameters[@]}; do
        for second_parameter in ${second_parameters[@]}; do
          target_folder=$zeroth_folder"/"$first_folder"_"$first_parameter"/"$second_folder"_"$second_parameter
          target_folder_list+=($target_folder)
        done
      done
    fi
  fi
done

printf "\n CHECKING NUMBER OF FILES \n \n"
for target_folder in ${target_folder_list[@]}; do
  cd $target_folder
  rm *csv
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
  echo $target_folder
  
  cp $UTIL_DIR"extract" $target_folder
  cp $UTIL_DIR"extract_colloids" $target_folder
  cp $UTIL_DIR"extraction/extract.py" $target_folder
  cp $UTIL_DIR"extraction/make_dircols.py" $target_folder
  cp $UTIL_DIR"extraction/make_vesiclecols.py" $target_folder
  cd $target_folder

  python3 extract.py -s --nstart $nstart --nend $nend --nint $nint
  python3 make_dircols.py $nstart $nend $nint
  python3 make_vesiclecols.py $nstart $nend $nint
  rm extract.py extract extract_colloids make_dircols.py make_vesiclecols.py

  cd $SWEEP_DIR
done
