#!/bin/bash
BASE_DIR_PATH=$PWD
SAVE_DIR_NAME="graph_data"
SAVE_DIR_PATH=$BASE_DIR_PATH"/"$SAVE_DIR_NAME

rm -r $SAVE_DIR_PATH
mkdir $SAVE_DIR_PATH

#for reference
# col-cds 
# id,x,y,z,mx,my,mz,nx,ny,nz,vx,vy,vz,normv,fphix,fphiy,fphiz,fsubx,fsuby,fsubz,fspringsx,fspringsy,fspringsz,tphix,tphiy,tphiz,tspringsx,tspringsy,tspringsz

# dircol-cds  
# x0,y0,z0,mx,my,mz,nx,ny,nz,alpha,sumtx,sumty,sumtz,sumfx,sumfy,sumfz

writev=1
writem=1 
writet=1
writef=1

listvxfiles=(); listvyfiles=(); listvzfiles=()
listmxfiles=(); listmyfiles=(); listmzfiles=()
listtxfiles=(); listtyfiles=(); listtzfiles=()
listfxfiles=(); listfyfiles=(); listfzfiles=()

first_folder="vesicle_radius"
first_parameters=(6.0 7.0 8.0 9.0 10.0)

second_folder="phi_interaction_external"
second_parameters=(0)

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

for target_folder in ${target_folder_list[@]}; do
  echo "Working in "$target_folder

  if [ $writev -eq 1 ]
  then
    listvxfiles+=($target_folder/vxOverTime)
    listvyfiles+=($target_folder/vyOverTime)
    listvzfiles+=($target_folder/vzOverTime)
    cd $target_folder
    rm -f vxOverTime
    rm -f vyOverTime
    rm -f vzOverTime
    echo "	writing v... "
    for colfile in col*; do
      awk -F',' '{if(NR == 2) {print $11}}' $colfile | xargs >> vxOverTime
      awk -F',' '{if(NR == 2) {print $12}}' $colfile | xargs >> vyOverTime
      awk -F',' '{if(NR == 2) {print $13}}' $colfile | xargs >> vzOverTime
    done
    cd $BASE_DIR_PATH
  fi

  if [ $writem -eq 1 ]
  then
    listmxfiles+=($target_folder/mxOverTime)
    listmyfiles+=($target_folder/myOverTime)
    listmzfiles+=($target_folder/mzOverTime)
    cd $target_folder
    rm -f mxOverTime
    rm -f myOverTime
    rm -f mzOverTime
    echo "	writing m... "
    for colfile in col*; do
      awk -F',' '{if(NR == 2) {print $5}}' $colfile | xargs >> mxOverTime
      awk -F',' '{if(NR == 2) {print $6}}' $colfile | xargs >> myOverTime
      awk -F',' '{if(NR == 2) {print $7}}' $colfile | xargs >> mzOverTime
    done
    cd $BASE_DIR_PATH
  fi

  if [ $writet -eq 1 ]
  then
    listtxfiles+=($target_folder/txOverTime)
    listtyfiles+=($target_folder/tyOverTime)
    listtzfiles+=($target_folder/tzOverTime)
    cd $target_folder
    rm -f txOverTime
    rm -f tyOverTime
    rm -f tzOverTime
    echo "	writing t... "
    for dircolfile in dircol*; do
      awk -F',' '{if(NR == 2) {print $11}}' $dircolfile | xargs >> txOverTime
      awk -F',' '{if(NR == 2) {print $12}}' $dircolfile | xargs >> tyOverTime
      awk -F',' '{if(NR == 2) {print $13}}' $dircolfile | xargs >> tzOverTime
    done
    cd $BASE_DIR_PATH
  fi

  if [ $writef -eq 1 ]
  then
    listfxfiles+=($target_folder/fxOverTime)
    listfyfiles+=($target_folder/fyOverTime)
    listfzfiles+=($target_folder/fzOverTime)
    cd $target_folder
    rm -f fxOverTime
    rm -f fyOverTime
    rm -f fzOverTime
    echo "	writing f... "
    for dircolfile in dircol*; do
      awk -F',' '{if(NR == 2) {print $14}}' $dircolfile | xargs >> fxOverTime
      awk -F',' '{if(NR == 2) {print $15}}' $dircolfile | xargs >> fyOverTime
      awk -F',' '{if(NR == 2) {print $16}}' $dircolfile | xargs >> fzOverTime
    done
    cd $BASE_DIR_PATH
  fi


done

last_folder=${target_folder_list[-1]}
for target_folder in ${target_folder_list[@]}; do
  if [[ $target_folder == $last_folder ]]
  then
    printf "%s\n" "$target_folder" >> $SAVE_DIR_PATH"/"vxOverTimeAndFolders
    printf "%s\n" "$target_folder" >> $SAVE_DIR_PATH"/"vyOverTimeAndFolders
    printf "%s\n" "$target_folder" >> $SAVE_DIR_PATH"/"vzOverTimeAndFolders
    printf "%s\n" "$target_folder" >> $SAVE_DIR_PATH"/"mxOverTimeAndFolders
    printf "%s\n" "$target_folder" >> $SAVE_DIR_PATH"/"myOverTimeAndFolders
    printf "%s\n" "$target_folder" >> $SAVE_DIR_PATH"/"mzOverTimeAndFolders
    printf "%s\n" "$target_folder" >> $SAVE_DIR_PATH"/"txOverTimeAndFolders
    printf "%s\n" "$target_folder" >> $SAVE_DIR_PATH"/"tyOverTimeAndFolders
    printf "%s\n" "$target_folder" >> $SAVE_DIR_PATH"/"tzOverTimeAndFolders
    printf "%s\n" "$target_folder" >> $SAVE_DIR_PATH"/"fxOverTimeAndFolders
    printf "%s\n" "$target_folder" >> $SAVE_DIR_PATH"/"fyOverTimeAndFolders
    printf "%s\n" "$target_folder" >> $SAVE_DIR_PATH"/"fzOverTimeAndFolders
  else
    printf "%s," "$target_folder" >> $SAVE_DIR_PATH"/"vxOverTimeAndFolders
    printf "%s," "$target_folder" >> $SAVE_DIR_PATH"/"vyOverTimeAndFolders
    printf "%s," "$target_folder" >> $SAVE_DIR_PATH"/"vzOverTimeAndFolders
    printf "%s," "$target_folder" >> $SAVE_DIR_PATH"/"mxOverTimeAndFolders
    printf "%s," "$target_folder" >> $SAVE_DIR_PATH"/"myOverTimeAndFolders
    printf "%s," "$target_folder" >> $SAVE_DIR_PATH"/"mzOverTimeAndFolders
    printf "%s," "$target_folder" >> $SAVE_DIR_PATH"/"txOverTimeAndFolders
    printf "%s," "$target_folder" >> $SAVE_DIR_PATH"/"tyOverTimeAndFolders
    printf "%s," "$target_folder" >> $SAVE_DIR_PATH"/"tzOverTimeAndFolders
    printf "%s," "$target_folder" >> $SAVE_DIR_PATH"/"fxOverTimeAndFolders
    printf "%s," "$target_folder" >> $SAVE_DIR_PATH"/"fyOverTimeAndFolders
    printf "%s," "$target_folder" >> $SAVE_DIR_PATH"/"fzOverTimeAndFolders
  fi      
done


if [ $writev -eq 1 ]
then
  paste -d , "${listvxfiles[@]}" >> $SAVE_DIR_PATH"/"vxOverTimeAndFolders
  paste -d , "${listvyfiles[@]}" >> $SAVE_DIR_PATH"/"vyOverTimeAndFolders
  paste -d , "${listvzfiles[@]}" >> $SAVE_DIR_PATH"/"vzOverTimeAndFolders
fi

if [ $writem -eq 1 ]
then
  paste -d , "${listmxfiles[@]}" >> $SAVE_DIR_PATH"/"mxOverTimeAndFolders
  paste -d , "${listmyfiles[@]}" >> $SAVE_DIR_PATH"/"myOverTimeAndFolders
  paste -d , "${listmzfiles[@]}" >> $SAVE_DIR_PATH"/"mzOverTimeAndFolders
fi

if [ $writet -eq 1 ]
then
  paste -d , "${listtxfiles[@]}" >> $SAVE_DIR_PATH"/"txOverTimeAndFolders
  paste -d , "${listtyfiles[@]}" >> $SAVE_DIR_PATH"/"tyOverTimeAndFolders
  paste -d , "${listtzfiles[@]}" >> $SAVE_DIR_PATH"/"tzOverTimeAndFolders
fi

if [ $writef -eq 1 ]
then
  paste -d , "${listfxfiles[@]}" >> $SAVE_DIR_PATH"/"fxOverTimeAndFolders
  paste -d , "${listfyfiles[@]}" >> $SAVE_DIR_PATH"/"fyOverTimeAndFolders
  paste -d , "${listfzfiles[@]}" >> $SAVE_DIR_PATH"/"fzOverTimeAndFolders
fi



echo "Extraction complete"
