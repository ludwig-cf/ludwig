#!/bin/bash
BASE_DIR_PATH=$PWD
SAVE_DIR_NAME="graph_data"
SAVE_DIR_PATH=$BASE_DIR_PATH"/"$SAVE_DIR_NAME

rm -r $SAVE_DIR_PATH
mkdir $SAVE_DIR_PATH

#for reference
# col-cds 
# id,x,y,z,mx,my,mz,nx,ny,nz,vx,vy,vz,normv,fphix,fphiy,fphiz,fsubx,fsuby,fsubz,fspringsx,fspringsy,fspringsz,tphix,tphiy,tphiz,tspringsx,tspringsy,tspringsz,iscentre, ishole

# dircol-cds  
# x0,y0,z0,mx,my,mz,nx,ny,nz,alpha,sumtx,sumty,sumtz,sumfx,sumfy,sumfz

writerc=1
writerh=1
writevc=1
writevh=1
writem=0 
writet=0
writef=0

listxcfiles=(); listycfiles=(); listzcfiles=()
listxhfiles=(); listyhfiles=(); listzhfiles=()

listvcxfiles=(); listvcyfiles=(); listvczfiles=()
listvhxfiles=(); listvhyfiles=(); listvhzfiles=()

listmxfiles=(); listmyfiles=(); listmzfiles=()
listtxfiles=(); listtyfiles=(); listtzfiles=()
listfxfiles=(); listfyfiles=(); listfzfiles=()

first_folder=""
first_parameters=("fullerene" "hexasphere" "trisphere")

second_folder=""
second_parameters=(1e-3 1e-4 1e-5 1e-6)

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
        target_folder=$first_folder""$first_parameter"/"$second_folder""$second_parameter
        target_folder_list+=($target_folder)
      done
    done
  fi
fi

for target_folder in ${target_folder_list[@]}; do
  echo "Working in "$target_folder

  if [ $writerc -eq 1 ]
  then
    listxcfiles+=($target_folder/xcOverTime)
    listycfiles+=($target_folder/ycOverTime)
    listzcfiles+=($target_folder/zcOverTime)
    cd $target_folder
    rm -f xcOverTime
    rm -f ycOverTime
    rm -f zcOverTime
    echo "	writing r centre... "
    for colfile in col*; do
      awk -F',' '{if($33 == 1) {print $2}}' $colfile | xargs >> xcOverTime
      awk -F',' '{if($33 == 1) {print $3}}' $colfile | xargs >> ycOverTime
      awk -F',' '{if($33 == 1) {print $4}}' $colfile | xargs >> zcOverTime
    done
    cd $BASE_DIR_PATH
  fi

  if [ $writerh -eq 1 ]
  then
    listxhfiles+=($target_folder/xhOverTime)
    listyhfiles+=($target_folder/yhOverTime)
    listzhfiles+=($target_folder/zhOverTime)
    cd $target_folder
    rm -f xhOverTime
    rm -f yhOverTime
    rm -f zhOverTime
    echo "	writing r hole... "
    for colfile in col*; do
      awk -F',' '{if($34 == 1) {print $2}}' $colfile | xargs >> xhOverTime
      awk -F',' '{if($34 == 1) {print $3}}' $colfile | xargs >> yhOverTime
      awk -F',' '{if($34 == 1) {print $4}}' $colfile | xargs >> zhOverTime
    done
    cd $BASE_DIR_PATH
  fi

  if [ $writevc -eq 1 ]
  then
    listvcxfiles+=($target_folder/vcxOverTime)
    listvcyfiles+=($target_folder/vcyOverTime)
    listvczfiles+=($target_folder/vczOverTime)
    cd $target_folder
    rm -f vcxOverTime
    rm -f vcyOverTime
    rm -f vczOverTime
    echo "	writing v centre... "
    for colfile in col*; do
      awk -F',' '{if($33 == 1) {print $11}}' $colfile | xargs >> vcxOverTime
      awk -F',' '{if($33 == 1) {print $12}}' $colfile | xargs >> vcyOverTime
      awk -F',' '{if($33 == 1) {print $13}}' $colfile | xargs >> vczOverTime
    done
    cd $BASE_DIR_PATH
  fi

  if [ $writevh -eq 1 ]
  then
    listvhxfiles+=($target_folder/vhxOverTime)
    listvhyfiles+=($target_folder/vhyOverTime)
    listvhzfiles+=($target_folder/vhzOverTime)
    cd $target_folder
    rm -f vhxOverTime
    rm -f vhyOverTime
    rm -f vhzOverTime
    echo "	writing v hole... "
    for colfile in col*; do
      awk -F',' '{if($34 == 1) {print $11}}' $colfile | xargs >> vhxOverTime
      awk -F',' '{if($34 == 1) {print $12}}' $colfile | xargs >> vhyOverTime
      awk -F',' '{if($34 == 1) {print $13}}' $colfile | xargs >> vhzOverTime
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
    printf "%s\n" "$target_folder" >> $SAVE_DIR_PATH"/"xcOverTimeAndFolders
    printf "%s\n" "$target_folder" >> $SAVE_DIR_PATH"/"ycOverTimeAndFolders
    printf "%s\n" "$target_folder" >> $SAVE_DIR_PATH"/"zcOverTimeAndFolders
    printf "%s\n" "$target_folder" >> $SAVE_DIR_PATH"/"xhOverTimeAndFolders
    printf "%s\n" "$target_folder" >> $SAVE_DIR_PATH"/"yhOverTimeAndFolders
    printf "%s\n" "$target_folder" >> $SAVE_DIR_PATH"/"zhOverTimeAndFolders
    printf "%s\n" "$target_folder" >> $SAVE_DIR_PATH"/"vcxOverTimeAndFolders
    printf "%s\n" "$target_folder" >> $SAVE_DIR_PATH"/"vcyOverTimeAndFolders
    printf "%s\n" "$target_folder" >> $SAVE_DIR_PATH"/"vczOverTimeAndFolders
    printf "%s\n" "$target_folder" >> $SAVE_DIR_PATH"/"vhxOverTimeAndFolders
    printf "%s\n" "$target_folder" >> $SAVE_DIR_PATH"/"vhyOverTimeAndFolders
    printf "%s\n" "$target_folder" >> $SAVE_DIR_PATH"/"vhzOverTimeAndFolders
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
    printf "%s," "$target_folder" >> $SAVE_DIR_PATH"/"xcOverTimeAndFolders
    printf "%s," "$target_folder" >> $SAVE_DIR_PATH"/"ycOverTimeAndFolders
    printf "%s," "$target_folder" >> $SAVE_DIR_PATH"/"zcOverTimeAndFolders
    printf "%s," "$target_folder" >> $SAVE_DIR_PATH"/"xhOverTimeAndFolders
    printf "%s," "$target_folder" >> $SAVE_DIR_PATH"/"yhOverTimeAndFolders
    printf "%s," "$target_folder" >> $SAVE_DIR_PATH"/"zhOverTimeAndFolders
 
    printf "%s," "$target_folder" >> $SAVE_DIR_PATH"/"vcxOverTimeAndFolders
    printf "%s," "$target_folder" >> $SAVE_DIR_PATH"/"vcyOverTimeAndFolders
    printf "%s," "$target_folder" >> $SAVE_DIR_PATH"/"vczOverTimeAndFolders
    printf "%s," "$target_folder" >> $SAVE_DIR_PATH"/"vhxOverTimeAndFolders
    printf "%s," "$target_folder" >> $SAVE_DIR_PATH"/"vhyOverTimeAndFolders
    printf "%s," "$target_folder" >> $SAVE_DIR_PATH"/"vhzOverTimeAndFolders
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

if [ $writerc -eq 1 ]
then
  paste -d , "${listxcfiles[@]}" >> $SAVE_DIR_PATH"/"xcOverTimeAndFolders
  paste -d , "${listycfiles[@]}" >> $SAVE_DIR_PATH"/"ycOverTimeAndFolders
  paste -d , "${listzcfiles[@]}" >> $SAVE_DIR_PATH"/"zcOverTimeAndFolders
fi

if [ $writerh -eq 1 ]
then
  paste -d , "${listxhfiles[@]}" >> $SAVE_DIR_PATH"/"xhOverTimeAndFolders
  paste -d , "${listyhfiles[@]}" >> $SAVE_DIR_PATH"/"yhOverTimeAndFolders
  paste -d , "${listzhfiles[@]}" >> $SAVE_DIR_PATH"/"zhOverTimeAndFolders
fi

if [ $writevc -eq 1 ]
then
  paste -d , "${listvcxfiles[@]}" >> $SAVE_DIR_PATH"/"vcxOverTimeAndFolders
  paste -d , "${listvcyfiles[@]}" >> $SAVE_DIR_PATH"/"vcyOverTimeAndFolders
  paste -d , "${listvczfiles[@]}" >> $SAVE_DIR_PATH"/"vczOverTimeAndFolders
fi

if [ $writevh -eq 1 ]
then
  paste -d , "${listvhxfiles[@]}" >> $SAVE_DIR_PATH"/"vhxOverTimeAndFolders
  paste -d , "${listvhyfiles[@]}" >> $SAVE_DIR_PATH"/"vhyOverTimeAndFolders
  paste -d , "${listvhzfiles[@]}" >> $SAVE_DIR_PATH"/"vhzOverTimeAndFolders
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
