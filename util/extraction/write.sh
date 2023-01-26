#!/bin/bash
BASE_DIR_PATH=$PWD
SAVE_DIR_NAME="graph_data"
SAVE_DIR_PATH=$BASE_DIR_PATH"/"$SAVE_DIR_NAME

rm -r $SAVE_DIR_PATH
mkdir $SAVE_DIR_PATH

#for reference
# col-cds 
# id,x,y,z,mx,my,mz,nx,ny,nz,vx,vy,vz,normv,fphix,fphiy,fphiz,fsubx,fsuby,fsubz,fspringsx,fspringsy,fspringsz,tphix,tphiy,tphiz,tspringsx,tspringsy,tspringsz,total_forcex,total_forcey,total_forcez,total_torquex,total_torquey,total_torquez,iscentre,ishole

# dircol-cds  
# x0,y0,z0,mx,my,mz,nx,ny,nz,alpha,sumtx,sumty,sumtz,sumfx,sumfy,sumfz

writerc=1
writerh=1
writevc=0
writevh=0
writem=0
writetphi=0
writefphi=0
writettot=0
writeftot=0
writevesicleftot=0
writevesiclettot=1

listxcfiles=(); listycfiles=(); listzcfiles=()
listxhfiles=(); listyhfiles=(); listzhfiles=()

listvcxfiles=(); listvcyfiles=(); listvczfiles=()
listvhxfiles=(); listvhyfiles=(); listvhzfiles=()

listmxfiles=(); listmyfiles=(); listmzfiles=()
listtphixfiles=(); listtphiyfiles=(); listtphizfiles=()
listfphixfiles=(); listfphiyfiles=(); listfphizfiles=()

listvesicleftotxfiles=();listvesicleftotyfiles=();listvesicleftotzfiles=();
listvesiclettotxfiles=();listvesiclettotyfiles=();listvesiclettotzfiles=();

zeroth_folders=(33cube_reorientation 39cube_reorientation 45cube_reorientation 51cube_reorientation)

first_folder="vesicle_template"
first_parameters=(icosphere trisphere)

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
      awk -F',' '{if($36 == 1) {print $2}}' $colfile | xargs >> xcOverTime
      awk -F',' '{if($36 == 1) {print $3}}' $colfile | xargs >> ycOverTime
      awk -F',' '{if($36 == 1) {print $4}}' $colfile | xargs >> zcOverTime
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
      awk -F',' '{if($37 == 1) {print $2}}' $colfile | xargs >> xhOverTime
      awk -F',' '{if($37 == 1) {print $3}}' $colfile | xargs >> yhOverTime
      awk -F',' '{if($37 == 1) {print $4}}' $colfile | xargs >> zhOverTime
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
      awk -F',' '{if($36 == 1) {print $11}}' $colfile | xargs >> vcxOverTime
      awk -F',' '{if($36 == 1) {print $12}}' $colfile | xargs >> vcyOverTime
      awk -F',' '{if($36 == 1) {print $13}}' $colfile | xargs >> vczOverTime
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
      awk -F',' '{if($37 == 1) {print $11}}' $colfile | xargs >> vhxOverTime
      awk -F',' '{if($37 == 1) {print $12}}' $colfile | xargs >> vhyOverTime
      awk -F',' '{if($37 == 1) {print $13}}' $colfile | xargs >> vhzOverTime
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

  if [ $writetphi -eq 1 ]
  then
    listtphixfiles+=($target_folder/tphixOverTime)
    listtphiyfiles+=($target_folder/tphiyOverTime)
    listtphizfiles+=($target_folder/tphizOverTime)
    cd $target_folder
    rm -f tphixOverTime
    rm -f tphiyOverTime
    rm -f tphizOverTime
    echo "	writing tphi... "
    for dircolfile in dircol*; do
      awk -F',' '{if(NR == 2) {print $11}}' $dircolfile | xargs >> tphixOverTime
      awk -F',' '{if(NR == 2) {print $12}}' $dircolfile | xargs >> tphiyOverTime
      awk -F',' '{if(NR == 2) {print $13}}' $dircolfile | xargs >> tphizOverTime
    done
    cd $BASE_DIR_PATH
  fi

  if [ $writefphi -eq 1 ]
  then
    listfphixfiles+=($target_folder/fphixOverTime)
    listfphiyfiles+=($target_folder/fphiyOverTime)
    listfphizfiles+=($target_folder/fphizOverTime)
    cd $target_folder
    rm -f fphixOverTime
    rm -f fphiyOverTime
    rm -f fphizOverTime
    echo "	writing fphi... "
    for dircolfile in dircol*; do
      awk -F',' '{if(NR == 2) {print $14}}' $dircolfile | xargs >> fphixOverTime
      awk -F',' '{if(NR == 2) {print $15}}' $dircolfile | xargs >> fphiyOverTime
      awk -F',' '{if(NR == 2) {print $16}}' $dircolfile | xargs >> fphizOverTime
    done
    cd $BASE_DIR_PATH
  fi

  if [ $writeftot -eq 1 ]
  then
    listftotxfiles+=($target_folder/ftotxOverTime)
    listftotyfiles+=($target_folder/ftotyOverTime)
    listftotzfiles+=($target_folder/ftotzOverTime)
    cd $target_folder
    rm -f ftotxOverTime
    rm -f ftotyOverTime
    rm -f ftotzOverTime
    echo "	writing ftot... "
    for colfile in col*; do
      awk -F',' '{if(NR == 2) {print $30}}' $dircolfile | xargs >> ftotxOverTime
      awk -F',' '{if(NR == 2) {print $31}}' $dircolfile | xargs >> ftotyOverTime
      awk -F',' '{if(NR == 2) {print $32}}' $dircolfile | xargs >> ftotzOverTime
    done
    cd $BASE_DIR_PATH
  fi

  if [ $writettot -eq 1 ]
  then
    listttotxfiles+=($target_folder/ttotxOverTime)
    listttotyfiles+=($target_folder/ttotyOverTime)
    listttotzfiles+=($target_folder/ttotzOverTime)
    cd $target_folder
    rm -f ttotxOverTime
    rm -f ttotyOverTime
    rm -f ttotzOverTime
    echo "	writing ttot... "
    for colfile in col*; do
      awk -F',' '{if(NR == 2) {print $33}}' $dircolfile | xargs >> ttotxOverTime
      awk -F',' '{if(NR == 2) {print $34}}' $dircolfile | xargs >> ttotyOverTime
      awk -F',' '{if(NR == 2) {print $35}}' $dircolfile | xargs >> ttotzOverTime
    done
    cd $BASE_DIR_PATH
  fi

  if [ $writevesicleftot -eq 1 ]
  then
    listvesicleftotxfiles+=($target_folder/vesicleftotxOverTime)
    listvesicleftotyfiles+=($target_folder/vesicleftotyOverTime)
    listvesicleftotzfiles+=($target_folder/vesicleftotzOverTime)
    cd $target_folder
    rm -f vesicleftotxOverTime
    rm -f vesicleftotyOverTime
    rm -f vesicleftotzOverTime
    echo "	writing vesicleftot... "
    for vesiclecolfile in vesiclecol*; do
      awk -F',' '{if(NR == 2) {print $16}}' $vesiclecolfile | xargs >> vesicleftotxOverTime
      awk -F',' '{if(NR == 2) {print $17}}' $vesiclecolfile | xargs >> vesicleftotyOverTime
      awk -F',' '{if(NR == 2) {print $18}}' $vesiclecolfile | xargs >> vesicleftotzOverTime
    done
    cd $BASE_DIR_PATH
  fi

  if [ $writevesiclettot -eq 1 ]
  then
    listvesiclettotxfiles+=($target_folder/vesiclettotxOverTime)
    listvesiclettotyfiles+=($target_folder/vesiclettotyOverTime)
    listvesiclettotzfiles+=($target_folder/vesiclettotzOverTime)
    cd $target_folder
    rm -f vesiclettotxOverTime
    rm -f vesiclettotyOverTime
    rm -f vesiclettotzOverTime
    echo "	writing vesiclettot... "
    for vesiclecolfile in vesiclecol*; do
      awk -F',' '{if(NR == 2) {print $19}}' $vesiclecolfile | xargs >> vesiclettotxOverTime
      awk -F',' '{if(NR == 2) {print $20}}' $vesiclecolfile | xargs >> vesiclettotyOverTime
      awk -F',' '{if(NR == 2) {print $21}}' $vesiclecolfile | xargs >> vesiclettotzOverTime
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
    printf "%s\n" "$target_folder" >> $SAVE_DIR_PATH"/"tphixOverTimeAndFolders
    printf "%s\n" "$target_folder" >> $SAVE_DIR_PATH"/"tphiyOverTimeAndFolders
    printf "%s\n" "$target_folder" >> $SAVE_DIR_PATH"/"tphizOverTimeAndFolders
    printf "%s\n" "$target_folder" >> $SAVE_DIR_PATH"/"fphixOverTimeAndFolders
    printf "%s\n" "$target_folder" >> $SAVE_DIR_PATH"/"fphiyOverTimeAndFolders
    printf "%s\n" "$target_folder" >> $SAVE_DIR_PATH"/"fphizOverTimeAndFolders
    printf "%s\n" "$target_folder" >> $SAVE_DIR_PATH"/"ttotxOverTimeAndFolders
    printf "%s\n" "$target_folder" >> $SAVE_DIR_PATH"/"ttotyOverTimeAndFolders
    printf "%s\n" "$target_folder" >> $SAVE_DIR_PATH"/"ttotzOverTimeAndFolders
    printf "%s\n" "$target_folder" >> $SAVE_DIR_PATH"/"ftotxOverTimeAndFolders
    printf "%s\n" "$target_folder" >> $SAVE_DIR_PATH"/"ftotyOverTimeAndFolders
    printf "%s\n" "$target_folder" >> $SAVE_DIR_PATH"/"ftotzOverTimeAndFolders
    printf "%s\n" "$target_folder" >> $SAVE_DIR_PATH"/"vesicleftotxOverTimeAndFolders
    printf "%s\n" "$target_folder" >> $SAVE_DIR_PATH"/"vesicleftotyOverTimeAndFolders
    printf "%s\n" "$target_folder" >> $SAVE_DIR_PATH"/"vesicleftotzOverTimeAndFolders
    printf "%s\n" "$target_folder" >> $SAVE_DIR_PATH"/"vesiclettotxOverTimeAndFolders
    printf "%s\n" "$target_folder" >> $SAVE_DIR_PATH"/"vesiclettotyOverTimeAndFolders
    printf "%s\n" "$target_folder" >> $SAVE_DIR_PATH"/"vesiclettotzOverTimeAndFolders

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
    printf "%s," "$target_folder" >> $SAVE_DIR_PATH"/"tphixOverTimeAndFolders
    printf "%s," "$target_folder" >> $SAVE_DIR_PATH"/"tphiyOverTimeAndFolders
    printf "%s," "$target_folder" >> $SAVE_DIR_PATH"/"tphizOverTimeAndFolders
    printf "%s," "$target_folder" >> $SAVE_DIR_PATH"/"fphixOverTimeAndFolders
    printf "%s," "$target_folder" >> $SAVE_DIR_PATH"/"fphiyOverTimeAndFolders
    printf "%s," "$target_folder" >> $SAVE_DIR_PATH"/"fphizOverTimeAndFolders
    printf "%s," "$target_folder" >> $SAVE_DIR_PATH"/"ttotxOverTimeAndFolders
    printf "%s," "$target_folder" >> $SAVE_DIR_PATH"/"ttotyOverTimeAndFolders
    printf "%s," "$target_folder" >> $SAVE_DIR_PATH"/"ttotzOverTimeAndFolders
    printf "%s," "$target_folder" >> $SAVE_DIR_PATH"/"ftotxOverTimeAndFolders
    printf "%s," "$target_folder" >> $SAVE_DIR_PATH"/"ftotyOverTimeAndFolders
    printf "%s," "$target_folder" >> $SAVE_DIR_PATH"/"ftotzOverTimeAndFolders
    printf "%s," "$target_folder" >> $SAVE_DIR_PATH"/"vesicleftotxOverTimeAndFolders
    printf "%s," "$target_folder" >> $SAVE_DIR_PATH"/"vesicleftotyOverTimeAndFolders
    printf "%s," "$target_folder" >> $SAVE_DIR_PATH"/"vesicleftotzOverTimeAndFolders
    printf "%s," "$target_folder" >> $SAVE_DIR_PATH"/"vesiclettotxOverTimeAndFolders
    printf "%s," "$target_folder" >> $SAVE_DIR_PATH"/"vesiclettotyOverTimeAndFolders
    printf "%s," "$target_folder" >> $SAVE_DIR_PATH"/"vesiclettotzOverTimeAndFolders
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

if [ $writetphi -eq 1 ]
then
  paste -d , "${listtphixfiles[@]}" >> $SAVE_DIR_PATH"/"tphixOverTimeAndFolders
  paste -d , "${listtphiyfiles[@]}" >> $SAVE_DIR_PATH"/"tphiyOverTimeAndFolders
  paste -d , "${listtphizfiles[@]}" >> $SAVE_DIR_PATH"/"tphizOverTimeAndFolders
fi

if [ $writefphi -eq 1 ]
then
  paste -d , "${listfphixfiles[@]}" >> $SAVE_DIR_PATH"/"fphixOverTimeAndFolders
  paste -d , "${listfphiyfiles[@]}" >> $SAVE_DIR_PATH"/"fphiyOverTimeAndFolders
  paste -d , "${listfphizfiles[@]}" >> $SAVE_DIR_PATH"/"fphizOverTimeAndFolders
fi

if [ $writettot -eq 1 ]
then
  paste -d , "${listttotxfiles[@]}" >> $SAVE_DIR_PATH"/"ttotxOverTimeAndFolders
  paste -d , "${listttotyfiles[@]}" >> $SAVE_DIR_PATH"/"ttotyOverTimeAndFolders
  paste -d , "${listttotzfiles[@]}" >> $SAVE_DIR_PATH"/"ttotzOverTimeAndFolders
fi

if [ $writeftot -eq 1 ]
then
  paste -d , "${listftotxfiles[@]}" >> $SAVE_DIR_PATH"/"ftotxOverTimeAndFolders
  paste -d , "${listftotyfiles[@]}" >> $SAVE_DIR_PATH"/"ftotyOverTimeAndFolders
  paste -d , "${listftotzfiles[@]}" >> $SAVE_DIR_PATH"/"ftotzOverTimeAndFolders
fi

if [ $writevesicleftot -eq 1 ]
then
  paste -d , "${listvesicleftotxfiles[@]}" >> $SAVE_DIR_PATH"/"vesicleftotxOverTimeAndFolders
  paste -d , "${listvesicleftotyfiles[@]}" >> $SAVE_DIR_PATH"/"vesicleftotyOverTimeAndFolders
  paste -d , "${listvesicleftotzfiles[@]}" >> $SAVE_DIR_PATH"/"vesicleftotzOverTimeAndFolders
fi

if [ $writevesiclettot -eq 1 ]
then
  paste -d , "${listvesiclettotxfiles[@]}" >> $SAVE_DIR_PATH"/"vesiclettotxOverTimeAndFolders
  paste -d , "${listvesiclettotyfiles[@]}" >> $SAVE_DIR_PATH"/"vesiclettotyOverTimeAndFolders
  paste -d , "${listvesiclettotzfiles[@]}" >> $SAVE_DIR_PATH"/"vesiclettotzOverTimeAndFolders
fi

echo "Extraction complete"
