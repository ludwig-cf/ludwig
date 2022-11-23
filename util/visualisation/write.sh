#!/bin/bash
BASEDIR=$PWD

#for reference
# id,x,y,z,mx,my,mz,nx,ny,nz,vx,vy,vz,normv,fphix,fphiy,fphiz,fsubx,fsuby,fsubz,fspringsx,fspringsy,fspringsz,tphix,tphiy,tphiz,tspringsx,tspringsy,tspringsz

writemx=1 
writemy=1
writemz=1
writevx=1
writevy=1
writevz=1

listvxfiles=()
listvyfiles=()
listvzfiles=()
listmxfiles=()
listmyfiles=()
listmzfiles=()

sweepingParam="beta"
sweepingRange=(0 pi4 pi2 3pi4 pi)
prefix=$sweepingParam
postfolders=("mask_phi_switch_0" "mask_phi_switch_1")

for param in ${sweepingRange[@]}; do
  for postfoldername in ${postfolders[@]}; do
    datafolder=$prefix"_"$param"/"$postfoldername

    if [ $writemx -eq 1 ]
    then
      listmxfiles+=($datafolder/mxOverTime)
      cd $datafolder
      rm -f mxOverTime

      echo "in "$datafolder" writing mx... "
      for colfile in col*; do
        awk -F',' '{if($1 == 1) {print $5}}' $colfile | xargs >> mxOverTime
      done
      cd $BASEDIR
    fi

    if [ $writemy -eq 1 ]
    then
      listmyfiles+=($datafolder/myOverTime)
      cd $datafolder
      rm -f myOverTime

      echo "in "$datafolder" writing my... "
      for colfile in col*; do
        awk -F',' '{if($1 == 1) {print $6}}' $colfile | xargs >> myOverTime
      done
      cd $BASEDIR
    fi

    if [ $writemz -eq 1 ]
    then
      listmzfiles+=($datafolder/mzOverTime)
      cd $datafolder
      rm -f mzOverTime

      echo "in "$datafolder" writing mz... "
      for colfile in col*; do
        awk -F',' '{if($1 == 1) {print $7}}' $colfile | xargs >> mzOverTime
      done
      cd $BASEDIR
    fi
 


    if [ $writevx -eq 1 ]
    then
      listvxfiles+=($datafolder/vxOverTime)
      cd $datafolder
      rm -f vxOverTime

      echo "in "$datafolder" writing vx... "
      for colfile in col*; do
        awk -F',' '{if($1 == 1) {print $11}}' $colfile | xargs >> vxOverTime
      done
      cd $BASEDIR
    fi

    if [ $writevy -eq 1 ]
    then
      listvyfiles+=($datafolder/vyOverTime)
      cd $datafolder
      rm -f vyOverTime

      echo "in "$datafolder" writing vy... "
      for colfile in col*; do
        awk -F',' '{if($1 == 1) {print $12}}' $colfile | xargs >> vyOverTime
      done
      cd $BASEDIR
    fi

    if [ $writevz -eq 1 ]
    then
      listvzfiles+=($datafolder/vzOverTime)
      cd $datafolder
      rm -f vzOverTime

      echo "in "$datafolder" writing vz... "
      for colfile in col*; do
        awk -F',' '{if($1 == 1) {print $13}}' $colfile | xargs >> vzOverTime
      done
      cd $BASEDIR
    fi
  done
done


if [ $writevx -eq 1 ]
then
  paste -d , "${listvxfiles[@]}" > vxOverTimeAndFolders
  echo "vx over time are in vxOverTimeAndFolders"
fi

if [ $writevy -eq 1 ]
then
  paste -d , "${listvyfiles[@]}" > vyOverTimeAndFolders
  echo "vy over time are in vyOverTimeAndFolders"
fi

if [ $writevz -eq 1 ]
then
  paste -d , "${listvzfiles[@]}" > vzOverTimeAndFolders
  echo "vz over time are in vzOverTimeAndFolders"
fi

if [ $writemx -eq 1 ]
then
  paste -d , "${listmxfiles[@]}" > mxOverTimeAndFolders
  echo "mx over time are in mxOverTimeAndFolders"
fi

if [ $writemy -eq 1 ]
then
  paste -d , "${listmyfiles[@]}" > myOverTimeAndFolders
  echo "my over time are in myOverTimeAndFolders"
fi

if [ $writemz -eq 1 ]
then
  paste -d , "${listmzfiles[@]}" > mzOverTimeAndFolders
  echo "mz over time are in mzOverTimeAndFolders"
fi

echo "Complete"
