#!/bin/bash
BASEDIR=$PWD

#for reference
# id,x,y,z,mx,my,mz,nx,ny,nz,vx,vy,vz,normv,fphix,fphiy,fphiz,fsubx,fsuby,fsubz,fspringsx,fspringsy,fspringsz,tphix,tphiy,tphiz,tspringsx,tspringsy,tspringsz

writenormv=0
writenormvss=0
writevx=1
writevy=0
writevz=0
writealpha=0

listvfiles=()
listvxfiles=()
listvssfiles=()
listalphafiles=()

sweepingParam="u0"
sweepingRange=(1e-4 1e-5 1e-6)
prefix=$sweepingParam

for param in ${sweepingRange[@]}; do
  datafolder=$prefix"_"$param
 
  if [ $writenormv=1 ]
  then
    listvfiles+=($datafolder/normvOverTime)
    cd $datafolder
    rm -f normvOverTime

    echo "in "$datafolder" writing velocities... "
    for colfile in col*; do
      awk -F',' '{if($1 == 1) {print $14}}' $colfile | xargs >> normvOverTime
    done
    cd $BASEDIR
  fi

  if [ $writevx=1 ]
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


  if [ $writenormvss=1 ]
  then
    listvssfiles+=($datafolder/normvss)
    cd $datafolder
    rm -f normvss

    echo "in "$datafolder" writing steady state velocities... "
    awk -F',' '{if($1 == 1) {print $14}}' "col-cds00020000.csv" | xargs > normvss
    cd $BASEDIR
  fi

  if [ $writealpha=1 ]
  then
    listalphafiles+=($datafolder/alpha)
    cd $datafolder
    rm -f alpha

    echo "in "$datafolder" writing orientation alpha... "
    for dircolfile in dircol*; do
      awk -F',' 'FNR == 2 {print $10}' $dircolfile | xargs >> alpha
    done
    cd $BASEDIR
  fi

done

if [ $writenormv=1 ]
then
  paste -d , "${listvfiles[@]}" > normvOverTimeAndFolders
  echo "Velocities over time are in normvOverTimeAndFolders"
fi

if [ $writevx=1 ]
then
  paste -d , "${listvxfiles[@]}" > vxOverTimeAndFolders
  echo "vx over time are in vxOverTimeAndFolders"
fi

if [ $writenormvss=1 ]
then
  paste -d , "${listvssfiles[@]}" > normvssOverFolders
  echo "Steady state velocities are in normvssOverFolders"
fi

if [ $writealpha=1 ]
then
  paste -d , "${listalphafiles[@]}" > alphaOverTimeAndFolders
  echo "Alphas over time are in alphaOverTimeAndFolders"
fi

echo "Complete"
