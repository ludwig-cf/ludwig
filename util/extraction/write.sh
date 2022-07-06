#!/bin/bash
BASEDIR=$PWD

writenormv=1
writenormvss=1
writealpha=0

listvfiles=()
listvssfiles=()
listalphafiles=()

sweepingParam="trisphere1em"
sweepingRange=(3 4 5 6)
prefix=$sweepingParam

for param in ${sweepingRange[@]}; do
  datafolder=$prefix""$param
  
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

  if [ $writenormvss=1 ]
  then
    listvssfiles+=($datafolder/normvss)
    cd $datafolder
    rm -f normvss

    echo "in "$datafolder" writing steady state velocities... "
    awk -F',' '{if($1 == 1) {print $14}}' "col-cds00100000.csv" | xargs > normvss
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
