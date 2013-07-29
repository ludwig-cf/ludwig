#! /bin/bash

for t in 128 256; do
for i in sor fft; do

#extract lines with 'psi update', remove the filename and the :
grep 'psi update' time_${i}_$t* | sed 's/time.*._//g' | sed 's/://g' > ${i}time_$t

#grep 'nerst planck' time_${i}_$t* | sed 's/time.*._//g' | sed 's/://g' > ${i}plancktime_$t

#extract lines with '[psi]' and put into .dat file for printing the electric field of the system
grep '\[psi\]' time_${i}_${t}_128 > psi_${i}_${t}.dat


#this bit finds the time on 2 procs and pastes it back into the timefile
x=$(wc -l < ${i}time_${t})

y=$(grep 'psi update' time_${i}_${t}_2 |  awk 'BEGIN {} { print $6 } END {}' | tail -1)

echo x $x
echo y $y

touch temp

for (( z=1; z<=$x; z++ ))
do
echo $y >> temp
done

paste ${i}time_${t} temp > temp2
mv temp2 ${i}time_${t}

rm temp

done
done

