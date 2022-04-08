#!/bin/bash
for i in `seq 1000 1000 10000`;
do
  tstep=$(printf "%08d" $i)
  ./extract -k vel.001-001.meta vel-${tstep}.001-001  
  ./extract -k phi.001-001.meta phi-${tstep}.001-001
done
