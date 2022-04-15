#!/bin/bash
for i in `seq 1000 1000 10000`;
do
  tstep=$(printf "%08d" $i)
  ./extract -k vel.001-001.meta vel-${tstep}.001-001  
  ./extract -k -s -d q.001-001.meta q-${tstep}.001-001
  ./extract_colloids config.cds${tstep} 1 col-cdsvel-${tstep}.csv
done
