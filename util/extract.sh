#!/bin/bash
for i in $(seq 1000 1000 1000);
do
  tstep=$(printf "%08d" $i)
  ./extract -k -s q-${tstep}.001-001
  ./extract -k vel-${tstep}.001-001
  ./extract_colloids config.cds${tstep}.001-001
done
