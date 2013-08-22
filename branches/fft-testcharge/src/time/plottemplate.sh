#! /bin/bash

function plotspeedup() {
for t in 64 128 256 512; do
cat <<EOF
  set style fill solid 1.0 border -1
  set pointsize 2
  set log y
  set log x
  set xlabel "MPI Tasks"
  set ylabel "Speedup"
  set term png
  set output "speedup${t}.png"
#  set term epslatex color
#  set output "codegraph.tex"
EOF
  if [ $t = 256 ]; then
    echo "plot \\"
    echo "\"ffttime_${t}\" u 1:(\$10*2/\$7), \"sortime_${t}\" u 1:(\$10*2/\$7), x,\\"
    echo "\"ffttime_16_${t}\" u 1:(\$10*2/\$7), \"ffttime_8_${t}\" u 1:(\$10*2/\$7)"
  else
    echo "plot \\"
    echo "\"ffttime_${t}\" u 1:(\$10*2/\$7), \"sortime_${t}\" u 1:(\$10*2/\$7), x"
  fi

done 

#  i_start=2
#  i_end=$(echo "$x - 7" | bc)
#  for i in $(seq $i_start $(echo $i_end -1 | bc)) ;# do
#    echo "\"timings.dat\" using $i:xticlabels(1), \\"
#  done
#  echo "\"timings.dat\" using $i_end:xticlabels(1)"
}

function plottime() {
for t in 64 128 256 512; do
cat <<EOF
  set style fill solid 1.0 border -1
  set pointsize 2
  set log y
  set log x
  set xlabel "MPI Tasks"
  set ylabel "Time (s)"
  set term png
  set output "abstime${t}.png"
#  set term epslatex color
#  set output "codegraph.tex"
EOF
  echo "plot \\"
  echo "\"ffttime_${t}\" u 1:7, \"sortime_${t}\" u 1:7"

done 
}

plotspeedup | gnuplot
plottime | gnuplot


#epstopdf codegraph.eps

