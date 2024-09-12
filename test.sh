#!/bin/bash

#	lmp="/Users/hemanthharidas/Desktop/Codes/lammps_test/lammps/build/lmp"
lmp="lmp_serial"

t1=$(date +%s%3N)
for inp in *.inp
do
#	echo $inp 
#	outfile=$(awk '{print substr($0, 1, length($0)-4)}' <<< "$inp")
#	$lmp -i $inp > $outfile.log

  prefix=${inp%%.*}
	echo "$prefix"
  $lmp -i $inp > $prefix.log 
done

echo -n "Total time: "
t2=$(date +%s%3N);dt=$(awk -v t1=$t1 -v t2=$t2 'BEGIN{print (t2-t1)/1000.}')
echo $dt

