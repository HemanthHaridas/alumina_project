#!/bin/bash

for inp in *.inp
do
#	echo $inp 
#	outfile=$(awk '{print substr($0, 1, length($0)-4)}' <<< "$inp")
#	lmp="/Users/hemanthharidas/Desktop/Codes/lammps_test/lammps/build/lmp"
#	$lmp -i $inp > $outfile.log

  lmp="lmp"
  prefix=${inp%%.*}
  t1=$(date +%s%3N)
  mpirun $lmp -i $inp > $prefix.log 2> >(grep -v "PMIX ERROR" >&2)
  t2=$(date +%s%3N)
	dt=$(awk -v t1=$t1 -v t2=$t2 'BEGIN{print (t2-t1)/1000.}')
	echo "$prefix $dt"
	dtlist+="$dt "
done
echo -n "Total time: "
echo $dtlist | awk '{for(i=1; i<=NF; i++) sum+=$i; print sum}'
