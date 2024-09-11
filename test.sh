#!/bin/bash

#	lmp="/Users/hemanthharidas/Desktop/Codes/lammps_test/lammps/build/lmp"
#lmp="lmp"
lmp="lmp_serial"
#rm all.inp 2> /dev/null
t1=$(date +%s%3N)
for inp in *.inp
do
#	echo $inp 
#	outfile=$(awk '{print substr($0, 1, length($0)-4)}' <<< "$inp")
#	$lmp -i $inp > $outfile.log

  prefix=${inp%%.*}
  #echo "log $prefix.log" >> all.inp
	#cat $inp >> all.inp
	#echo "clear" >> all.inp
	##-- separate input files: ---
	echo "$prefix"
  $lmp -i $inp > $prefix.log 
#	echo "$prefix $dt";	dtlist+="$dt "
done
echo -n "Total time: "
t2=$(date +%s%3N);dt=$(awk -v t1=$t1 -v t2=$t2 'BEGIN{print (t2-t1)/1000.}')
echo $dt
#echo $dtlist | awk '{for(i=1; i<=NF; i++) sum+=$i; print sum}'t1=$(date +%s%3N)

#	2> >(grep -v "PMIX ERROR" >&2)
#t1=$(date +%s%3N)
#$lmp -i all.inp > all.log
#t2=$(date +%s%3N)
#awk -v t1=$t1 -v t2=$t2 'BEGIN{print (t2-t1)/1000.}' 
