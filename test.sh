#!/bin/bash

for inp in *.inp
do
#	echo $inp 
	outfile=$(awk '{print substr($0, 1, length($0)-4)}' <<< "$inp")
	/Users/hemanthharidas/Desktop/Codes/lammps_test/lammps/build/lmp -i $inp > $outfile.log
done
