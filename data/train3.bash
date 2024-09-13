syslist="gibSEowCNAlAltraj gibSEowtraj al6OHtraj al6H2Otraj al6AlAltraj 3m1traj d2btod3traj 2m1tod2btraj d2tod1traj 32wOOtraj 32wOHtraj NaOH30wPTtraj m1NaOHPTtraj 32wtraj 32wPTtraj aloh42h2obtraj m1NaOHtraj 2m10traj 2m1btraj 2m1ctraj aloh4scanAlOH d1scan aloh4toaloh5scan 2m1tod3scan w2 w6 d1na d3na aloh4na w Naw Na5w NaOH Na wscan al3oh122na al3o3oh6na al3oh9na gibbulk boehmiteb gib001 gib001top gib001bot gib110 gib825"
products="gibSEowCNAlAltraj gibSEowtraj al6OHtraj al6H2Otraj al6AlAltraj 3m1traj d2btod3traj 2m1tod2btraj d2tod1traj 32wOOtraj 32wOHtraj NaOH30wPTtraj m1NaOHPTtraj 32wtraj 32wPTtraj 2m10traj 2m1btraj 2m1ctraj m1NaOHtraj aloh42h2obtraj aloh4scanAlOH wscan d1scan aloh4toaloh5scan 2m1tod3scan d3na d1na w2 w6 Na5w Naw gib001 gib825 gib110"
#btraj: 2m1tod2b
#ctraj: d2btod2
if [ "$calc" == 'Ereact' ]; then
awk "BEGIN{
    c=1.0
    WGT=10.0; print WGT*${DEspe[gibSEowCNAlAltraj]}
    WGT=10.0; print WGT*${DEspe[gibSEowtraj]}
    WGT=10.0; print WGT*${DEspe[al6OHtraj]}
    WGT=10.0; print WGT*${DEspe[al6H2Otraj]}
    WGT=10.0; print WGT*${DEspe[al6AlAltraj]}
    WGT=10.0; print WGT*${DEspe[3m1traj]}
    WGT=10.0; print WGT*${DEspe[d2btod3traj]}
    WGT=10.0; print WGT*${DEspe[2m1tod2btraj]}
    WGT=15.0; print WGT*${DEspe[d2tod1traj]}
    WGT=50.0; print WGT*${DEspe[32wOOtraj]}
    WGT=25.0; print WGT*${DEspe[32wOHtraj]}
    WGT=5.0; print WGT*${DEspe[NaOH30wPTtraj]}
    WGT=5.0; print WGT*${DEspe[m1NaOHPTtraj]}
    WGT=5.0; print WGT*${DEspe[32wtraj]}
    WGT=10.0; print WGT*${DEspe[32wPTtraj]}
    WGT=10.0; print WGT*${DEspe[2m10traj]}
    WGT=10.0; print WGT*${DEspe[2m1btraj]}
    WGT=10.0; print WGT*${DEspe[2m1ctraj]}
    WGT=300./133; print WGT*${DEspe[m1NaOHtraj]}
    WGT=4.0; print WGT*${DEspe[aloh42h2obtraj]}
    WGT=10.0; print WGT*${DEspe[aloh4scanAlOH]}
    WGT=20.0; print WGT*${DEspe[wscan]}
    WGT=10.0; print WGT*${DEspe[d1scan]}
    WGT=2.0; print WGT*${DEspe[aloh4toaloh5scan]}
    WGT=2.0; print WGT*${DEspe[2m1tod3scan]}
    WGT=1.0; print WGT*(c*(${E[d3na]}-(2*${E[aloh4na]}))-(${Er[d3na]}))
    WGT=1.0; print WGT*(c*(${E[d1na]}+${E[w]}-(2*${E[aloh4na]}))-(${Er[d1na]}))
    WGT=5.0; print WGT*(c*(${E[w2]}-(2*${E[w]}))-(${Er[w2]}))
    WGT=5.0; print WGT*(c*(${E[w6]}-(6*${E[w]}))-(${Er[w6]}))
    WGT=.25; print WGT*(c*(${E[Na5w]}-(5*${E[w]}+${E[Na]}))-(${Er[Na5w]}))
    WGT=1.0;  print WGT*(c*(${E[Naw]}-(${E[w]}+${E[Na]}))-(${Er[Naw]}))
    WGT=0.5; print WGT*(c*(${E[gib001]}-(${E[gib001top]}+${E[gib001bot]}))-(${Er[gib001]}))
    WGT=0.5; print WGT*(c*(${E[gib825]}-(${E[gibbulk]}))-(${Er[gib825]}))
    WGT=0.25; print WGT*(c*(${E[gib110]}-(${E[gibbulk]}))-(${Er[gib110]}))
  }" > 1.tmp

fi
