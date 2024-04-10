import sys
import subprocess
import resource
import time
import os 
import nlopt

import scipy.optimize
import math
import numpy
import typing

import pandas

# helper functions
def createParameters(keys: typing.List[str], values: typing.List[float]) -> typing.Dict[str, str]:
	currentSet	=	{}
	for key, value in zip(keys, values):
		currentSet[key]	=	value
	return currentSet

# Initial parameters
paramVector_1	=	[ 	23.8239,  222.2696,  15.8096, 673.5521, 
					   -31.1295,  -16.6255,  26.8294,  16.0370, 
					  1779.9059,    0.4126,   6.6646,   5.0304, 
					   319.3682,  548.1608, 501.2680, 341.1614, 
					     1.1930,   31.3144, 180.6524,   7.6038,
					     0.8350,   -0.7506,  -0.4747,   2.8207,
					     1.5931,    1.0700,   0.2075,   0.0539,
					     0.0759,    2.0955,   3.4559,   1.2418,
					     3.2600,   80.0000,  70.0000, 235.0000, 
					   154.0000
					 ]

paramVector_2	=	numpy.array(paramVector_1)

keys	=	[	"chi_Al",     "chi_H",      "chi_Na",    "chi_O",
				"ct_AlOH",    "ct_HOH",     "eps_AlO",   "eps_AlOH",
				"eps_HOH",    "eps_NaO",    "eps_OH",    "eps_OO",  
				"eta_Al",     "eta_H",      "eta_Na",    "eta_O",   
				"gamma_Al",   "gamma_H",    "gamma_Na",  "gamma_O",
				"gaussH_Al",  "gaussH_AlO", "gaussH_OH", "gaussR_Al",
				"gaussR_AlO", "gaussR_OH",  "gaussW_Al", "gaussW_AlO",
				"gaussW_OH",  "sigma_AlO",  "sigma_NaO", "sigma_OH",
				"sigma_OO",   "swG_AlO",    "swG_OH",    "swS_AlO",
				"swS_OH" 
			]


# Training set
training_set	=	[ 	"gibSEowCNAlAltraj", "gibSEowtraj",        "al6OHtraj",     "al6H2Otraj", 
						      "al6AlAltraj",     "3m1traj",      "d2btod3traj",   "2m1tod2btraj", 
						       "d2tod1traj",   "32wOOtraj",        "32wOHtraj",  "NaOH30wPTtraj", 
						     "m1NaOHPTtraj",     "32wtraj",        "32wPTtraj", "aloh42h2obtraj", 
						       "m1NaOHtraj",    "2m10traj",         "2m1btraj",       "2m1ctraj", 
						    "aloh4scanAlOH",      "d1scan", "aloh4toaloh5scan",    "2m1tod3scan", 
						               "w2",          "w6",             "d1na",           "d3na", 
						          "aloh4na",           "w",              "Naw",           "Na5w", 
						             "NaOH",          "Na",            "wscan",     "al3oh122na", 
						       "al3o3oh6na",    "al3oh9na",          "gibbulk",      "boehmiteb", 
						           "gib001",   "gib001top",        "gib001bot",         "gib110", 
						           "gib825"
					]

products	=	[	"gibSEowCNAlAltraj", "gibSEowtraj",   "al6OHtraj",       "al6H2Otraj", 
				       	  "al6AlAltraj",     "3m1traj", "d2btod3traj",     "2m1tod2btraj", 
				          "d2tod1traj",    "32wOOtraj",   "32wOHtraj",    "NaOH30wPTtraj", 
				        "m1NaOHPTtraj",      "32wtraj",   "32wPTtraj",         "2m10traj", 
				            "2m1btraj",     "2m1ctraj",  "m1NaOHtraj",   "aloh42h2obtraj", 
				       "aloh4scanAlOH",        "wscan",      "d1scan", "aloh4toaloh5scan", 
				         "2m1tod3scan",         "d3na",        "d1na",               "w2", 
				                  "w6",         "Na5w",         "Naw",           "gib001", 
				              "gib825",       "gib110"
				]

# This came from ./data/Eref.dat
reference_data	=	[	53.34110,  -17.5500000000000,   67.59140000,   54.8697,
						39.29550,  137.9060000000000,    4.60300000,    0.7247,
						-4.76830,   -4.8418500000000,  -45.92000000,  -93.9900,
					   -24.87000,  -10.6600000000000,  -36.53000000,  -25.1700,
					    14.95700,  206.6600000000000,  -41.36000000, -128.3600,
					   162.92000, -101.9800000000000,   16.02900000,  135.0100,
					    70.98000,  -21.5088000000000,   91.83110000,  -92.7814,
					    28.70340,   24.6990079916871,    2.20880933,    6.1508,
					    17.69662,  224.6836382000000,  -14.64000000,   -9.7900,
					    63.36000,   49.8797000000000,  -36.11500000,  -10.8767,
					    49.04000,   66.5780000000000,  217.33000000,  152.3200,
					   227.86000,  262.4800000000000,  321.90000000,  -27.4000,
					   -86.90000, -204.8128000000000, -224.39676900,   98.4900,  
					   155.90000,  131.1100000000000,  146.29000000,  171.7000,  
					   115.50000,  160.5000000000000,  137.50000000,  170.5000 
					]

reference_data_keys	=	[	   "aloh5",    "aloh4w",          "d1",           "d3",
							      "d2",    "al2oh6",      "o2h3ts",       "d2ohts",
						     "d2ohts2",        "w2",          "w6",         "Na5w",
						         "Naw",   "aloh5na",        "d3na",         "d1na",
					      "aloh3wOHTS",     "aloh6",      "NaOH3A",      "NaOH10A",
					         "aloh3w3",   "aloh6na", "gibSE_h2oup", "gibSE_bothup",
					         "aloh3+w", "aloh3wfr4",       "aloh3",       "gib001",
					      "gibbulkOH2",    "gib825",       "gib85",       "gib100",
					          "gib105",    "gib110",     "aloh3w2",   "aloh3wnaoh",
					       "aloh3naoh",       "2m1",       "2m1na",       "d12naw",
					             "d2b",     "dtrib",    "al3o3oh6",     "al3oh122",
					        "al4o3oh9",   "al6oh18",    "al6oh182",          "OHw",
					            "OH4w",      "2u22",        "3p33",         "NSA5",
					            "NSA4",    "NSA128",      "NSA200",       "NSA300",
					            "NSA1",   "NSAb200",     "NSAb100",      "NSAb500"
						]

energy_reference	=	createParameters(keys = reference_data_keys, values = reference_data)

constraints_data	=	[	 1,  2,   2,   2,
							 1,  1,   8,  15,
							15, 15, 304, 304
						]

constraints_data_keys	=	[	  "h3oho", "o2h3ts",       "o2h3m",         "Nawc",
								  "h5o2m", "h5o2ts",  "aloh3wOHTS",        "d2ohm",
								"d2ohts2", "d2ohts", "gibSE_h2oup", "gibSE_bothup"
							]


counter_data	=	[	8,  9,  8, 17,
						2,  3,  5,  5,
					   12, 19, 19
					]

counter_data_keys	=	[		   "H3Ow",  "h5o2m", "h5o2ts",             "Na5w",
							  		 "Na",   "Nawc",    "Naw", "aloh4toaloh5scan",
					    	"2m1tod3scan", "d1scan"
						]

pbc_data		=	["NSA", "gib", "boehmite"]
trajs			=	[item for item in training_set if "traj" in item]
pbc_data_keys	=	[]

# suboptimial, but works
for item in training_set:
	for comp in pbc_data:
		if item.find(comp) != -1:
			pbc_data_keys.append(item)

pressure_data_keys		=	["gibbulk", "boehmiteb"]
pressure_data_values	=	{  "gibbulk" : [6657, 4512, 4963,  499, 350,  140],
							 "boehmiteb" : [ 110,  105,  112, -672, -10, 2706]
							}
# Exclusions
charge_exclusion	=	pbc_data_keys + [	 "aloh4w", "aloh3wOHTS",          "d2ohts2",  "aloh3w3",
							"aloh3w2",    "aloh6na", "aloh42h2o_wshell",    "2m1na",
							 "d12naw",        "d2b",         "al3o3oh6",  "al3oh12",
						   "al4o3oh9",    "al6oh18",         "al6oh182", "al3oh9na"
						]
charge_exclusion	=	charge_exclusion + [x for x in training_set if x.find("scan") != -1]
charge_exclusion	=	charge_exclusion + [x for x in training_set if x.find("traj") != -1]

xyz_exclusion		=	["aloh42h2o_wshell"]
xyz_exclusion		=	xyz_exclusion + [x for x in training_set if x.find("scan") != -1]
xyz_exclusion		=	xyz_exclusion + [x for x in training_set if x.find("traj") != -1]

dist_exclusion		=	pbc_data_keys + ["Na"]
dist_exclusion_data	=	[]
for item in training_set:
	for comp in dist_exclusion:
		if item.find(comp) != -1:
			dist_exclusion_data.append(item)

def clean_slate() -> None:
	os.system("rm -r *log *.frc *.q ener.dat")

def create_LAMMPS_Input(parameters: typing.Dict[str, str]) -> None:
	filename	=	parameters["filename"]
	with open(f"{filename}.inp", "w") as inputObject:
		# Header section
		inputObject.write("units		real\n")
		inputObject.write("boundary	{:<2s}\n".format(parameters["boundaries"]))
		inputObject.write("atom_style	full\n")
		inputObject.write("read_data	{}/{}-q0.data\n".format(parameters["datadir"], parameters["filename"]))
		inputObject.write("\n"*1)

		# Charge section
		inputObject.write("set type 1 charge  1.296\n")
		inputObject.write("set type 2 charge -0.898\n")
		inputObject.write("set type 3 charge  0.449\n")
		inputObject.write("set type 4 charge  1.000\n")
		inputObject.write("\n"*1)

		# Pair style section
		inputObject.write("pair_style hybrid/overlay {} {} sw lj/smooth/linear 5.0 gauss/cut 5.0 coord/gauss/cut 5.0 \n".format(parameters["coulps"], parameters["coulcut"]))
		inputObject.write("pair_coeff * * {}\n".format(parameters["coulps"]))
		inputObject.write("pair_coeff * * {}\n".format(parameters["coulps"]))
		inputObject.write("pair_coeff * * lj/smooth/linear 0 0 0\n")
		inputObject.write("pair_coeff * * sw param.sw Al O H NULL\n")
		inputObject.write("pair_coeff * * gauss/cut 0 0 1 0\n")
		inputObject.write("pair_coeff 2 2 lj/smooth/linear {}e-2 {}\n".format(parameters["eps_OO"], parameters["sigma_OO"]))
		inputObject.write("pair_coeff 1 2 lj/smooth/linear {}e-2 {}\n".format(parameters["eps_AlO"], parameters["sigma_AlO"]))
		inputObject.write("pair_coeff 2 3 lj/smooth/linear {}e-2 {}\n".format(parameters["eps_OH"], parameters["sigma_OH"]))
		inputObject.write("pair_coeff 2 3 gauss/cut {} {} {} 2.0\n".format(parameters["gaussH_OH"], parameters["gaussR_OH"], parameters["gaussW_OH"]))
		inputObject.write("pair_coeff 1 2 coord/gauss/cut {} {} {} 6.0\n".format(parameters["gaussH_AlO"], parameters["gaussR_AlO"], parameters["gaussW_AlO"]))
		inputObject.write("pair_coeff 1 1 gauss/cut {} {} {} 5.0\n".format(parameters["gaussH_Al"], parameters["gaussR_Al"], parameters["gaussW_Al"]))
		inputObject.write("pair_coeff 2 4 lj/smooth/linear {}e-2 {}\n".format(parameters["eps_NaO"], parameters["sigma_NaO"]))
		inputObject.write("\n"*1)

		# kspace section
		inputObject.write("kspace_style {}\n".format(parameters["kspace"]))
		inputObject.write("\n"*1)

		# thermostat
		inputObject.write("thermo_style custom {}\n".format(parameters["thermostyle"]))
		inputObject.write("thermo 1\n")
		inputObject.write("\n"*1)

		# charge section
		inputObject.write("fix 1 all qeq/point 1 8.5 1.0e-6 200 param.qeq\n")
		inputObject.write("\n"*1)

		# define required groups
		inputObject.write("group constr {}\n".format(parameters["constraint"]))
		inputObject.write("group counter {}\n".format(parameters["counter"]))
		inputObject.write("group fixed union constr counter\n")
		inputObject.write("group todump subtract all counter\n")
		inputObject.write("fix 2 counter setforce 0.0 0.0 0.0\n")
		inputObject.write("\n"*1)

		# check for conditional arguments
		if parameters["qdump"] == "T":
			inputObject.write("compute 1 all property/atom q\n")
			inputObject.write("dump 1 todump custom 1 {}.q id c_1\n".format(parameters["filename"]))
			inputObject.write("dump_modify 1 sort id\n")
			inputObject.write("\n"*1)

		if parameters["traj"] == "T":
			inputObject.write("dump 3 todump custom 1 {}.FF.frc id fx fy fz\n".format(parameters["filename"]))
			inputObject.write("dump_modify 3 sort id\n")
			inputObject.write("rerun {}/{}.dump dump x y z box yes\n".format(parameters["datadir"], parameters["filename"]))
			inputObject.write("\n"*1)
		elif (parameters["scan"] == "T"):
			inputObject.write("dump 3 todump custom 1 {}.FF.frc id fx fy fz\n".format(parameters["filename"]))
			inputObject.write("dump_modify 3 sort id\n")
			inputObject.write("rerun {}/{}.xyz dump x y z box no format xyz\n".format(parameters["outdir"], parameters["filename"]))
			inputObject.write("\n"*1)
		else:
			inputObject.write("run 0\n");
			inputObject.write("print \"{}: $(pe) $(fnorm) {}\" append ener.dat screen no\n".format(parameters["filename"], parameters["supplements"]))
			inputObject.write("\n"*1)

	with open("param.qeq", "w") as paramObject:
		paramObject.write("1 {} {} {}e-2 0 0.0\n".format(parameters["chi_Al"], parameters["eta_Al"], parameters["gamma_Al"]))
		paramObject.write("2 {} {} {}e-2 0 0.0\n".format(parameters["chi_O"], parameters["eta_O"], parameters["gamma_O"]))
		paramObject.write("3 {} {} {}e-2 0 0.0\n".format(parameters["chi_H"], parameters["eta_H"], parameters["gamma_H"]))
		paramObject.write("4 {} {} {}e-2 0 0.0\n".format(parameters["chi_Na"], parameters["eta_Na"], parameters["gamma_Na"]))

	with open("param.sw", "w") as swObject:
		swObject.write("#j  i   k  eps_ijk   sigma_ij  a_ij  lambda_ijk  gamma_ij0.7617  cthet      A  B  p  q  tol\n")
		swObject.write("O   H   H  {}    {}e-2   1     1           {}e-2       {}e-2  0  1  0  0  0.01\n".format(parameters["eps_HOH"],   parameters["swS_OH"],  parameters["swG_OH"],  parameters["ct_HOH"]))
		swObject.write("H   O   O  0     {}e-2   1     0           {}e-2       0      0  1  0  0  0.01\n".format(parameters["swS_OH"],   parameters["swG_OH"]))
		swObject.write("O   Al  H  {}e2  {}e-2   1     1           {}e-2       {}e-2  0  1  0  0  0.01\n".format(parameters["eps_AlOH"], parameters["swS_AlO"], parameters["swG_AlO"], parameters["ct_AlOH"]))
		swObject.write("Al  O   O  0     {}e-2   1     1           {}e-2       0      0  1  0  0  0.01\n".format(parameters["swS_AlO"],  parameters["swG_AlO"]))
		swObject.write("O   Al  Al 0     {}e-2   1     1           {}e-2       0      0  1  0  0  0.01\n".format(parameters["swS_AlO"],  parameters["swG_AlO"]))
		swObject.write("Al  Al  Al 0     0       0     0           0           0      0  0  0  0  0.01\n")
		swObject.write("Al  Al  O  0     0       0     0           0           0      0  0  0  0  0.01\n")
		swObject.write("Al  Al  H  0     0       0     0           0           0      0  0  0  0  0.01\n")
		swObject.write("Al  O   Al 0     0       0     0           0           0      0  0  0  0  0.01\n")
		swObject.write("Al  O   H  0     0       0     0           0           0      0  0  0  0  0.01\n")
		swObject.write("Al  H   Al 0     0       0     0           0           0      0  0  0  0  0.01\n")
		swObject.write("Al  H   O  0     0       0     0           0           0      0  0  0  0  0.01\n")
		swObject.write("Al  H   H  0     0       0     0           0           0      0  0  0  0  0.01\n")
		swObject.write("O   Al  O  0     0       0     0           0           0      0  0  0  0  0.01\n")
		swObject.write("O   O   Al 0     0       0     0           0           0      0  0  0  0  0.01\n")
		swObject.write("O   O   O  0     0       0     0           0           0      0  0  0  0  0.01\n")
		swObject.write("O   O   H  0     0       0     0           0           0      0  0  0  0  0.01\n")
		swObject.write("O   H   Al 0     0       0     0           0           0      0  0  0  0  0.01\n")
		swObject.write("O   H   O  0     0       0     0           0           0      0  0  0  0  0.01\n")
		swObject.write("H   O   H  0     0       0     0           0           0      0  0  0  0  0.01\n")
		swObject.write("H   Al  Al 0     0       0     0           0           0      0  0  0  0  0.01\n")
		swObject.write("H   Al  H  0     0       0     0           0           0      0  0  0  0  0.01\n")
		swObject.write("H   Al  O  0     0       0     0           0           0      0  0  0  0  0.01\n")
		swObject.write("H   O   Al 0     0       0     0           0           0      0  0  0  0  0.01\n")
		swObject.write("H   H   Al 0     0       0     0           0           0      0  0  0  0  0.01\n")
		swObject.write("H   H   O  0     0       0     0           0           0      0  0  0  0  0.01\n")
		swObject.write("H   H   H  0     0       0     0           0           0      0  0  0  0  0.01\n")

def run_lammps(*args):
	parsed_values	=	[round(arg, 6) for arg in args[0]]

	# Need headers for error calculations
	trajs			=	[item for item in training_set if "traj" in item]
	headers			=	trajs + products + ["q", "E", "f", "fm", "p", "TOT"]
	errors			=	pandas.DataFrame(columns = headers)
	pbc_info		=	pbc_data + trajs

	for system in training_set:
		# Input parameters
		parameters 		=	createParameters(keys = keys, values = parsed_values)
		constraints 	=	createParameters(keys = constraints_data_keys, values = constraints_data)
		counters 		=	createParameters(keys = counter_data_keys, values = counter_data)

		parameters["supplements"] = ""

		if any(info in system for info in pbc_info):
			boundaries	=	"p p p"
			coulps		=	"coul/long"
			coulcut		=	"8.5"
			kspace		=	"ewald 1.0e-5"
			periodic	=	"XYZ"
			parameters["boundaries"]	=	boundaries
			parameters["coulps"]		=	coulps
			parameters["coulcut"]		=	coulcut
			parameters["kspace"]		=	kspace
			parameters["periodic"]		=	periodic
		else:
			boundaries	=	"f f f"
			coulps		=	"coul/cut"
			coulcut		=	"15.0"
			kspace		=	"none"
			periodic	=	"NONE"
			parameters["boundaries"]	=	boundaries
			parameters["coulps"]		=	coulps
			parameters["coulcut"]		=	coulcut
			parameters["kspace"]		=	kspace
			parameters["periodic"]		=	periodic

		if system.find("scan") != -1:
			spe 		=	"T"
			traj 		=	"F"
			outdir		=	"./data/scan"
			datadir		=	"./data/data"
			thermostyle	=	"pe"
			scan 		=	"T"

			# set the parameters
			parameters["spe"]			=	spe
			parameters["traj"]			=	traj
			parameters["outdir"]		=	outdir
			parameters["datadir"]		=	datadir
			parameters["thermostyle"]	=	thermostyle
			parameters["scan"]			=	scan
		if system.find("traj") != -1:
			spe 		=	"T"
			traj 		=	"T"
			datadir		=	"./data/traj"
			thermostyle	=	"pe"
			scan 		=	"F"
			boundaries	=	"p p p"

			# set the parameters
			parameters["spe"]			=	spe
			parameters["traj"]			=	traj
			parameters["datadir"]		=	datadir
			parameters["thermostyle"]	=	thermostyle
			parameters["scan"]			=	scan
			parameters["boundaries"]	=	boundaries
		if system.find("traj") == -1 and system.find("scan") == -1:
			scan		=	"F"
			spe 		=	"F"
			traj 		=	"F"
			datadir		=	"./data/data"
			thermostyle	=	"pe fnorm"
			parameters["supplements"] = ""
			if system in pressure_data_keys:
				thermostyle	=	"pe pxx pyy pzz pxy pxz pyz"
				supplements	=	"$(pxx) $(pyy) $(pzz) $(pxy) $(pxz) $(pyz)"
				parameters["supplements"]	=	supplements

			# set the parameters
			parameters["spe"]			=	spe
			parameters["traj"]			=	traj
			parameters["datadir"]		=	datadir
			parameters["thermostyle"]	=	thermostyle
			parameters["scan"]			=	scan

		if system in constraints_data_keys:
			constraint 	=	"id {}".format(constraints[system])
		else:
			constraint 	=	"empty"

		if system in counter_data_keys:
			counter 	=	"id {}".format(counters[system])
		else:
			counter 	=	"empty"

		if system in charge_exclusion:
			parameters["qdump"]	=	"F"
		else:
			parameters["qdump"]	=	"T"

		parameters["constraint"]	=	constraint
		parameters["counter"]		=	counter
		parameters["filename"]		=	system

		# create the LAMMPS input file
		create_LAMMPS_Input(parameters)

def param_optimizer(*args) -> typing.List[float]:
	# clear out old files
	run_lammps(*args)
	clean_slate()
	commands	=	"bash test.sh"
	os.system(commands)

	# Having run the input files
	# Now we need to parse the output files

	# Parse the ener.dat file first and create a dict
	ener_dat		=	{}
	with open("ener.dat") as enerObject:
		for line in enerObject:
			line 	=	line.split()
			ener_dat[line[0].strip(":")]	=	[float(x) for x in line[1:]]

	error_fmax		=	numpy.mean(numpy.array([ener_dat[key][1]**2 for key in ener_dat]))
	energy_ener_f	=	[ener_dat[key][0] for key in ener_dat]

	# Create containers to hold error information
	error_charge	=	{}
	error_energy	=	{}
	error_force		=	{}
	error_pressure	=	{}

	for system in training_set:
		# calculate differences in charges
		if system not in charge_exclusion:
			xyz_file	=	f"./data/xyz/{system}.xyz"
			esp_file	=	f"./data/esp/{system}.esp"
			charge_file	=	f"{system}.q"
			with open(esp_file) as espObject, open(charge_file) as chargeObject:
				esp_data				=	[float(data.split()[-1]) for data in espObject.readlines()[2:]]
				charge_data				=	[float(data.split()[-1]) for data in chargeObject.readlines()[9:]]
				error_q_file			=	[(esp_value - charge_value) for (esp_value, charge_value) in zip(esp_data, charge_data)]
				error_charge[system]	=	[x for x in error_q_file]
		
		# First process the scans
		# calculate differences in energies
		if system.find("scan") != -1:
			energy_file		=	f"./data/scan/{system}.ener"
			log_file		=	f"./{system}.log"
			with open(energy_file) as energyObject, open(log_file) as logObject:
				energy_ref				=	[float(data.split()[-1]) for data in energyObject.readlines()]
				nlines					=	len(energy_ref) + 3
				_ff_energy				=	[data for data in logObject.readlines()[-1 * nlines:]][:-3]
				ff_energy 				=	numpy.array([float(data.split()[0]) for data in _ff_energy])
				ff_energy_s				=	ff_energy - min(ff_energy)
				error_e_file			=	[abs(ff_value - ref_value) for (ff_value, ref_value) in zip(ff_energy_s, energy_ref)]
				error_energy[system]	=	numpy.mean(numpy.array(error_e_file))
		
		# calculate differences in forces
		if system.find("scan") != -1:
			force_file	=	f"./data/scan/{system}.frc"
			ff_file		=	f"./{system}.FF.frc"
			with open(force_file) as forceObject, open(ff_file) as ffObject:
				_ref_data 				=	[data.split()[1:] for data in forceObject.readlines() if len(data.split()) == 4]
				ref_data 				=	[float(value) for force in _ref_data for value in force]
				_ff_force				=	[data.split()[1:] for data in ffObject.readlines() if len(data.split()) == 4 and data.find("ITEM") == -1]
				ff_force				=	[float(value) for force in _ff_force for value in force]
				error_f_file			=	[(ff_value - ref_value)**2 for (ff_value, ref_value) in zip(ff_force, ref_data)]
				error_f_file			=	numpy.array(error_f_file)
				error_f_file			=	error_f_file.reshape(-1, 3)	# reshapes the array in x, y, z
				error_force[system]		=	numpy.mean(numpy.sqrt(numpy.sum(error_f_file, axis = 1)))	# Takes the sum along the rows

		# Now process trajectories
		if system.find("traj") != -1:
			energy_file		=	f"./data/traj/{system}.ener"
			log_file		=	f"./{system}.log"
			with open(energy_file) as energyObject, open(log_file) as logObject:
				energy_ref				=	numpy.array([float(data.split()[-1]) for data in energyObject.readlines()])
				energy_ref_s			=	energy_ref - min(energy_ref)
				nlines					=	len(energy_ref) + 3
				_ff_energy				=	[data for data in logObject.readlines()[-1 * nlines:]][:-3]
				ff_energy 				=	numpy.array([float(data.split()[0]) for data in _ff_energy])
				ff_energy_s				=	ff_energy - min(ff_energy)
				error_e_file			=	[abs(ff_value - ref_value) for (ff_value, ref_value) in zip(ff_energy_s, energy_ref_s)]
				error_energy[system]	=	numpy.mean(numpy.array(error_e_file))

		if system.find("32wtraj") != -1:
			water_file		=	f"./data/traj/{system}.frc"
			ff_file			=	f"./{system}.FF.frc"
			with open(water_file) as waterObject, open(ff_file) as ffObject:
				_ref_data 				=	[data.split() for data in waterObject.readlines()]
				ref_data				=	[float(value) for force in _ref_data for value in force]
				_ff_force				=	[data.split()[1:] for data in ffObject.readlines() if len(data.split()) == 4 and data.find("ITEM") == -1]
				ff_force				=	[float(value) for force in _ff_force for value in force]
				error_f_file			=	[(ff_value - ref_value)**2 for (ff_value, ref_value) in zip(ff_force, ref_data)]
				error_f_file			=	numpy.array(error_f_file)
				error_f_file			=	error_f_file.reshape(-1, 3)	# reshapes the array in x, y, z
				error_force[system]		=	numpy.mean(numpy.sqrt(numpy.sum(error_f_file, axis = 1)))	# Takes the sum along the rows

		if system.find("traj") == -1 and system.find("scan") == -1:
			if system in pressure_data_keys:
				ff_data					=	ener_dat[system][2:]
				ref_data				=	pressure_data_values[system]
				error_p_file			=	[abs(ff_value - ref_value) for (ff_value, ref_value) in zip(ff_data, ref_data)]
				error_pressure[system]	=	numpy.mean(numpy.array(error_p_file))

	# Now take care of reactions
	reaction_energies	=	[	10.0 * error_energy["gibSEowCNAlAltraj"],
								0.0 * error_energy["gibSEowtraj"],
								0.0 * error_energy["al6OHtraj"],
								0.0 * error_energy["al6H2Otraj"],
								0.0 * error_energy["al6AlAltraj"],
								0.0 * error_energy["3m1traj"],
								0.0 * error_energy["d2btod3traj"],
								0.0 * error_energy["2m1tod2btraj"],
								5.0 * error_energy["d2tod1traj"],
								0.0 * error_energy["32wOOtraj"],
								5.0 * error_energy["32wOHtraj"],
								5.0 * error_energy["NaOH30wPTtraj"],
								5.0 * error_energy["m1NaOHPTtraj"],
								5.0 * error_energy["32wtraj"],
								0.0 * error_energy["32wPTtraj"],
								0.0 * error_energy["2m10traj"],
								0.0 * error_energy["2m1btraj"],
								0.0 * error_energy["2m1ctraj"],
								00./133 * error_energy["m1NaOHtraj"],
								4.0 * error_energy["aloh42h2obtraj"],
								0.0 * error_energy["aloh4scanAlOH"],
								0.0 * error_energy["wscan"],
								0.0 * error_energy["d1scan"],
								2.0 * error_energy["aloh4toaloh5scan"],
								2.0 * error_energy["2m1tod3scan"],
								1.00 * (1 * (ener_dat["d3na"][0] 	- (2 * ener_dat["aloh4na"][0])) - (energy_reference["d3na"])),
								5.00 * (1 * (ener_dat["w2"][0]   	- (2 * ener_dat["w"][0])) 		- (energy_reference["w2"])),
								5.00 * (1 * (ener_dat["w6"][0]   	- (6 * ener_dat["w"][0]))		- (energy_reference["w6"])),
								1.00 * (1 * (ener_dat["d1na"][0] 	+ ener_dat["w"][0] 			 	- (2 * ener_dat["aloh4na"][0])) - (energy_reference["d1na"])),
								0.25 * (1 * (ener_dat["Na5w"][0]   	- (5 * ener_dat["w"][0]		 	+ ener_dat["Na"][0]))			- (energy_reference["Na5w"])),
								1.00 * (1 * (ener_dat["Naw"][0]		- (ener_dat["w"][0]				+ ener_dat["Na"][0]))			- (energy_reference["Naw"])),
								0.50 * (1 * (ener_dat["gib001"][0] 	- (ener_dat["gib001top"][0]		+ ener_dat["gib001bot"][0]))	- (energy_reference["gib001"])),
								0.50 * (1 * (ener_dat["gib825"][0] 	- (ener_dat["gibbulk"][0]))		- (energy_reference["gib825"])),
								0.25 * (1 * (ener_dat["gib110"][0] 	- (ener_dat["gibbulk"][0]))		- (energy_reference["gib110"]))
							]

	# Now compute error terms
	fmax_error		=	error_fmax
	charge_error	=	numpy.sqrt(numpy.mean(numpy.array([charge**2 for (key,value) in error_charge.items() for charge in value])))
	reactions_error	=	numpy.sqrt(numpy.mean(numpy.array([energy**2 for energy in reaction_energies])))
	force_error		=	numpy.mean(numpy.array([value**2 for (key,value) in error_force.items()]))
	pressure_error	=	numpy.mean(numpy.array([value**2 for (key,value) in error_pressure.items()]))

	# Now we need to compute the final error
	final_error		=	100 * ((0.0005 * fmax_error) + (300 * charge_error) + (0.005 * force_error) + (1 * reactions_error) + (2.5e-6 * pressure_error)) / (300 + 1 + 0.05 + 0.005 + 2.5e-6)
	print("{:10.3f}".format(final_error))
	return final_error

def main() -> None:
	clean_slate()	# remove files from previous runs
	margin		=	float(sys.argv[1])	# get the margin for fitting from user

	first_minimization	=	scipy.optimize.minimize(param_optimizer, paramVector_2, method='Nelder-Mead',options={'adaptive': True,'maxiter': 200000})
	print(first_minimization.x)

	# first do a global minimization 
	# optimizer			=	nlopt.opt(nlopt.G_MLSL_LDS, paramVector_2.size)	
	# local_optimizer		=	nlopt.opt(nlopt.LN_SBPLX, paramVector_2.size)
	# optimizer.set_local_optimizer(local_optimizer)
	# local_optimizer.set_xtol_rel(1e-3)

	# minParams	=	first_minimization - (margin * abs(first_minimization))	# lower bound for parameters
	# maxParams	=	first_minimization + (margin * abs(first_minimization))	# upper bound for parameters

	# optimizer.set_lower_bounds(minParams)
	# optimizer.set_upper_bounds(maxParams)
	# optimizer.set_min_objective(param_optimizer)

	# # Now perform the optimization
	# optimized_values	=	optimizer.optimize(first_minimization)
	# print(optimized_values)

if __name__ == '__main__':
	main()