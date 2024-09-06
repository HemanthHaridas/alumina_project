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
import pprint
import pandas

# helper functions
def createParameters(keys: typing.List[str], values: typing.List[float]) -> typing.Dict[str, str]:
	currentSet	=	{}
	for key, value in zip(keys, values):
		currentSet[key]	=	value
	return currentSet

def clean_slate() -> None:
	os.system("rm -r *log *.frc *.q ener.dat")

# Initial parameters

# paramVector_1	=	[ 	15.0369,  208.1295,  17.8096,  673.5521, 
# 					   -31.1295,  26.8294,   16.0370, 1779.9059,    
# 					   	 0.4126,   6.6646,  319.3682,  548.1608, 
# 					   501.2680, 341.1614, 	  1.1930,   31.3144, 
# 					   180.6524,   7.6038,	  0.8350,   -5.0000,  
# 					     2.8207,   1.5931,    0.2075,    0.0539,	  
# 					     2.0955,   3.4559,    1.2418,	80.0000,
# 					     4.0000,   2.5000
# 					 ]

# paramVector_1	=	[	 15.5624,  222.9389,   14.3795,  670.8081,  
# 						-36.3151,   27.2211,   16.2616, 1520.3276,    
# 						  0.4288,    6.2526,  314.1858,  517.4529,  
# 						479.1339,  341.8465,    1.2525,   33.1349,  
# 						203.2376,    7.9525,    0.8928,   -0.8250,    
# 						  2.8543,    1.6455,    0.1492,    0.0525,    
# 						  2.0901,    3.4540,    1.2502,   78.3163
# 					]

# optimized => do not touch
# paramVector_1	=	[ 1.55772786e+01,  2.22966571e+02,  1.44545379e+01,  6.70828813e+02,
# 					 -3.62924396e+01,  2.72344683e+01,  1.66724846e+01,  1.50608440e+03,
# 					  4.30608159e-01,  6.25484589e+00,  3.14210981e+02,  5.17526901e+02,
# 					  4.79767886e+02,  3.41836517e+02,  1.26907355e+00,  3.35004878e+01,
# 					  2.00410021e+02,  7.95711878e+00,  8.92319805e-01, -8.42976078e-01,
# 					  2.85423777e+00,  1.64603409e+00,  1.49481528e-01,  5.01216129e-02,
# 					  2.09012624e+00,  3.45336336e+00,  1.25019399e+00,  8.11080263e+01
# 					  ]

# paramVector_1	=	[ 2.37522275e+00,  2.83499821e+02,  1.60781592e+01,  6.45059775e+02,
# 					 -3.54790008e+01,  3.65865953e+01,  1.57917272e+01,  1.63425829e+03,
# 					  2.93063968e-01,  1.28579425e+01,  3.46794498e+02,  4.07990713e+02,
# 					  5.20913943e+02,  2.74243335e+02,  1.75673970e+00,  3.11509855e+01,
# 					  2.10939894e+02,  7.47066436e+00,  5.87807438e-01, -5.17718429e+00,
# 					  2.84513523e+00,  4.38559152e-01,  2.27039382e+00,  6.73985811e-01,
# 					  2.06128716e+00,  3.58349563e+00,  1.18766529e+00,  5.62999094e+01,
# 					  3.65919776e+00,  1.59260000e+00]

# paramVector_1	=	[ 2.35593262e+00,  2.86213049e+02,  1.59381254e+01,  6.45788826e+02,
# 					 -3.53644361e+01,  1.30000000e+02,  1.59489926e+01,  1.63753963e+03,
# 					  2.87800684e-01,  1.26859545e+01,  3.49223126e+02,  4.10299314e+02,
# 					  5.18865607e+02,  2.73891442e+02,  1.77105981e+00,  3.14710296e+01,
# 					  2.13130344e+02,  7.53077862e+00,  5.87572565e-01, -4.99456675e+00,
# 					  2.83276463e+00,  4.40040341e-01,  2.47411032e+00,  6.69120587e-01,
# 					  2.05963469e+00,  3.58889896e+00,  1.18622962e+00,  5.72679585e+01,
# 					  4.09049158e+00,  1.59168235e+00]

# paramVector_1	=	[ 2.47007393e+00,  3.02442125e+02,  1.48235040e+01,  6.43492278e+02,
# 					 -4.07260124e+01,  1.20104265e+02,  1.66955474e+01,  1.68149171e+03,
# 					  4.66215314e-01,  1.24064884e+01,  3.71222949e+02,  4.31710129e+02,
# 					  5.28830600e+02,  2.38268348e+02,  1.56222346e+00,  3.42688692e+01,
# 					  2.15743582e+02,  7.17561797e+00,  5.60298751e-01, -4.58997434e+00,
# 					  2.46025062e+00,  4.40375567e-01,  2.43347445e+00,  5.97631276e-01,
# 					  1.90849294e+00,  3.47986114e+00,  1.19079358e+00,  5.20561576e+01,
# 					  3.88816326e+00,  1.66370914e+00]	=> optimized

# paramVector_1	=	[ 2.47007393e+00,  3.02442125e+02,  1.48235040e+01,  6.43492278e+02,
# 					 -4.07260124e+01,  1.20104265e+02,  1.66955474e+01,  1.68149171e+03,
# 					  4.66215314e-01,  1.24064884e+01,  3.71222949e+02,  4.31710129e+02,
# 					  5.28830600e+02,  2.38268348e+02,  1.56222346e+00,  3.42688692e+01,
# 					  2.15743582e+02,  7.17561797e+00,  5.60298751e-01, -4.58997434e+00,
# 					  2.46025062e+00,  4.40375567e-01,  1.83347445e+00,  5.97631276e-01,
# 					  1.70849294e+00,  3.47986114e+00,  1.19079358e+00,  5.20561576e+01,
# 					  3.88816326e+00,  1.66370914e+00]

# paramVector_1	=	[ 2.40659866e+00,  2.64380071e+02,  1.22519437e+01,  6.04623561e+02,
# 					 -3.11190237e+01,  2.01872948e+01,  1.67621567e+01,  1.82383660e+03,
# 					  6.17822104e-01,  1.22971105e+01,  3.53813871e+02,  4.38985907e+02,
# 					  5.13462554e+02,  2.40780694e+02,  1.25915621e+00,  3.02278746e+01,
# 					  1.99022240e+02,  7.13122849e+00,  5.35183491e-01, -6.49808073e+00,
# 					  2.62358725e+00,  4.43621297e-01,  1.98976420e+00,  3.20786216e-01,
# 					  2.03926618e+00,  3.35711041e+00,  1.18555157e+00,  5.73592821e+01,
# 					  4.00000000e+00,  2.50000000e+00]

paramVector_1	=		[ 2.40484897e+00,  2.64146788e+02,  1.19112656e+01,  6.04616383e+02,
						 -3.16753520e+01,  2.02726990e+01,  1.65359835e+01,  1.40455673e+03,
						  6.02076817e-01,  1.22828245e+01,  3.54359909e+02,  4.39208316e+02,
						  5.14515604e+02,  2.41018010e+02,  1.34159245e+00,  3.31221119e+01,
						  2.08062458e+02,  7.13861574e+00,  2.03952700e+00,  3.36328156e+00,
						  1.18540049e+00,  5.83529540e+01, -5.60690692e+00,  1.99787464e+00,  
						  0.30565977e+00,  4.09454150e+00,  2.94716235e+00, -0.15569286e+00,
						  4.03310597e+00,  0.48100599e+00,  3.03012667e+00,  6.23974649e+00] 

# paramVector_1_gauss	=	[ 2.35847225e+00  2.64116698e+02  1.18835569e+01  6.04840002e+02
# 						 -3.29340264e+01  2.01870695e+01  1.64251333e+01  1.41838502e+03
# 						  5.95656189e-01  1.23666349e+01  3.55137842e+02  4.38435024e+02
# 						  5.11348125e+02  2.40731442e+02  1.35172901e+00  3.32493349e+01
# 						  2.09026822e+02  7.23278378e+00  2.03829674e+00  3.36912964e+00
# 						  1.18589631e+00  5.56815940e+01 -8.79307551e+00  1.98240411e+00
# 						  3.40742332e-01  3.11103498e+00  2.43509009e+00 -3.52953330e-01
# 						  3.32232881e+00  5.15127932e-01  3.45622907e+00  5.07236623e+00]

# paramVector_1_optimized	=	[ 2.40484897e+00,  2.64146788e+02,  1.19112656e+01,  6.04616383e+02,
# 							 -3.16753520e+01,  2.02726990e+01,  1.65359835e+01,  1.40455673e+03,
# 							  6.02076817e-01,  1.22828245e+01,  3.54359909e+02,  4.39208316e+02,
# 							  5.14515604e+02,  2.41018010e+02,  1.34159245e+00,  3.31221119e+01,
# 							  2.08062458e+02,  7.13861574e+00,  2.03952700e+00,  3.36328156e+00,
# 							  1.18540049e+00,  5.83529540e+01  
						  	# ]
# pair_coeff 1 1 coord/gauss/cut 	-0.341131 3.091155 0.524181 3.286606 4.987455
# pair_coeff 1 2 coord/gauss/cut 	-8.847336 1.980663 0.339094 3.13086	2.450685

# pair_coeff 1 1 coord/gauss/cut 	-0.476172 3.952414 0.354829 2.598755 5.685127
# pair_coeff 1 2 coord/gauss/cut 	-6.615408 2.117999 0.417449 3.746032	2.600773

# paramVector_1 	=			[-8.83440579e+00,  1.98137455e+00,	3.39124401e-01,  4.08095572e+00,  2.44216023e+00, 
# 							 -3.56182652e-01,  3.23345452e+00,  5.13116484e-01,  3.40826822e+00,  5.02113923e+00 
# 							] # => tetra-coordinated

# paramVector_1 	=	[-6.61538554,  2.11799984,  0.41744866,  3.74601097,  2.60076542, 
# 					 -0.47618264,  3.95239731,  0.35483022,  2.59880852,  5.68511993]

# paramVector_1 	=	[
# 					-5.60690692,  1.99787464,  0.30565977,  4.09454150,   2.94716235 
# 					-0.15569286,  4.03310597,  0.48100599,  3.03012667,   6.23974649
# 					]

# paramVector_1 	=	[
# 					-5.60690692,  1.99787464,  0.30565977,  6.09454150,   2.94716235, 
# 					-0.15569286,  4.03310597,  0.48100599,  3.03012667,   6.23974649
# 					]
# paramVector_1 	=			[-6.49864123,  1.98975484,  0.32077941,  6.0000000,   2.16333265]
# paramVector_1	=	[ 2.3312,  269.0241,   13.8019,  654.2201,  
# 					-33.1663,   38.8260,    9.2827, 1499.5983,    
# 				 	  0.1709,   10.1862,  349.2552,  462.5450,  
# 				 	621.1912,  272.1089,    0.6145,   17.5571,  
# 				 	214.9270,    8.3138,    1.8561,   -5.7342,    
# 				 	  2.1478,    3.5286,    0.4307,    0.2493,    
# 				 	  2.5321,    2.9679,    1.6483,   55.1057
# 					  ]

keys 				=	[	"chi_Al",     	"chi_H",      	"chi_Na",    	"chi_O",
						"ct_AlOH",    	"eps_AlO",   	"eps_AlOH",		"eps_HOH",    
						"eps_NaO",    	"eps_OH",    	"eta_Al",     	"eta_H",      
						"eta_Na",    	"eta_O",   		"gamma_Al",   	"gamma_H",    
						"gamma_Na",  	"gamma_O",		"sigma_AlO",  	"sigma_NaO", 	
						"sigma_OH",		"swG_AlO",		"gaussH_AlO", 	"gaussR_AlO", 	
						"gaussW_AlO", 	"coord_AlO",	"radius_AlO",	"gaussH_Al",  
						"gaussR_Al",	"gaussW_Al",  	"coord_Al",		"radius_Al"]

# keys =	["gaussH_AlO", "gaussR_AlO", 	"gaussW_AlO", "coord_AlO",	"radius_AlO",
# 		 "gaussH_Al",  "gaussR_Al",		"gaussW_Al",  "coord_Al",	"radius_Al"]

# Original data => might need this again

# paramVector_1	=	[
# 					-33.8017,   25.1659,   16.1241, 1788.7428,    
# 					  0.4384,    5.9865,    0.9469,    3.2604,    
# 					  0.1802,    1.9375,    3.4286,    0.9027,   
# 					 90.8703
# 					 ]

# keys	=	[
# 				"ct_AlOH",    	"eps_AlO",   	"eps_AlOH", 	"eps_HOH",    
# 				"eps_NaO",    	"eps_OH",   	"gaussH_Al",	"gaussR_Al",	
# 				"gaussW_Al",	"sigma_AlO",  	"sigma_NaO", 	"sigma_OH",		
# 				"swG_AlO"
# 			]

paramVector_2	=	numpy.array(paramVector_1)


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

# Maxime - corrected counter_data 
counter_data	=	[	[8,9],   [8,9],   [8,9], 
					  [17,18],   [2,3],   [5,6], 
					    [5,6], [12,13], [19,20], 
					  [19,20]
					]

counter_data_keys	=	[ 				"H3Ow", 	       "h5o2m", 	  "h5o2ts", 
						  				"Na5w", 		 	  "Na", 		"Nawc", 
						  				 "Naw", "aloh4toaloh5scan",	 "2m1tod3scan", 	  
						  			  "d1scan"
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
pressure_data_values	=	{  "gibbulk" : [6657, 4512, 4963,  499, 350, 1407],
							 "boehmiteb" : [ 110,  105,  112, -672, -10, 2706]
							}
#Maxime: replaced charge exclusion with charge inclusion
charge_do = [ "aloh4na", "d1na", "d3na", "Na5w", "NaOH", "Na", "Naw", "w2", "w6", "w" ]
# Exclusions
#charge_exclusion	=	pbc_data_keys + [	 "aloh4w", "aloh3wOHTS",          "d2ohts2",  "aloh3w3",
#							"aloh3w2",    "aloh6na", "aloh42h2o_wshell",    "2m1na",
#							 "d12naw",        "d2b",         "al3o3oh6",  "al3oh12",
#						   "al4o3oh9",    "al6oh18",         "al6oh182", "al3oh9na"
#						]
#charge_exclusion	=	charge_exclusion + [x for x in training_set if x.find("scan") != -1]
#charge_exclusion	=	charge_exclusion + [x for x in training_set if x.find("traj") != -1]

#Maxime: not needed
#xyz_exclusion		=	["aloh42h2o_wshell"]
#xyz_exclusion		=	xyz_exclusion + [x for x in training_set if x.find("scan") != -1]
#xyz_exclusion		=	xyz_exclusion + [x for x in training_set if x.find("traj") != -1]
#
#dist_exclusion		=	pbc_data_keys + ["Na"]
#dist_exclusion_data	=	[]
#for item in training_set:
#	for comp in dist_exclusion:
#		if item.find(comp) != -1:
#			dist_exclusion_data.append(item)

def create_LAMMPS_Input(parameters: typing.Dict[str, str]) -> None:
	filename	=	parameters["filename"]

	# set fixed parameters
	# parameters["gaussH_AlO"]	=  -6.49864123
	# parameters["gaussR_AlO"]	=	1.98975484
	# parameters["gaussW_AlO"]	=	0.32077941
	# parameters["coord_AlO"]		= 	4.16488520
	# parameters["radius_AlO"]	=	2.16333265

	# parameters["chi_Al"]	=	14.6462
	# parameters["chi_H"]		=	271.2607   
	# parameters["chi_Na"]	=	18.2857  
	# parameters["chi_O"]		=	242.4888
   	
	# parameters["eta_Al"]	=	356.2723  
	# parameters["eta_H"]		=	648.4402  
	# parameters["eta_Na"]	=	518.0958
	# parameters["eta_O"]		=	417.4323    
	
	# parameters["gamma_Al"]	=	1.1999
	# parameters["gamma_H"]	=	33.9670  
	# parameters["gamma_Na"]	=	192.4232    
	# parameters["gamma_O"]	=	8.3731 

	parameters["ct_HOH"]	=	-16.6225
	parameters["eps_OO"]	=	  4.9193
	parameters["sigma_OO"]	=	  3.2700

	parameters["gaussH_OH"]	=	 -0.4747
	parameters["gaussR_OH"]	=	  1.0500
	parameters["gaussW_OH"]	=	  0.0759

	parameters["swG_OH"] 	=	 70.0000
	parameters["swS_AlO"]	=	235.0000
	parameters["swS_OH"]	=	154.0000

	# Now resue the previously optimized parameters
	# for key, value in zip(keys_optimized, paramVector_1_optimized):
	# 	parameters[key]	=	value

	# parameters["radius_AlO"]	=	1.5926
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
		# inputObject.write("pair_style hybrid/overlay {} {} sw lj/smooth/linear 5.0 gauss/cut 5.0 \n".format(parameters["coulps"], parameters["coulcut"]))
		inputObject.write("pair_coeff * * {}\n".format(parameters["coulps"]))
		inputObject.write("pair_coeff * * {}\n".format(parameters["coulps"]))
		inputObject.write("pair_coeff * * lj/smooth/linear 0 0 0\n")
		inputObject.write("pair_coeff * * sw param.sw Al O H NULL\n")
		inputObject.write("pair_coeff * * gauss/cut 0 0 1 0\n")
		inputObject.write("\n")		
		inputObject.write("pair_coeff 1 2 lj/smooth/linear 	{}e-2 {}\n".format(parameters["eps_AlO"], 	parameters["sigma_AlO"]))
		inputObject.write("pair_coeff 2 2 lj/smooth/linear 	{}e-2 {}\n".format(parameters["eps_OO"], 	parameters["sigma_OO"]))
		inputObject.write("pair_coeff 2 3 lj/smooth/linear 	{}e-2 {}\n".format(parameters["eps_OH"], 	parameters["sigma_OH"]))
		inputObject.write("pair_coeff 2 4 lj/smooth/linear 	{}e-2 {}\n".format(parameters["eps_NaO"], 	parameters["sigma_NaO"]))
		inputObject.write("\n")
		inputObject.write("pair_coeff 1 1 coord/gauss/cut 	{} {} {} {} {} \n".format(parameters["gaussH_Al"], 		parameters["gaussR_Al"], 	parameters["gaussW_Al"],	parameters["coord_AlO"],	parameters["radius_AlO"]))
		inputObject.write("pair_coeff 1 2 coord/gauss/cut 	{} {} {} {} {} \n".format(parameters["gaussH_AlO"], 	parameters["gaussR_AlO"], 	parameters["gaussW_AlO"],	parameters["coord_Al"],		parameters["radius_Al"]))
		inputObject.write("pair_coeff 2 3 gauss/cut 	{} {} {} 2.0\n".format(parameters["gaussH_OH"], 	parameters["gaussR_OH"], 	parameters["gaussW_OH"]))

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
		#Maxime: constr not needed
		#inputObject.write("group constr {}\n".format(parameters["constraint"]))
		inputObject.write("group counter {}\n".format(parameters["counter"]))
		#inputObject.write("group fixed union constr counter\n")
		inputObject.write("group todump subtract all counter\n")
		#inputObject.write("fix 2 counter setforce 0.0 0.0 0.0\n")
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
		paramObject.write("1 {} {} {}e-2 0 0.0\n".format(parameters["chi_Al"], 	parameters["eta_Al"], 	parameters["gamma_Al"]))
		paramObject.write("2 {} {} {}e-2 0 0.0\n".format(parameters["chi_O"], 	parameters["eta_O"], 	parameters["gamma_O"]))
		paramObject.write("3 {} {} {}e-2 0 0.0\n".format(parameters["chi_H"], 	parameters["eta_H"],	parameters["gamma_H"]))
		paramObject.write("4 {} {} {}e-2 0 0.0\n".format(parameters["chi_Na"], 	parameters["eta_Na"], 	parameters["gamma_Na"]))

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
      #Maxime: counters are arrays#
			#counter 	=	"id {}".format(counters[system])
			counter 	=	"id {}".format(' '.join(map(str, counters[system])))
			#print(counter);sys.exit()
		else:
			counter 	=	"empty"
    #Maxime edit
		#if system in charge_exclusion:
		if system in charge_do:
						#parameters["qdump"]	=	"F"
			parameters["qdump"]	=	"T"
		else:
						#parameters["qdump"]	=	"T"
			parameters["qdump"]	=	"F"

		parameters["constraint"]	=	constraint
		parameters["counter"]		=	counter
		parameters["filename"]		=	system

		# create the LAMMPS input file
		create_LAMMPS_Input(parameters)

def param_optimizer(*args) -> typing.List[float]:
	# clear out old files
	run_lammps(*args)
	clean_slate()

	# print the current set of parameters to a file
	with open("parameter_set_gauss_cut.dat", "a") as fileObject:
		fileObject.write(''.join(f"{param:10.4f}" for param in args[0]))
		fileObject.write("\n")

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

	fmax_values		=	[ener_dat[key][1] for key in ener_dat]
	energy_ener_f	=	[ener_dat[key][0] for key in ener_dat]

	# Create containers to hold error information
	error_charge	=	{}
	error_energy	=	{}
	error_force		=	{}
	error_pressure	=	{}

	for system in training_set:
		# calculate differences in charges
		#Maxime edit
		#if system not in charge_exclusion:
		if system in charge_do:
			xyz_file	=	f"./data/xyz/{system}.xyz"
			esp_file	=	f"./data/esp/{system}.esp"
			charge_file	=	f"{system}.q"
			with open(esp_file) as espObject, open(charge_file) as chargeObject:
				esp_data				=	[float(data.split()[-1]) for data in espObject.readlines()[2:]]
				charge_data				=	[float(data.split()[-1]) for data in chargeObject.readlines()[9:]]
				error_q_file			=	[(esp_value - charge_value) for (esp_value, charge_value) in zip(esp_data, charge_data)]
				error_charge[system]	=	[x for x in error_q_file]
				#print (system, error_charge[system])
		
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
				error_f_file			=	numpy.sqrt(error_f_file.reshape(-1, 3))	# reshapes the array in x, y, z
				error_force[system]		=	numpy.mean((numpy.sum(error_f_file, axis = 1)))	# Takes the sum along the rows

		# Now process trajectories
		if system.find("traj") != -1:
			energy_file		=	f"./data/traj/{system}.ener"
			log_file		=	f"./{system}.log"
			with open(energy_file) as energyObject, open(log_file) as logObject:
				energy_ref				=	numpy.array([float(data.split()[-1]) for data in energyObject.readlines()])
				#Maxime: To get the same error as mine
				#energy_ref_s			=	energy_ref - min(energy_ref)
				energy_ref_s			=	energy_ref - numpy.mean(energy_ref)
				nlines					=	len(energy_ref) + 3
				_ff_energy				=	[data for data in logObject.readlines()[-1 * nlines:]][:-3]
				ff_energy 				=	numpy.array([float(data.split()[0]) for data in _ff_energy])
				#ff_energy_s				=	ff_energy - min(ff_energy)
				ff_energy_s				=	ff_energy - numpy.mean(ff_energy)
				error_e_file			=	[abs(ff_value - ref_value) for (ff_value, ref_value) in zip(ff_energy_s, energy_ref_s)]
				error_energy[system]	=	numpy.mean(numpy.array(error_e_file))

			# Now calculate forces
			force_file	=	f"./data/traj/{system}.frc"
			ff_file		=	f"./{system}.FF.frc"
			with open(force_file) as forceObject, open(ff_file) as ffObject:
				_ref_data 				=	[data.split() for data in forceObject.readlines() if len(data.split()) == 3]
				ref_data 				=	[float(value) for force in _ref_data for value in force]
				ref_force_matrix		=	numpy.array(ref_data).reshape(-1, 3)
				_ff_force				=	[data.split()[1:] for data in ffObject.readlines() if len(data.split()) == 4 and data.find("ITEM") == -1]
				ff_force				=	[float(value) for force in _ff_force for value in force]
				ff_force_matrix			=	numpy.array(ff_force).reshape(-1, 3)
				error_f_file			=	numpy.sum(numpy.power(ff_force_matrix - ref_force_matrix, 2), axis = 1)
				error_force[system]		=	numpy.mean(numpy.sqrt(error_f_file))

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
				error_f_file			=	numpy.sqrt(error_f_file.reshape(-1, 3))	# reshapes the array in x, y, z
				error_force[system]		=	numpy.mean((numpy.sum(error_f_file, axis = 1)))	# Takes the sum along the rows

		if system.find("traj") == -1 and system.find("scan") == -1:
			if system in pressure_data_keys:
				ff_data					=	ener_dat[system][2:]
				ref_data				=	pressure_data_values[system]
				error_p_file			=	[abs(ref_value - ff_value) for (ff_value, ref_value) in zip(ff_data, ref_data)]
				error_pressure[system]	=	0
				for value in error_p_file:
					error_pressure[system] += value
				error_pressure[system] = error_pressure[system]/6

	# Now take care of reactions
	#Maxime: corrected weights
	reaction_energies	=	[
							10.0 * error_energy["gibSEowCNAlAltraj"],
							10.0 * error_energy["gibSEowtraj"],
							10.0 * error_energy["al6OHtraj"],
							10.0 * error_energy["al6H2Otraj"],
							10.0 * error_energy["al6AlAltraj"],
							10.0 * error_energy["3m1traj"],
							10.0 * error_energy["d2btod3traj"],
							10.0 * error_energy["2m1tod2btraj"],
							15.0 * error_energy["d2tod1traj"],
							50.0 * error_energy["32wOOtraj"],
							25.0 * error_energy["32wOHtraj"],
							5.0 * error_energy["NaOH30wPTtraj"],
							5.0 * error_energy["m1NaOHPTtraj"],
							5.0 * error_energy["32wtraj"],
							10.0 * error_energy["32wPTtraj"],
							10.0 * error_energy["2m10traj"],
							10.0 * error_energy["2m1btraj"],
							10.0 * error_energy["2m1ctraj"],
							300./133 * error_energy["m1NaOHtraj"],
							4.0 * error_energy["aloh42h2obtraj"],
							10.0 * error_energy["aloh4scanAlOH"],
							20.0 * error_energy["wscan"],
							10.0 * error_energy["d1scan"],
							2.0 * error_energy["aloh4toaloh5scan"],
							2.0 * error_energy["2m1tod3scan"],
							1.00 * (1 * (ener_dat["d3na"][0] 	- (2 * ener_dat["aloh4na"][0])) - (energy_reference["d3na"])),
							1.00 * (1 * (ener_dat["d1na"][0] 	+ ener_dat["w"][0] 			 	- (2 * ener_dat["aloh4na"][0])) - (energy_reference["d1na"])),
							5.00 * (1 * (ener_dat["w2"][0]   	- (2 * ener_dat["w"][0])) 		- (energy_reference["w2"])),
							5.00 * (1 * (ener_dat["w6"][0]   	- (6 * ener_dat["w"][0]))		- (energy_reference["w6"])),
							0.25 * (1 * (ener_dat["Na5w"][0]   	- (5 * ener_dat["w"][0]		 	+ ener_dat["Na"][0]))			- (energy_reference["Na5w"])),
							1.00 * (1 * (ener_dat["Naw"][0]		- (ener_dat["w"][0]				+ ener_dat["Na"][0]))			- (energy_reference["Naw"])),
							0.50 * (1 * (ener_dat["gib001"][0] 	- (ener_dat["gib001top"][0]		+ ener_dat["gib001bot"][0]))	- (energy_reference["gib001"])),
							0.50 * (1 * (ener_dat["gib825"][0] 	- (ener_dat["gibbulk"][0]))		- (energy_reference["gib825"])),
							0.25 * (1 * (ener_dat["gib110"][0] 	- (ener_dat["gibbulk"][0]))		- (energy_reference["gib110"]))
							]

	# Now compute error terms
	fmax_error		=	numpy.mean(numpy.array([value**2 for value in fmax_values]))
	charge_error	=	numpy.sqrt(numpy.mean(numpy.array([charge**2 for (key,value) in error_charge.items() for charge in value])))
	reactions_error	=	numpy.sqrt(numpy.mean(numpy.array([energy**2 for energy in reaction_energies])))
	force_error		=	numpy.mean(numpy.array([value**2 for (key,value) in error_force.items()]))
	pressure_error	=	numpy.mean(numpy.array([value**2 for (key,value) in error_pressure.items()]))

	# Now we need to compute the final error
	charge_w 		= 300.
	reactions_w  	= 1.
	force_w  		= 0.05
	fmax_w   		= 0.005
	pressure_w   	= 2.5e-6

	errors  = [ charge_error, reactions_error, force_error, fmax_error, pressure_error ]
	weights = [ charge_w, reactions_w, force_w, fmax_w, pressure_w ]
	for i in range(len(errors)):
		errors[i] *= weights[i]

	final_error		=	100 * sum(errors) / sum(weights)
	errors.append(final_error)

  #### PRINTING #####
	#Print all energy errors
	# pprint.pprint (reaction_energies)
	#Print all errors
	print(''.join(f"{error:15.3f}" for error in errors))
	with open("errors_gauss_cut.dat", "a") as errorObject:
		errorObject.write(''.join(f"{error:15.3f}" for error in errors))
		errorObject.write("\n")
	return final_error

def main() -> None:
	clean_slate()	# remove files from previous runs
  
	# param_optimizer(paramVector_2)
	# sys.exit()

	# margin		=	float(sys.argv[1])	# get the margin for fitting from user
	first_minimization	=	scipy.optimize.minimize(param_optimizer, paramVector_2, method='Nelder-Mead',options={'adaptive': True,'maxiter': 1000, 'fatol':1e-3, 'xatol':1e-3})
	print(first_minimization.x)

	# # first do a global minimization 
	# optimizer			=	nlopt.opt(nlopt.G_MLSL_LDS, paramVector_2.size)	
	# local_optimizer		=	nlopt.opt(nlopt.LN_SBPLX, paramVector_2.size)
	# optimizer.set_local_optimizer(local_optimizer)
	# local_optimizer.set_xtol_rel(1e-3)

	# minParams	=	paramVector_2 - (margin * abs(paramVector_2))	# lower bound for parameters
	# maxParams	=	paramVector_2 + (margin * abs(paramVector_2))	# upper bound for parameters

	# optimizer.set_lower_bounds(minParams)
	# optimizer.set_upper_bounds(maxParams)
	# optimizer.set_min_objective(param_optimizer)

	# # Now perform the optimization
	# optimized_values	=	optimizer.optimize(paramVector_2)
	# print(optimized_values)

if __name__ == '__main__':
	main()
