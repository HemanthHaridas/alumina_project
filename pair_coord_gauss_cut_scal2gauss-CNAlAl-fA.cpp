// clang-format off
/* ----------------------------------------------------------------------
   LAMMPS - Large-scale Atomic/Molecular Massively Parallel Simulator
   https://www.lammps.org/, Sandia National Laboratories
   LAMMPS development team: developers@lammps.org

   Copyright (2003) Sandia Corporation.  Under the terms of Contract
   DE-AC04-94AL85000 with Sandia Corporation, the U.S. Government retains
   certain rights in this software.  This software is distributed under
   the GNU General Public License.

   See the README file in the top-level LAMMPS directory.
------------------------------------------------------------------------- */

/* ----------------------------------------------------------------------
   Contributing authors: Arben Jusufi, Axel Kohlmeyer (Temple U.)
------------------------------------------------------------------------- */

#include "pair_coord_gauss_cut.h"

#include <cmath>
#include <iostream>
#include <fstream>
#include "atom.h"
#include "comm.h"
#include "force.h"
#include "neigh_list.h"
#include "neighbor.h"
#include "memory.h"
#include "error.h"

#include "math_const.h"
//#include <algorithm> // for std::max_element

using namespace LAMMPS_NS;
using namespace MathConst;

/* ---------------------------------------------------------------------- */
// Base case: Single argument prints without trailing space
template<typename T>
void print(const T& arg) {
    std::cout << arg << std::endl;  // Print the last argument followed by a newline
}

// Recursive case: Print first argument followed by a space
template<typename T, typename... Args>
void print(const T& first, const Args&... rest) {
    std::cout << first << ' ';  // Print the argument followed by a space
    print(rest...);             // Recursive call with the remaining arguments
}

////#define watch(x) std::cout << (#x) << "= " << (x) << std::endl

PairCoordGaussCut::PairCoordGaussCut(LAMMPS *lmp) : Pair(lmp)
{
  respa_enable = 0;
  writedata = 1;
}

/* ---------------------------------------------------------------------- */

PairCoordGaussCut::~PairCoordGaussCut()
{
  if (allocated) {
    memory->destroy(setflag);
    memory->destroy(cutsq);
    memory->destroy(cut);
//    memory->destroy(hgauss);
    memory->destroy(lambda);
    memory->destroy(B);
    memory->destroy(pgauss);
    memory->destroy(offset);
    memory->destroy(rnh);
    memory->destroy(NN_coord);
    memory->destroy(poly);
    memory->destroy(central_type);
//    memory->destroy(scale_x);
//    memory->destroy(scale_ab);
//    memory->destroy(NN_scale);
//    memory->destroy(offset_scale);
  }
}

/* ---------------------------------------------------------------------- */

void PairCoordGaussCut::init_style() 
{
  neighbor->add_request(this, NeighConst::REQ_FULL);
}

void PairCoordGaussCut::compute(int eflag, int vflag) {
  int    i, j, ii, jj, inum, jnum, itype, jtype, k;
  double rsq, xtmp, ytmp, ztmp, delx, dely, delz, evdwl, fpair;
  double rexp, U_attr, factor_lj;
  double factor, nr, dr;
  int    *ilist, *jlist, *numneigh, **firstneigh;

  evdwl  =  0.0;
  ev_init(eflag,vflag);

  double **x   =  atom->x;
  double **f   =  atom->f;
  int *type    =  atom->type;
  int ntypes   =  atom->ntypes;
  int nlocal   =  atom->nlocal;
	tagint *tag = atom->tag;

  double *special_lj =  force->special_lj;
  int newton_pair    =  force->newton_pair;

  inum          =   list->inum;
  ilist         =   list->ilist;
  numneigh      =   list->numneigh;
  firstneigh    =   list->firstneigh;

  // create a 2D list to hold the coordination number of each atom with each type
  //double coord_tmp[inum][ntypes+1];
 	double* coord_tmp = new double[ntypes+1];

//	int numneigh_max = 0;
//  for (ii = 0; ii < inum; ii++) {
//		i = ilist[ii];
//		if (numneigh[i] > numneigh_max) {
//					numneigh_max = numneigh[i];
//		}
//    //for (jj = 0; jj < atom->ntypes+1; jj++) {
//    for (jtype = 1; jtype <= ntypes; jtype++) {
//        coord_tmp[ii][jtype] = 0;
//    }
//  }
//  double* r = new double[numneigh_max];
//  bool* keep_neigh = new bool[numneigh_max];

	int npairs = 0;
 	int* central_type_list = new int[ntypes+1];

	int jtype_max = 1;
  for (itype = 1; itype <= ntypes; itype++) {
  	for (jtype = 1; jtype <= ntypes; jtype++) {
			if (setflag[itype][jtype]) {
//				print(itype,jtype,setflag[itype][jtype]);
				npairs++;
				central_type_list[npairs] = central_type[itype][jtype];
				if (jtype > jtype_max) jtype_max=jtype;
			}
	  }
	}
	
  for (ii = 0; ii < inum; ii++) {
    i          =  ilist[ii];
    itype      =  type[i];

		//Skip this ii if the type is not central
		bool found = false;
		for (k = 1; k <= npairs; k++) {
			if (itype == central_type_list[k]) {
				found=true;
				break;
			}
		}
		if (!found) continue;

    xtmp       =  x[i][0];
    ytmp       =  x[i][1];
    ztmp       =  x[i][2];
    jlist      =  firstneigh[i];
    jnum       =  numneigh[i];
//    std::cout << ii << "\n";

		double* r = new double[jnum];
	  bool* keep_neigh = new bool[jnum];
    for (jtype = 1; jtype <= jtype_max; jtype++) coord_tmp[jtype] = 0;

    for (jj = 0; jj < jnum; jj++) {
      j     =  jlist[jj];
      j    &= NEIGHMASK;
      jtype =  type[j];
//    std::cout << jj << "\n";

//		print(i,j,itype,jtype,setflag[itype][jtype]);
			keep_neigh[jj] = false;

			if (setflag[itype][jtype]) {

        delx  =  xtmp - x[j][0];
        dely  =  ytmp - x[j][1];
        delz  =  ztmp - x[j][2];
        rsq   =  delx*delx + dely*dely + delz*delz;

        if (rsq <= cutsq[itype][jtype]) {
			    keep_neigh[jj] = true;
          r[jj]    =  sqrt(rsq);

  				factor =  r[jj] / rnh[itype][jtype];
        	nr     =  1 - pow(factor, NN_coord[itype][jtype]);
        	dr     =  1 - pow(factor, 2*NN_coord[itype][jtype]);
  
        	//coord_tmp[ii][jtype] +=  nr / dr;
        	coord_tmp[jtype] +=  nr / dr;
//					print(tag[i],tag[j],itype,jtype,r[jj],sqrt(cutsq[itype][jtype]), coord_tmp[ii][jtype]);
				}
      }
//      std::cout << "First loop - jj: " << jj << ", jtype: " << jtype << ", keep_neigh[jj]: " << keep_neigh[jj] << std::endl;
    }
// checking CNs
		double scal[ntypes];
		int k;

		double CN = coord_tmp[1];
//    for (jtype = 1; jtype <= ntypes; jtype++) {
						
//			  if (setflag[itype][jtype]) {
		jtype=2;
//		scal[jtype]=poly[itype][jtype][3];

					//double CN = coord_tmp[ii][jtype];
//					double CN = coord_tmp[jtype];
					//print (itype, jtype, CN, poly[itype][jtype][0], poly[itype][jtype][1]);
//					double CNlo = poly[itype][jtype][0];
//					double CNhi = poly[itype][jtype][1];
//					int npoly = int(poly[itype][jtype][2]);
					double x;
//					if (CN < CNlo) {
//						x = CNlo;
//					} else if (CN > CNhi) {
//						x = CNhi;
//					} else {
//						x = CN;
//					}
					x = CN;
					// a*(x-x1)^2 * (x-x2)^2 + b*x + c
//					double A = poly[itype][jtype][3];
//					double x1 = poly[itype][jtype][4];
//					double x2 = poly[itype][jtype][5];
//					double x3 = poly[itype][jtype][6];
//					double b = poly[itype][jtype][7];
//					double c = poly[itype][jtype][8];
//					double n =  poly[itype][jtype][4];
					double H1 =  poly[itype][jtype][3];
					double k1 =  poly[itype][jtype][4];
					double x1 =  poly[itype][jtype][5];
					double H2 =  poly[itype][jtype][6];
					double k2 =  poly[itype][jtype][7];
					double x2 =  poly[itype][jtype][8];
					double H3 =  poly[itype][jtype][9];
					scal[jtype] = H1*exp(-k1*pow(x-x1,2)) + H2*exp(-k2*pow(x-x2,2)) - H3 ;
//					double k3 =  poly[itype][jtype][10];
//					double x3 =  poly[itype][jtype][11];
//					double H4 =  poly[itype][jtype][12];

					//double A = poly[itype][jtype][3];
					//double beta = poly[itype][jtype][4];
					//double n = poly[itype][jtype][5];
					//height_gauss[jtype] = a*pow(x-x1,2)*pow(x-x2,2) + b*x + c;
					//height_gauss[jtype] = a*pow(x-x1,2)*pow(x-x2,2)*pow(x-x3,2) + b*x + c;
					//height_gauss[jtype] = H1*exp(-k1*pow(x-x1,2)) + H2*exp(-k2*pow(x-x2,2)) + H3*exp(-k3*pow(x-x3,2)) - H4 ;
//						height_gauss[jtype] = H1 / (1 + A*pow(abs(x-x1),n)) + H2 / (1 + A*pow(abs(x-x2),n));
//					scal[jtype] = A * pow( (1.+(pow(beta,n))*pow(x,n)), 1./(-1.*A*2.*n));
//	      	for (k = 0; k <= npoly; k++) {
//						height_gauss[jtype] += poly[itype][jtype][k+3]*pow(x,npoly-k);
//			//			print (itype,jtype,k,npoly-k, poly[itype][jtype][k+3]);
//					}
//       	}
//		}
    for (jj = 0; jj < jnum; jj++) {

      if (!keep_neigh[jj]) continue;

      j           =  jlist[jj];
      factor_lj   =  special_lj[sbmask(j)];
      j           &= NEIGHMASK;
      jtype =  type[j];
//		No 1-1 term
      if (jtype==1) continue;
	//print(i,j,it]ype,jtype,setflag[itype][jtype], keep_neigh[jj]);
//			print(i,j,itype,jtype,r[jj],coord_tmp[jtype]);

//      std::cout << "Second loop - jj: " << jj << ", jtype: " << jtype << ", keep_neigh[jj]: " << keep_neigh[jj] << std::endl;
//      rexp     =  (r[jj]-B[itype][jtype])/lambda[itype][jtype];
//      U_attr   =  (height_gauss[jtype] / sqrt(MY_2PI) / lambda[itype][jtype]) * exp(-0.5 * rexp * rexp);
//      fpair    =  factor_lj*rexp/r[jj]*U_attr/lambda[itype][jtype];

			//**later: merge B and scal**//
			//U_attr = B[itype][jtype]*scal[jtype] *exp(-lambda[itype][jtype]*r[jj]);
			U_attr = scal[jtype] *exp(-lambda[itype][jtype]*r[jj]);
			fpair = lambda[itype][jtype]/r[jj] * U_attr;

      f[i][0]   +=    delx*fpair;
      f[i][1]   +=    dely*fpair;
      f[i][2]   +=    delz*fpair;

      if (newton_pair || j < nlocal) {
        f[j][0]    -=    delx*fpair;
        f[j][1]    -=    dely*fpair;
        f[j][2]    -=    delz*fpair;
      }

      if (eflag) {
        evdwl   =  U_attr - offset[itype][jtype];
        evdwl   *= factor_lj;
      }

      if (evflag) ev_tally(i, j, nlocal, newton_pair, evdwl, 0.0, fpair, delx, dely, delz);
    }
		delete[] r;
		delete[] keep_neigh;
  }
if (vflag_fdotr) virial_fdotr_compute();
delete[] central_type_list;
delete[] coord_tmp;
}

/* ----------------------------------------------------------------------
   allocate all arrays
------------------------------------------------------------------------- */

void PairCoordGaussCut::allocate()
{
  allocated = 1;
  int n = atom->ntypes;

  memory->create(setflag,  n+1,  n+1,  "pair:setflag");
//  memory->create(scale_x,	  n+1,  n+1,  2, 100, "pair:scale_x");
//  memory->create(scale_ab,  n+1,  n+1,  2, 100, "pair:scale_ab");
  for (int i = 1; i <= n; i++) {
         for (int j = 1; j <= n; j++) {
             setflag[i][j] = 0;
         }
  }

  memory->create(cutsq,     n+1,  n+1,  "pair:cutsq");
  memory->create(cut,       n+1,  n+1,  "pair:cut");
//  memory->create(hgauss,    n+1,  n+1,  "pair:hgauss");
  memory->create(lambda,    n+1,  n+1,  "pair:lambda");
  memory->create(B,       n+1,  n+1,  "pair:B");
  memory->create(pgauss,    n+1,  n+1,  "pair:pgauss");
  memory->create(offset,    n+1,  n+1,  "pair:offset");
  memory->create(rnh,       n+1,  n+1,  "pair:rnh");
  memory->create(NN_coord,  n+1,  n+1,  "pair:NN_coord");
  memory->create(poly, 		  n+1,  n+1, 10, "pair:poly");
  memory->create(central_type, n+1,  n+1, "pair:central_type");
}

/* ----------------------------------------------------------------------
   global settings
------------------------------------------------------------------------- */
// Utility function that prints all arguments with std::cout


void PairCoordGaussCut::settings(int narg, char **arg)
{
  //if (narg != 1) error->all(FLERR,"Illegal pair_style command");
  if (narg != 2) error->all(FLERR,"Illegal pair_style command");
  cut_global = utils::numeric(FLERR,arg[0],false,lmp);
//	std::cout << "arg[0] = " << arg[0] << "\n";
// reset cutoffs that have been explicitly set

  const char* filename;
	filename = arg[1];
  std::ifstream file(filename);
	std::cout << "filename=" << filename << std::endl;
	// ATTENTION allocate //
  if (!allocated) allocate();

	double coeff;
	size_t count=0;
  // Read the entire line character by character
	
  int i, j;
//reading itype, jtype, CNlo, CNhi, polynomial order
	//we need npoly = 4
	while (file >> i >> j >> poly[i][j][0] >> poly[i][j][1] >> poly[i][j][2]) {
		count = 0;
	  while (file >> coeff) {
				count++;
	      poly[i][j][count+2] = coeff;
//				print(i,j,count,coeff);
	      if (count == poly[i][j][2]+1) break;
		}
	}
  file.close();

  if (allocated) {
    int i,j;
    for (i = 1; i <= atom->ntypes; i++)
      for (j = i; j <= atom->ntypes; j++)
        if (setflag[i][j]) cut[i][j] = cut_global;
  } else {
	   std::cout << "Not allocated" << std::endl;
	}
}

/* ----------------------------------------------------------------------
   set coeffs for one or more type pairs
------------------------------------------------------------------------- */

void PairCoordGaussCut::coeff(int narg, char **arg)
{
  if (narg < 6 || narg > 7) error->all(FLERR,"Incorrect args for pair coefficients");
  if (!allocated) allocate();

  //int ilo,ihi,jlo,jhi;
  //utils::bounds(FLERR,arg[0],1,atom->ntypes,ilo,ihi,error);
  //utils::bounds(FLERR,arg[1],1,atom->ntypes,jlo,jhi,error);
	//std::cout << "ilo ihi " << ilo << " " << ihi << " " << jlo << " " << jhi << "\n";
	// For this pair style ilo=ihi and jlo=jhi (unique pairs of atom types, not a range)

  int i  =  utils::numeric(FLERR,arg[0],false,lmp);
  int j  =  utils::numeric(FLERR,arg[1],false,lmp);
  int central_type_one  =  utils::numeric(FLERR,arg[2],false,lmp);
 // double B_one      =   utils::numeric(FLERR,arg[3],false,lmp);
  double lambda_one   =   utils::numeric(FLERR,arg[3],false,lmp);
  double rnh_one      =   utils::numeric(FLERR,arg[4],false,lmp);
  double NN_coord_one =   utils::numeric(FLERR,arg[5],false,lmp);
  double hgauss_one   =   utils::numeric(FLERR,arg[2],false,lmp);
//	print(arg[2],arg[3],arg[4],arg[5]);

	// The CN is between central_type (which must be i or j) - and i or j (the one that is not central_type)
	// We need the central atom to also be the first atom of the pair for convenience
	if ( central_type_one == j) {
    std::swap(i, j); // Swaps the values of i and j
	} else if ( central_type_one != i ) {
    error->all(FLERR, "The central atom type for the coordination number must be one of the types of the pair");
	}
  if (lambda_one <= 0.0)
    error->all(FLERR,"Incorrect args for pair coefficients");

  double cut_one     = cut_global;
  if (narg == 7) cut_one = utils::numeric(FLERR,arg[7],false,lmp);

  int count = 0;
//  for (int i = ilo; i <= ihi; i++) {
//    for (int j = MAX(jlo,i); j <= jhi; j++) {
//      hgauss[i][j]    =   hgauss_one;

   central_type[i][j] =  central_type_one;
//   B[i][j]      =  B_one;
   lambda[i][j] =  lambda_one;
   cut[i][j]        =  cut_one;
   rnh[i][j]        =  rnh_one; 
   NN_coord[i][j]   =  NN_coord_one; 
   setflag[i][j]    =  1;
   count++;
//    }
//  }

  if (count == 0) error->all(FLERR,"Incorrect args for pair coefficients");
}

/* ----------------------------------------------------------------------
   init for one type pair i,j and corresponding j,i
------------------------------------------------------------------------- */

double PairCoordGaussCut::init_one(int i, int j)
{
// We don't need mixing
//  if (setflag[i][j] == 0) {
//    hgauss[i][j]  =  mix_energy(fabs(hgauss[i][i]), fabs(hgauss[j][j]), fabs(lambda[i][i]), fabs(lambda[j][j]));

    // If either of the particles is repulsive (ie, if hgauss > 0),
    // then the interaction between both is repulsive.
//    double sign_hi = (hgauss[i][i] >= 0.0) ? 1.0 : -1.0;
//    double sign_hj = (hgauss[j][j] >= 0.0) ? 1.0 : -1.0;
//    hgauss[i][j]  *= MAX(sign_hi, sign_hj);
//    lambda[i][j]  =  mix_distance(lambda[i][i], lambda[j][j]);
//    B[i][j]     =  mix_distance(B[i][i], B[j][j]);
//    cut[i][j]     =  mix_distance(cut[i][i], cut[j][j]);
//  }

  //pgauss[i][j]    = hgauss[i][j] / sqrt(MY_2PI) / lambda[i][j];

  // if (offset_flag) {
  //   double rexp   = (cut[i][j]-B[i][j])/lambda[i][j];
  //   offset[i][j]  = pgauss[i][j] * exp(-0.5*rexp*rexp);
  // } else offset[i][j] = 0.0;

//**We want the central particle to be i so we don't want that:**
 // hgauss[j][i]    =  hgauss[i][j];
 // lambda[j][i]    =  lambda[i][j];
 // B[j][i]       =  B[i][j];
  //pgauss[j][i]    =  pgauss[i][j];
 // offset[j][i]    =  offset[i][j];
 // cut[j][i]       =  cut[i][j];
 // rnh[j][i]       =  rnh[i][j];

  // compute I,J contribution to long-range tail correction
  // count total # of atoms of type I and J via Allreduce

  if (tail_flag) {
    int *type  = atom->type;
    int nlocal = atom->nlocal;

    double count[2],all[2];
    count[0]   = count[1] = 0.0;
    for (int k = 0; k < nlocal; k++) {
      if (type[k] == i) count[0] += 1.0;
      if (type[k] == j) count[1] += 1.0;
    }
    MPI_Allreduce(count,all,2,MPI_DOUBLE,MPI_SUM,world);
  }
  return cut[i][j];
}

/* ----------------------------------------------------------------------
   proc 0 writes to restart file
------------------------------------------------------------------------- */

void PairCoordGaussCut::write_restart(FILE *fp)
{
  write_restart_settings(fp);

  int i,j;
  for (i = 1; i <= atom->ntypes; i++)
    for (j = i; j <= atom->ntypes; j++) {
      fwrite(&setflag[i][j],  sizeof(int), 1, fp);
      if (setflag[i][j]) {
//      fwrite(&hgauss[i][j], sizeof(double), 1, fp);
        fwrite(&B[i][j],    sizeof(double), 1, fp);
        fwrite(&lambda[i][j], sizeof(double), 1, fp);
        fwrite(&cut[i][j],    sizeof(double), 1, fp);
      }
    }
}

/* ----------------------------------------------------------------------
   proc 0 reads from restart file, bcasts
------------------------------------------------------------------------- */

void PairCoordGaussCut::read_restart(FILE *fp)
{
  read_restart_settings(fp);
  allocate();

  int i,j;
  int me = comm->me;
  for (i = 1; i <= atom->ntypes; i++)
    for (j = i; j <= atom->ntypes; j++) {
      if (me == 0) utils::sfread(FLERR, &setflag[i][j],  sizeof(int), 1, fp, nullptr, error);
      MPI_Bcast(&setflag[i][j], 1, MPI_INT, 0, world);
      if (setflag[i][j]) {
        if (me == 0) {
//          utils::sfread(FLERR, &hgauss[i][j], sizeof(double), 1, fp, nullptr, error);
          utils::sfread(FLERR, &B[i][j],    sizeof(double), 1, fp, nullptr, error);
          utils::sfread(FLERR, &lambda[i][j], sizeof(double), 1, fp, nullptr, error);
          utils::sfread(FLERR, &cut[i][j],    sizeof(double), 1, fp, nullptr, error);
        }
//        MPI_Bcast(&hgauss[i][j], 1, MPI_DOUBLE, 0, world);
        MPI_Bcast(&B[i][j],    1, MPI_DOUBLE, 0, world);
        MPI_Bcast(&lambda[i][j], 1, MPI_DOUBLE, 0, world);
        MPI_Bcast(&cut[i][j],    1, MPI_DOUBLE, 0, world);
      }
    }
}

/* ----------------------------------------------------------------------
   proc 0 writes to restart file
------------------------------------------------------------------------- */

void PairCoordGaussCut::write_restart_settings(FILE *fp)
{
  fwrite(&cut_global,   sizeof(double),   1, fp);
  fwrite(&offset_flag,  sizeof(int),      1, fp);
  fwrite(&mix_flag,     sizeof(int),      1, fp);
}

/* ----------------------------------------------------------------------
   proc 0 reads from restart file, bcasts
------------------------------------------------------------------------- */

void PairCoordGaussCut::read_restart_settings(FILE *fp)
{
  int me = comm->me;
  if (me == 0) {
    utils::sfread(FLERR,   &cut_global,   sizeof(double),   1, fp,   nullptr, error);
    utils::sfread(FLERR,   &offset_flag,  sizeof(int),      1, fp,   nullptr, error);
    utils::sfread(FLERR,   &mix_flag,     sizeof(int),      1, fp,   nullptr, error);
  }
  MPI_Bcast(&cut_global,   1, MPI_DOUBLE, 0, world);
  MPI_Bcast(&offset_flag,  1, MPI_INT,    0, world);
  MPI_Bcast(&mix_flag,     1, MPI_INT,    0, world);
}

/* ----------------------------------------------------------------------
   proc 0 writes to data file
------------------------------------------------------------------------- */

void PairCoordGaussCut::write_data(FILE *fp)
{
  for (int i = 1; i <= atom->ntypes; i++)
    //fprintf(fp,"%d %g %g %g\n",  i, hgauss[i][i],  B[i][i],  lambda[i][i]);
    fprintf(fp,"%d %g %g\n",  i, B[i][i],  lambda[i][i]);
}

/* ----------------------------------------------------------------------
   proc 0 writes all pairs to data file
------------------------------------------------------------------------- */

void PairCoordGaussCut::write_data_all(FILE *fp)
{
  for (int i = 1; i <= atom->ntypes; i++)
    for (int j = i; j <= atom->ntypes; j++)
      //fprintf(fp,"%d %d %g %g %g %g\n",   i, j, hgauss[i][j],  B[i][j],  lambda[i][j],  cut[i][j]);
      fprintf(fp,"%d %d %g %g %g\n",   i, j, B[i][j],  lambda[i][j],  cut[i][j]);
}

/* ---------------------------------------------------------------------- */


/* ---------------------------------------------------------------------- */
double PairCoordGaussCut::memory_usage()
{
  const int n  =  atom->ntypes;

  double bytes = Pair::memory_usage();

  bytes += 7.0*((n+1.0)*(n+1.0) * sizeof(double) + (n+1.0)*sizeof(double *));
  bytes += 1.0*((n+1.0)*(n+1.0) * sizeof(int) + (n+1.0)*sizeof(int *));

  return bytes;
}
