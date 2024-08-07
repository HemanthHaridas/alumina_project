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
#include "atom.h"
#include "comm.h"
#include "force.h"
#include "neigh_list.h"
#include "neighbor.h"
#include "memory.h"
#include "error.h"

#include "math_const.h"

using namespace LAMMPS_NS;
using namespace MathConst;

/* ---------------------------------------------------------------------- */

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
    memory->destroy(hgauss);
    memory->destroy(sigmah);
    memory->destroy(rmh);
    memory->destroy(pgauss);
    memory->destroy(offset);
    memory->destroy(coord);
    memory->destroy(rnh);
  }
}

/* ---------------------------------------------------------------------- */

void PairCoordGaussCut::init_style() 
{
  neighbor->add_request(this, NeighConst::REQ_FULL);
}

void PairCoordGaussCut::compute(int eflag, int vflag) {
  int    i, j, ii, jj, inum, jnum, itype, jtype;
  double xtmp, ytmp, ztmp, delx, dely, delz, evdwl, fpair;
  double rsq, r, rexp, ugauss, factor_lj, coord_tmp;
  double factor_coord, coord_nr, coord_dr;
  int    *ilist, *jlist, *numneigh, **firstneigh;

  evdwl  =  0.0;
  ev_init(eflag,vflag);

  double **x   =  atom->x;
  double **f   =  atom->f;
  int *type    =  atom->type;
  int nlocal   =  atom->nlocal;

  double *special_lj =  force->special_lj;
  int newton_pair    =  force->newton_pair;

  inum          =   list->inum;
  ilist         =   list->ilist;
  numneigh      =   list->numneigh;
  firstneigh    =   list->firstneigh;


  for (ii = 0; ii < inum; ii++) {
    i          =  ilist[ii];
    xtmp       =  x[i][0];
    ytmp       =  x[i][1];
    ztmp       =  x[i][2];
    itype      =  type[i];
    jlist      =  firstneigh[i];
    jnum       =  numneigh[i];
    coord_tmp  =  0.0;
      
    for (jj = 0; jj < jnum; jj++) {
      j           =  jlist[jj];
      factor_lj   =  special_lj[sbmask(j)];
      j           &= NEIGHMASK;

      delx  =  xtmp - x[j][0];
      dely  =  ytmp - x[j][1];
      delz  =  ztmp - x[j][2];
      rsq   =  delx*delx + dely*dely + delz*delz;
      jtype =  type[j];

      r            =  sqrt(rsq);
      factor_coord =  (r - rnh[itype][jtype]) / rnh[itype][jtype];
      coord_nr     =  1 - pow(factor_coord, 6);
      coord_dr     =  1 - pow(factor_coord, 12);
      // rexp         =  (r-rmh[itype][jtype])/sigmah[itype][jtype];

      // check if outerloop is Al and inner loop is O
      if (itype == typea && jtype == typeb) {
        coord_tmp    =  coord_tmp + (coord_nr / coord_dr);
      }
    }

    for (jj = 0; jj < jnum; jj++) {
      j           =  jlist[jj];
      factor_lj   =  special_lj[sbmask(j)];
      j           &= NEIGHMASK;

      delx  =  xtmp - x[j][0];
      dely  =  ytmp - x[j][1];
      delz  =  ztmp - x[j][2];
      rsq   =  delx*delx + dely*dely + delz*delz;
      jtype =  type[j];

      r            =  sqrt(rsq);
      rexp         =  (r-rmh[itype][jtype])/sigmah[itype][jtype];

      if (itype == typea && jtype == typeb) {
        if (coord_tmp <= coord[itype][itype]) {
           double scale_factor  =  (coord_tmp / coord[itype][itype]) * hgauss[itype][jtype];
           ugauss               =  (scale_factor / sqrt(MY_2PI) / sigmah[itype][jtype]) * exp(-1 * rexp * rexp);
           // std::cout << ii << "\t" << itype << "\t" << jj << "\t" << jtype << "\t" << ugauss << "\t" << coord_tmp << "\n";
        }
        else {
           double pre_exponent  =  (coord_tmp - coord[itype][itype]);
           double scale_factor  =  hgauss[itype][jtype] * exp(-1 * pre_exponent * pre_exponent);
           ugauss               =  (scale_factor / sqrt(MY_2PI) / sigmah[itype][jtype]) * exp(-1 * rexp * rexp);
           // std::cout << ii << "\t" << itype << "\t" << jj << "\t" << jtype << "\t" << ugauss << "\t" << coord_tmp << "\n";
        }
      }
      else{
          ugauss               =  (hgauss[itype][jtype]/ sqrt(MY_2PI) / sigmah[itype][jtype]) * exp(-1 * rexp * rexp);
          // std::cout << ii << "\t" << itype << "\t" << jj << "\t" << jtype << "\t" << ugauss << "\t" << coord_tmp << "\n";
      }

      fpair        =  factor_lj*rexp/r*ugauss/sigmah[itype][jtype];

      f[i][0]   +=    delx*fpair;
      f[i][1]   +=    dely*fpair;
      f[i][2]   +=    delz*fpair;

      if (newton_pair || j < nlocal) {
         f[j][0]    -=    delx*fpair;
         f[j][1]    -=    dely*fpair;
         f[j][2]    -=    delz*fpair;
      }

      if (eflag) {
         evdwl   =  ugauss - offset[itype][jtype];
         evdwl   *= factor_lj;
      }

      if (evflag) ev_tally(i, j, nlocal, newton_pair, evdwl, 0.0, fpair, delx, dely, delz);
    }
  }

  if (vflag_fdotr) virial_fdotr_compute();
}

/* ----------------------------------------------------------------------
   allocate all arrays
------------------------------------------------------------------------- */

void PairCoordGaussCut::allocate()
{
  allocated = 1;
  int n = atom->ntypes;

  memory->create(setflag,  n+1,  n+1,  "pair:setflag");
  for (int i = 1; i <= n; i++)
    for (int j = i; j <= n; j++)
      setflag[i][j] = 0;

  memory->create(cutsq,    n+1,  n+1,  "pair:cutsq");
  memory->create(cut,      n+1,  n+1,  "pair:cut");
  memory->create(coord,    n+1,  n+1,  "pair:coord");
  memory->create(hgauss,   n+1,  n+1,  "pair:hgauss");
  memory->create(sigmah,   n+1,  n+1,  "pair:sigmah");
  memory->create(rmh,      n+1,  n+1,  "pair:rmh");
  memory->create(pgauss,   n+1,  n+1,  "pair:pgauss");
  memory->create(offset,   n+1,  n+1,  "pair:offset");
  memory->create(rnh,      n+1,  n+1,  "pair:rnh");
}

/* ----------------------------------------------------------------------
   global settings
------------------------------------------------------------------------- */

void PairCoordGaussCut::settings(int narg, char **arg)
{
  if (narg != 1) error->all(FLERR,"Illegal pair_style command");

  cut_global = utils::numeric(FLERR,arg[0],false,lmp);

  // reset cutoffs that have been explicitly set

  if (allocated) {
    int i,j;
    for (i = 1; i <= atom->ntypes; i++)
      for (j = i; j <= atom->ntypes; j++)
        if (setflag[i][j]) cut[i][j] = cut_global;
  }
}

/* ----------------------------------------------------------------------
   set coeffs for one or more type pairs
------------------------------------------------------------------------- */

void PairCoordGaussCut::coeff(int narg, char **arg)
{
  if (narg < 6 || narg > 8) error->all(FLERR,"Incorrect args for pair coefficients");
  if (!allocated) allocate();

  int ilo,ihi,jlo,jhi;
  utils::bounds(FLERR,arg[0],1,atom->ntypes,ilo,ihi,error);
  utils::bounds(FLERR,arg[1],1,atom->ntypes,jlo,jhi,error);

  double hgauss_one   =   utils::numeric(FLERR,arg[2],false,lmp);
  double rmh_one      =   utils::numeric(FLERR,arg[3],false,lmp);
  double sigmah_one   =   utils::numeric(FLERR,arg[4],false,lmp);
  double coord_one    =   utils::numeric(FLERR,arg[5],false,lmp);
  double rnh_one      =   utils::numeric(FLERR,arg[6],false,lmp);

  typea               =   ilo;
  typeb               =   jlo; 

  if (sigmah_one <= 0.0)
    error->all(FLERR,"Incorrect args for pair coefficients");


  double cut_one     = cut_global;
  // if (narg == 8) cut_one = utils::numeric(FLERR,arg[8],false,lmp);

  int count = 0;
  for (int i = ilo; i <= ihi; i++) {
    for (int j = MAX(jlo,i); j <= jhi; j++) {
      hgauss[i][j]    =   hgauss_one;
      sigmah[i][j]    =   sigmah_one;
      rmh[i][j]       =   rmh_one;
      cut[i][j]       =   cut_one;
      coord[i][j]     =   coord_one;
      rnh[i][j]       =   rnh_one; 

      hgauss[j][i]    =   hgauss[i][j];
      sigmah[j][i]    =   sigmah[i][j];
      rmh[j][i]       =   rmh[i][j];
      rnh[j][i]       =   rnh[i][j];
      coord[j][i]     =   coord[i][j];
      pgauss[j][i]    =   pgauss[i][j];
      offset[j][i]    =   offset[i][j];
      cut[j][i]       =   cut[i][j];
      // std::cout << ilo << " " << ihi << " " << jlo << " " << jhi << " " << coord[i][j] << " " << rmh[i][j] << "\n";
      setflag[i][j]  =  1;
      count++;
    }
  }

  if (count == 0) error->all(FLERR,"Incorrect args for pair coefficients");
}

/* ----------------------------------------------------------------------
   init for one type pair i,j and corresponding j,i
------------------------------------------------------------------------- */

double PairCoordGaussCut::init_one(int i, int j)
{
  if (setflag[i][j] == 0) {
    hgauss[i][j]  =  mix_energy(fabs(hgauss[i][i]), fabs(hgauss[j][j]), fabs(sigmah[i][i]), fabs(sigmah[j][j]));

    // If either of the particles is repulsive (ie, if hgauss > 0),
    // then the interaction between both is repulsive.
    double sign_hi = (hgauss[i][i] >= 0.0) ? 1.0 : -1.0;
    double sign_hj = (hgauss[j][j] >= 0.0) ? 1.0 : -1.0;
    hgauss[i][j]  *= MAX(sign_hi, sign_hj);
    sigmah[i][j]  =  mix_distance(sigmah[i][i], sigmah[j][j]);
    rmh[i][j]     =  mix_distance(rmh[i][i], rmh[j][j]);
    cut[i][j]     =  mix_distance(cut[i][i], cut[j][j]);
  }

  pgauss[i][j]    = hgauss[i][j] / sqrt(MY_2PI) / sigmah[i][j];

  // if (offset_flag) {
  //   double rexp   = (cut[i][j]-rmh[i][j])/sigmah[i][j];
  //   offset[i][j]  = pgauss[i][j] * exp(-0.5*rexp*rexp);
  // } else offset[i][j] = 0.0;

  hgauss[j][i]    =  hgauss[i][j];
  sigmah[j][i]    =  sigmah[i][j];
  rmh[j][i]       =  rmh[i][j];
  coord[j][i]     =  coord[i][j];
  pgauss[j][i]    =  pgauss[i][j];
  offset[j][i]    =  offset[i][j];
  cut[j][i]       =  cut[i][j];
  rnh[j][i]       =  rnh[i][j];

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
        fwrite(&hgauss[i][j], sizeof(double), 1, fp);
        fwrite(&rmh[i][j],    sizeof(double), 1, fp);
        fwrite(&sigmah[i][j], sizeof(double), 1, fp);
        fwrite(&cut[i][j],    sizeof(double), 1, fp);
        fwrite(&coord[i][j],  sizeof(double), 1, fp);
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
          utils::sfread(FLERR, &hgauss[i][j], sizeof(double), 1, fp, nullptr, error);
          utils::sfread(FLERR, &rmh[i][j],    sizeof(double), 1, fp, nullptr, error);
          utils::sfread(FLERR, &sigmah[i][j], sizeof(double), 1, fp, nullptr, error);
          utils::sfread(FLERR, &cut[i][j],    sizeof(double), 1, fp, nullptr, error);
          utils::sfread(FLERR, &coord[i][j],  sizeof(double), 1, fp, nullptr, error);
        }
        MPI_Bcast(&hgauss[i][j], 1, MPI_DOUBLE, 0, world);
        MPI_Bcast(&rmh[i][j],    1, MPI_DOUBLE, 0, world);
        MPI_Bcast(&sigmah[i][j], 1, MPI_DOUBLE, 0, world);
        MPI_Bcast(&cut[i][j],    1, MPI_DOUBLE, 0, world);
        MPI_Bcast(&coord[i][j],  1, MPI_DOUBLE, 0, world);
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
    fprintf(fp,"%d %g %g %g %g\n",  i, hgauss[i][i],  rmh[i][i],  sigmah[i][i], coord[i][i]);
}

/* ----------------------------------------------------------------------
   proc 0 writes all pairs to data file
------------------------------------------------------------------------- */

void PairCoordGaussCut::write_data_all(FILE *fp)
{
  for (int i = 1; i <= atom->ntypes; i++)
    for (int j = i; j <= atom->ntypes; j++)
      fprintf(fp,"%d %d %g %g %g %g %g\n",   i, j, hgauss[i][j],  rmh[i][j],  sigmah[i][j],  cut[i][j],  coord[i][j]);
}

/* ---------------------------------------------------------------------- */

// double PairCoordGaussCut::single(int /*i*/, int /*j*/, int itype, int jtype, double rsq, double /*factor_coul*/, double factor_lj, double &fforce)
// {
//    double r, rexp,ugauss,phigauss, coord_tmp;

//    r      =  sqrt(rsq);
//    rexp   =  (r-rmh[itype][jtype])/sigmah[itype][jtype];
//    if (coord_tmp <= 6) {
//       double scale_factor  =  (coord_tmp/ coord[itype][jtype]) * hgauss[itype][jtype];
//       ugauss               =  (scale_factor / sqrt(MY_2PI) / sigmah[itype][jtype]) * exp(-1 * rexp * rexp);
//    }
//    else {
//       double pre_exponent  =  (coord_tmp - coord[itype][jtype]) / 0.5;
//       double scale_factor  =  hgauss[itype][jtype] * exp(-1 * pre_exponent * pre_exponent);
//       ugauss               =  (scale_factor / sqrt(MY_2PI) / sigmah[itype][jtype]) * exp(-1 * rexp * rexp);
//    }
//   fforce =  factor_lj*rexp/r*ugauss/sigmah[itype][jtype];

//   phigauss = ugauss - offset[itype][jtype];
//   return factor_lj*phigauss;
// }

/* ---------------------------------------------------------------------- */
double PairCoordGaussCut::memory_usage()
{
  const int n  =  atom->ntypes;

  double bytes = Pair::memory_usage();

  bytes += 7.0*((n+1.0)*(n+1.0) * sizeof(double) + (n+1.0)*sizeof(double *));
  bytes += 1.0*((n+1.0)*(n+1.0) * sizeof(int) + (n+1.0)*sizeof(int *));

  return bytes;
}
