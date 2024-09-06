// clang-format off
/* ----------------------------------------------------------------------
   LAMMPS - Large-scale Atomic/Molecular Massively Parallel Simulator
   https://www.lammps.org/, Sandia National Laboratories
   LAMMPS development team: developers@lammps.org

   This software is distributed under the GNU General Public License.

   See the README file in the top-level LAMMPS directory.
------------------------------------------------------------------------- */

/* ----------------------------------------------------------------------
   Contributing author: Axel Kohlmeyer (Temple U)
------------------------------------------------------------------------- */

#include "pair_coord_gauss_cut_omp.h"

#include "atom.h"
#include "comm.h"
#include "force.h"
#include "neigh_list.h"
#include "suffix.h"
#include "math_const.h"
#include <cmath>
#include <iostream>

#include "omp_compat.h"
using namespace LAMMPS_NS;
using namespace MathConst;
/* ---------------------------------------------------------------------- */

PairCoordGaussCutOMP::PairCoordGaussCutOMP(LAMMPS *lmp) :
  PairCoordGaussCut(lmp), ThrOMP(lmp, THR_PAIR)
{
  suffix_flag |= Suffix::OMP;
  respa_enable = 0;
  comm_forward = 1;
  comm_reverse = 1;
}

/* ---------------------------------------------------------------------- */

void PairCoordGaussCutOMP::compute(int eflag, int vflag)
{
  ev_init(eflag,vflag);

  const int nall = atom->nlocal + atom->nghost;
  const int nthreads = comm->nthreads;
  const int inum = list->inum;

#if defined(_OPENMP)
#pragma omp parallel LMP_DEFAULT_NONE LMP_SHARED(eflag, vflag)
#endif
  {
    int ifrom, ito, tid;

    loop_setup_thr(ifrom, ito, tid, inum, nthreads);
    ThrData *thr = fix->get_thr(tid);
    thr->timer(Timer::START);
    ev_setup_thr(eflag, vflag, nall, eatom, vatom, nullptr, thr);

    if (evflag) {
      if (eflag) {
        if (force->newton_pair) eval<1,1,1>(ifrom, ito, thr);
        else eval<1,1,0>(ifrom, ito, thr);
      } else {
        if (force->newton_pair) eval<1,0,1>(ifrom, ito, thr);
        else eval<1,0,0>(ifrom, ito, thr);
      }
    } else {
      if (force->newton_pair) eval<0,0,1>(ifrom, ito, thr);
      else eval<0,0,0>(ifrom, ito, thr);
    }

    thr->timer(Timer::PAIR);
    reduce_thr(this, eflag, vflag, thr);
  } // end of omp parallel region
}

template <int EVFLAG, int EFLAG, int NEWTON_PAIR>
void PairCoordGaussCutOMP::eval(int iifrom, int iito, ThrData * const thr)
{

  int    i, j, ii, jj, inum, jnum, itype, jtype;
  double xtmp, ytmp, ztmp, delx, dely, delz, evdwl, fpair;
  double rsq, r, rexp, ugauss, factor_lj;
  double factor_coord, coord_nr, coord_dr, coord_tmp;
  int    *ilist, *jlist, *numneigh, **firstneigh;

  evdwl = 0.0;

  const auto * _noalias const x   =   (dbl3_t *) atom->x[0];
  auto * _noalias const f         =   (dbl3_t *) thr->get_f()[0];
  const int * _noalias const type =   atom->type;
  const int nlocal                =   atom->nlocal;
  const double * _noalias const special_lj = force->special_lj;
  double fxtmp,fytmp,fztmp;

  ilist       =   list->ilist;
  numneigh    =   list->numneigh;
  firstneigh  =   list->firstneigh;
  
  int n_ii    =   iito - iifrom;

  // loop over neighbors of my atoms
  for (ii = iifrom; ii < iito; ++ii) {

    i           =   ilist[ii];
    xtmp        =   x[i].x;
    ytmp        =   x[i].y;
    ztmp        =   x[i].z;
    itype       =   type[i];
    jlist       =   firstneigh[i];
    jnum        =   numneigh[i];
    coord_tmp   =   0.0;
    fxtmp       =   0.0;
    fytmp       =   0.0;
    fztmp       =   0.0;
    // std::cout << iifrom << " " << iito << "\n";
    
    for (jj = 0; jj < jnum; jj++) {
      j         =   jlist[jj];
      factor_lj =   special_lj[sbmask(j)];
      j         &=  NEIGHMASK;

      delx  =   xtmp - x[j].x;
      dely  =   ytmp - x[j].y;
      delz  =   ztmp - x[j].z;
      rsq   =   delx*delx + dely*dely + delz*delz;
      jtype =   type[j];

      r             =   sqrt(rsq);
      factor_coord  =   (r) / rnh[itype][jtype];
      coord_nr      =   1 - pow(factor_coord, 8);
      coord_dr      =   1 - pow(factor_coord, 16);

      if (itype == typea && jtype == typeb) {
        coord_tmp     =   coord_tmp + (coord_nr / coord_dr);
      }
    }

    for (jj = 0; jj < jnum; jj++) {
      j         =   jlist[jj];
      factor_lj =   special_lj[sbmask(j)];
      j         &=  NEIGHMASK;

      delx  =   xtmp - x[j].x;
      dely  =   ytmp - x[j].y;
      delz  =   ztmp - x[j].z;
      rsq   =   delx*delx + dely*dely + delz*delz;
      jtype =   type[j];

      if (rsq <= cutsq[itype][jtype]) {
        r             =   sqrt(rsq);      
        rexp          =   (r-rmh[itype][jtype])/sigmah[itype][jtype];
        // Equation 11 in the Project log
        if (itype == typea && jtype == typeb) {
          if (coord_tmp <= coord[itype][jtype]) {
            double scale_factor  =  (coord_tmp / coord[itype][jtype]) * hgauss[itype][jtype];
            ugauss               =  (scale_factor / sqrt(MY_2PI) / sigmah[itype][jtype]) * exp(-1 * rexp * rexp);
               // std::cout << ii << "\t" << jj << "\t" << scale_factor << "\t" << scale_factor << "\t" << coord_tmp << "\t" << coord[itype][jtype] << "\n";
          }
          else {
            double pre_exponent  =  (coord_tmp - coord[itype][jtype]);
            double scale_factor  =  hgauss[itype][jtype] * exp(-1 * pre_exponent * pre_exponent);
            ugauss               =  (scale_factor / sqrt(MY_2PI) / sigmah[itype][jtype]) * exp(-1 * rexp * rexp);
               // std::cout << ii << "\t" << jj << "\t" << scale_factor << "\t" << scale_factor << "\t" << coord_tmp << "\t" << coord[itype][jtype] << "\n";
            }
          }
          else {
             ugauss               =  (hgauss[itype][jtype]/ sqrt(MY_2PI) / sigmah[itype][jtype]) * exp(-1 * rexp * rexp);
          }

        fpair       =   factor_lj*rexp/r*ugauss/sigmah[itype][jtype];
        ugauss      =   pgauss[itype][jtype]*exp(-0.5*rexp*rexp);
        fpair       =   factor_lj*rexp/r*ugauss/sigmah[itype][jtype];

        fxtmp       +=  delx*fpair;
        fytmp       +=  dely*fpair;
        fztmp       +=  delz*fpair;
        if (NEWTON_PAIR || j < nlocal) {
            f[j].x -= delx*fpair;
            f[j].y -= dely*fpair;
            f[j].z -= delz*fpair;
          }

        if (EFLAG) {
            evdwl = ugauss - offset[itype][jtype];
            evdwl *= factor_lj;
        }

      if (EVFLAG) ev_tally(i, j, nlocal, NEWTON_PAIR, evdwl, 0.0, fpair, delx, dely, delz);
      }
    }
    f[i].x += fxtmp;
    f[i].y += fytmp;
    f[i].z += fztmp;
  }
}

/* ---------------------------------------------------------------------- */

double PairCoordGaussCutOMP::memory_usage()
{
  double bytes = memory_usage_thr();
  bytes += PairCoordGaussCut::memory_usage();

  return bytes;
}
