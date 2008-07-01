/*****************************************************************************
 *
 *  d3q15.c
 *
 *  D3Q15 model definitions.
 *
 *  $Id: d3q15.c,v 1.6.6.7 2008-07-01 18:42:26 erlend Exp $
 *
 *  Kevin Stratford (kevin@epcc.ed.ac.uk)
 *
 *****************************************************************************/

#include "pe.h"
#include "d3q15.h"

#ifdef _D3Q15_

/*****************************************************************************
 *
 *  There are 15 eigenvectors:
 *
 *  rho             density
 *  rho cv[p][X]    (x-component of velocity)
 *  rho cv[p][Y]    (y-component of velocity) 
 *  rho cv[p][Z]    (z-component of velocity)
 *  q[p][X][X]      (xx component of deviatoric stress)
 *  q[p][X][Y]      (xy component of deviatoric stress)
 *  q[p][X][Z]      (xz ...
 *  q[p][Y][Y]      (yy ...
 *  q[p][Y][Z]      (yz ...
 *  q[p][Z][Z]      (zz ...
 *  chi1[p]         (1st ghost mode)
 *  jchi1[p][X]     (x-component of ghost current chi1[p]*cv[p][X])
 *  jchi1[p][Y]     (y-component of ghost current chi1[p]*cv[p][Y])
 *  jchi1[p][Z]     (z-component of ghost current chi1[p]*cv[p][Z])
 *  chi3[p]         (2nd ghost mode cv[p][X]*cv[p][Y]*cv[p][Z])
 *
 *  We define the following:
 *
 *  cv[NVEL][3]      lattice velocities (integers)
 *  q_[NVEL][3][3]   kinetic projector c[p][i]*c[p][j] - c_s^2 d_[i][j]
 *  wv[NVEL]         quadrature weight for each velocity
 *  norm_[NVEL]      normaliser for each mode
 *
 *  ma_[NVEL][NVEL]  full matrix of eigenvectors (doubles)
 *  mi_[NVEL][NVEL]  inverse of ma_[][]
 *
 *  The eigenvectors are the rows of the matrix ma_[NVEL][NVEL].
 *
 *****************************************************************************/

#define w0 (16.0/72.0)
#define w1 ( 8.0/72.0)
#define w3 ( 1.0/72.0)

#define c0        0.0
#define c1        1.0
#define c2        2.0
#define r3   (1.0/3.0)
#define r6   (1.0/6.0)
#define t3   (2.0/3.0)

#define wa ( 3.0/72.0)
#define wb ( 9.0/72.0)
#define wc ( 4.0/72.0)

const int cv[NVEL][3] = {{ 0, 0, 0},
			 { 1,  1,  1}, { 1,  1, -1}, { 1,  0,  0},
			 { 1, -1,  1}, { 1, -1, -1}, { 0,  1,  0},
                         { 0,  0,  1}, { 0,  0, -1}, { 0, -1,  0},
			 {-1,  1,  1}, {-1,  1, -1}, {-1,  0,  0},
			 {-1, -1,  1}, {-1, -1, -1}};

const double wv[NVEL] = {w0,
			 w3, w3, w1, w3, w3, w1, w1,
			 w1, w1, w3, w3, w1, w3, w3};
const double norm_[NVEL] = {1.0, 3.0, 3.0, 3.0, 9.0/2.0, 9.0, 9.0,
			    9.0/2.0,9.0,9.0/2.0, 0.5, 1.5, 1.5, 1.5, 9.0};

const  double q_[NVEL][3][3] = {
  {{-r3, 0.0, 0.0},{ 0.0,-r3, 0.0},{ 0.0, 0.0,-r3}},
  {{ t3, 1.0, 1.0},{ 1.0, t3, 1.0},{ 1.0, 1.0, t3}},
  {{ t3, 1.0,-1.0},{ 1.0, t3,-1.0},{-1.0,-1.0, t3}},
  {{ t3, 0.0, 0.0},{ 0.0,-r3, 0.0},{ 0.0, 0.0,-r3}},
  {{ t3,-1.0, 1.0},{-1.0, t3,-1.0},{ 1.0,-1.0, t3}},
  {{ t3,-1.0,-1.0},{-1.0, t3, 1.0},{-1.0, 1.0, t3}},
  {{-r3, 0.0, 0.0},{ 0.0, t3, 0.0},{ 0.0, 0.0,-r3}},
  {{-r3, 0.0, 0.0},{ 0.0,-r3, 0.0},{ 0.0, 0.0, t3}},
  {{-r3, 0.0, 0.0},{ 0.0,-r3, 0.0},{ 0.0, 0.0, t3}},
  {{-r3, 0.0, 0.0},{ 0.0, t3, 0.0},{ 0.0, 0.0,-r3}},
  {{ t3,-1.0,-1.0},{-1.0, t3, 1.0},{-1.0, 1.0, t3}},
  {{ t3,-1.0, 1.0},{-1.0, t3,-1.0},{ 1.0,-1.0, t3}},
  {{ t3, 0.0, 0.0},{ 0.0,-r3, 0.0},{ 0.0, 0.0,-r3}},
  {{ t3, 1.0,-1.0},{ 1.0, t3,-1.0},{-1.0,-1.0, t3}},
  {{ t3, 1.0, 1.0},{ 1.0, t3, 1.0},{ 1.0, 1.0, t3}}};

const double ma_[NVEL][NVEL] = 
  {{ c1, c1, c1, c1, c1, c1, c1, c1, c1, c1, c1, c1, c1, c1, c1},
   { c0, c1, c1, c1, c1, c1, c0, c0, c0, c0,-c1,-c1,-c1,-c1,-c1},
   { c0, c1, c1, c0,-c1,-c1, c1, c0, c0,-c1, c1, c1, c0,-c1,-c1},
   { c0, c1,-c1, c0, c1,-c1, c0, c1,-c1, c0, c1,-c1, c0, c1,-c1},
   {-r3, t3, t3, t3, t3, t3,-r3,-r3,-r3,-r3, t3, t3, t3, t3, t3},
   { c0, c1, c1, c0,-c1,-c1, c0, c0, c0, c0,-c1,-c1, c0, c1, c1},
   { c0, c1,-c1, c0, c1,-c1, c0, c0, c0, c0,-c1, c1, c0,-c1, c1},
   {-r3, t3, t3,-r3, t3, t3, t3,-r3,-r3, t3, t3, t3,-r3, t3, t3},
   { c0, c1,-c1, c0,-c1, c1, c0, c0, c0, c0, c1,-c1, c0,-c1, c1},
   {-r3, t3, t3,-r3, t3, t3,-r3, t3, t3,-r3, t3, t3,-r3, t3, t3},
   {-c2,-c2,-c2, c1,-c2,-c2, c1, c1, c1, c1,-c2,-c2, c1,-c2,-c2},
   { c0,-c2,-c2, c1,-c2,-c2, c0, c0, c0, c0, c2, c2,-c1, c2, c2},
   { c0,-c2,-c2, c0, c2, c2, c1, c0, c0,-c1,-c2,-c2, c0, c2, c2},
   { c0,-c2, c2, c0,-c2, c2, c0, c1,-c1, c0,-c2, c2, c0,-c2, c2},
   { c0, c1,-c1, c0,-c1, c1, c0, c0, c0, c0,-c1, c1, c0, c1,-c1}};

const double mi_[NVEL][NVEL] =
  {{ w0, c0, c0, c0,-r3, c0, c0,-r3, c0,-r3,-w0, c0, c0, c0, c0},
   { w3, wa, wa, wa, wa, wb, wb, wa, wb, wa,-w3,-wa,-wa,-wa, wb},
   { w3, wa, wa,-wa, wa, wb,-wb, wa,-wb, wa,-w3,-wa,-wa, wa,-wb},
   { w1, r3, c0, c0, r3, c0, c0,-r6, c0,-r6, wc, r6, c0, c0, c0},
   { w3, wa,-wa, wa, wa,-wb, wb, wa,-wb, wa,-w3,-wa, wa,-wa,-wb},
   { w3, wa,-wa,-wa, wa,-wb,-wb, wa, wb, wa,-w3,-wa, wa, wa, wb},
   { w1, c0, r3, c0,-r6, c0, c0, r3, c0,-r6, wc, c0, r6, c0, c0},
   { w1, c0, c0, r3,-r6, c0, c0,-r6, c0, r3, wc, c0, c0, r6, c0},
   { w1, c0, c0,-r3,-r6, c0, c0,-r6, c0, r3, wc, c0, c0,-r6, c0},
   { w1, c0,-r3, c0,-r6, c0, c0, r3, c0,-r6, wc, c0,-r6, c0, c0},
   { w3,-wa, wa, wa, wa,-wb,-wb, wa, wb, wa,-w3, wa,-wa,-wa,-wb},
   { w3,-wa, wa,-wa, wa,-wb, wb, wa,-wb, wa,-w3, wa,-wa, wa, wb},
   { w1,-r3, c0, c0, r3, c0, c0,-r6, c0,-r6, wc,-r6, c0, c0, c0},
   { w3,-wa,-wa, wa, wa, wb,-wb, wa,-wb, wa,-w3, wa, wa,-wa, wb},
   { w3,-wa,-wa,-wa, wa, wb, wb, wa, wb, wa,-w3, wa, wa, wa,-wb}};


#ifdef _MPI_

MPI_Datatype xtypes_cv[xcount] = {MPI_DOUBLE};
int xblocklens[xcount] = {5};
int xdisp_right[xcount] = {1};
int xdisp_left[xcount] = {10};

MPI_Datatype ytypes[ycount] = {MPI_DOUBLE};
int yblocklens[ycount] = {2*NVEL};
MPI_Aint ydisp_right[ycount] = {0};
MPI_Aint ydisp_left[ycount] = {0};

MPI_Datatype ztypes[zcount] = {MPI_DOUBLE};
int zblocklens[zcount] = {2*NVEL};
MPI_Aint zdisp_right[zcount] = {0};
MPI_Aint zdisp_left[zcount] = {0};

#endif

#endif
