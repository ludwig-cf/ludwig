/*****************************************************************************
 *
 *  d3q15.c
 *
 *  D3Q15 model definitions.
 *
 *  Kevin Stratford (kevin@epcc.ed.ac.uk)
 *
 *  Edinburgh Soft Matter and Statistical Physics Group and
 *  Edinburgh Parallel Computing Centre
 *
 *  (c) 2008-2014 The University of Edinburgh
 *  Contributing authors:
 *    Kevin Stratford (kevin@epcc.ed.ac.uk)
 *    Ronojoy Adhikari computed this D3Q15 basis.
 *
 *****************************************************************************/

#include "pe.h"
#include "d3q15.h"

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
 *  Reduced halo swap information
 *  CVXBLOCK         number of separate blocks to send in x-direction
 *  CVYBLOCK         ditto                            ... y-direction
 *  CVZBLOCK         ditto                            ... z-direction
 *
 *  For each direction there is then an array of ...
 *
 *  blocklen         block lengths
 *  disp_fwd         displacements of block from start (forward direction)
 *  disp_bwd         displacements of block from start (backward direction)
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


const int xblocklen_cv[CVXBLOCK] = {5};
const int xdisp_fwd_cv[CVXBLOCK] = {1};
const int xdisp_bwd_cv[CVXBLOCK] = {10};

const int yblocklen_cv[CVYBLOCK] = {2, 1, 2};
const int ydisp_fwd_cv[CVYBLOCK] = {1, 6, 10};
const int ydisp_bwd_cv[CVYBLOCK] = {4, 9, 13};

const int zblocklen_cv[CVZBLOCK] = {1, 1, 1, 1, 1};
const int zdisp_fwd_cv[CVZBLOCK] = {1, 4, 7, 10, 13};
const int zdisp_bwd_cv[CVZBLOCK] = {2, 5, 8, 11, 14};
