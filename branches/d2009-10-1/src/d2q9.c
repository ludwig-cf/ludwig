/*****************************************************************************
 *
 *  d2q9.c
 *
 *  D2Q9 definitions.
 *
 *  $Id: d2q9.c,v 1.1.2.1 2010-02-13 15:34:46 kevin Exp $
 *
 *  Edinburgh Soft Matter and Statistical Physics Group and
 *  Edinburgh Parallel Computing Centre
 *
 *  Kevin Stratford (kevin@epcc.ed.ac.uk)
 *  (c) 2010 The University of Edinburgh
 *
 *****************************************************************************/

#include "d2q9.h"

#ifdef _D2Q9_

/*****************************************************************************
 *
 *  There are 9 eigenvectors:
 *
 *  rho             (density)
 *  rho cv[p][X]    (x-component of velocity)
 *  rho cv[p][Y]    (y-component of velocity) 
 *  q[p][X][X]      (xx component of deviatoric stress)
 *  q[p][X][Y]      (xy component of deviatoric stress)
 *  q[p][Y][Y]      (yy ...
 *  chi1[p]         (1st ghost mode)
 *  jchi1[p][X]     (x-component of ghost current chi1[p]*rho*cv[p][X])
 *  jchi1[p][Y]     (y-component of ghost current chi1[p]*rho*cv[p][Y])
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
 *  blocklens        reduced halo datatype blocklengths
 *  disp_fwd         reduced halo datatype displacements
 *  disp_bwd         reduced halo datatype displacements
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
 *  Note that all z-components are zero.
 *
 *****************************************************************************/


const int cv[NVEL][3] = {{ 0,  0,  0},
			 { 1,  1,  0}, { 1,  0,  0},
			 { 1, -1,  0}, { 0,  1,  0},
			 { 0, -1,  0}, {-1,  1,  0},
			 {-1,  0,  0}, {-1, -1,  0}};

const double q_[NVEL][3][3] = {
  {{-1.0/3.0, 0.0, 0.0}, { 0.0,-1.0/3.0, 0.0}, { 0.0, 0.0, 0.0}},
  {{ 2.0/3.0, 1.0, 0.0}, { 1.0, 2.0/3.0, 0.0}, { 0.0, 0.0, 0.0}},
  {{ 2.0/3.0, 0.0, 0.0}, { 0.0,-1.0/3.0, 0.0}, { 0.0, 0.0, 0.0}},
  {{ 2.0/3.0,-1.0, 0.0}, {-1.0, 2.0/3.0, 0.0}, { 0.0, 0.0, 0.0}},
  {{-1.0/3.0, 0.0, 0.0}, { 0.0, 2.0/3.0, 0.0}, { 0.0, 0.0, 0.0}},
  {{-1.0/3.0, 0.0, 0.0}, { 0.0, 2.0/3.0, 0.0}, { 0.0, 0.0, 0.0}},
  {{ 2.0/3.0,-1.0, 0.0}, {-1.0, 2.0/3.0, 0.0}, { 0.0, 0.0, 0.0}},
  {{ 2.0/3.0, 0.0, 0.0}, { 0.0,-1.0/3.0, 0.0}, { 0.0, 0.0, 0.0}},
  {{ 2.0/3.0, 1.0, 0.0}, { 1.0, 2.0/3.0, 0.0}, { 0.0, 0.0, 0.0}}};

#define w0 (16.0/36.0)
#define w1  (4.0/36.0)
#define w2  (1.0/36.0)

const double wv[NVEL] = {w0, w2, w1, w2, w1, w1, w2, w1, w2};
 
const double norm_[NVEL] = {1.0, 3.0, 3.0, 9.0/2.0, 9.0, 9.0/2.0, 1.0/4.0,
			    3.0/8.0, 3.0/8.0};

#define c0 0.0
#define c1 1.0
#define c2 2.0
#define c4 4.0
#define r3 (1.0/3.0)
#define t3 (2.0/3.0)
#define r4 (1.0/4.0)
#define r6 (1.0/6.0)

const double ma_[NVEL][NVEL] = {
  { c1, c1,  c1,  c1,  c1,  c1,  c1,  c1,  c1},
  { c0, c1,  c1,  c1,  c0,  c0, -c1, -c1, -c1},
  { c0, c1,  c0, -c1,  c1, -c1,  c1,  c0, -c1},
  {-r3, t3,  t3,  t3, -r3, -r3,  t3,  t3,  t3},
  { c0, c1,  c0, -c1,  c0,  c0, -c1,  c0,  c1},
  {-r3, t3, -r3,  t3,  t3,  t3,  t3, -r3,  t3},
  { c1, c4, -c2,  c4, -c2, -c2,  c4, -c2,  c4},
  { c0, c4, -c2,  c4,  c0,  c0, -c4,  c2, -c4},
  { c0, c4,  c0, -c4, -c2,  c2,  c4,  c0, -c4}};

#define wa (6.0/72.0)
#define wb (4.0/72.0)
#define wc (3.0/72.0)

const double mi_[NVEL][NVEL] = {
  {w0,  c0,  c0, -t3,  c0, -t3,  w1,  c0,  c0},
  {w2,  wa,  wa,  wa,  r4,  wa,  w2,  wc,  wc},
  {w1,  r3,  c0,  r3,  c0, -r6, -wb, -wa,  c0},
  {w2,  wa, -wa,  wa, -r4,  wa,  w2,  wc, -wc},
  {w1,  c0,  r3, -r6,  c0,  r3, -wb,  c0, -wa},
  {w1,  c0, -r3, -r6,  c0,  r3, -wb,  c0,  wa},
  {w2, -wa,  wa,  wa, -r4,  wa,  w2, -wc,  wc},
  {w1, -r3,  c0,  r3,  c0, -r6, -wb,  wa,  c0},
  {w2, -wa, -wa,  wa,  r4,  wa,  w2, -wc, -wc}};


const int xblocklen_cv[CVXBLOCK] = {3};
const int xdisp_fwd_cv[CVXBLOCK] = {1};
const int xdisp_bwd_cv[CVXBLOCK] = {6};

const int yblocklen_cv[CVYBLOCK] = {1, 1, 1};
const int ydisp_fwd_cv[CVYBLOCK] = {1, 4, 6};
const int ydisp_bwd_cv[CVYBLOCK] = {3, 5, 8};

const int zblocklen_cv[CVZBLOCK] = {0};
const int zdisp_fwd_cv[CVZBLOCK] = {0};
const int zdisp_bwd_cv[CVZBLOCK] = {0};

#endif

