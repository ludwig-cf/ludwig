/*****************************************************************************
 *
 *  leesedwards_s.h
 *
 *  Edinburgh Soft Matter and Statistical Physics Group and
 *  Edinburgh Parallel Computing Centre
 *
 *  (c) 2015 The University of Edinburgh
 *
 *  Authors:
 *    Kevin Stratford (kevin@epcc.ed.ac.uk
 *
 *****************************************************************************/

#ifndef LEESEDWARDS_S_H
#define LEESEDWARDS_S_H

#include "leesedwards.h"

enum shear_type {LE_LINEAR, LE_OSCILLATORY};

struct le_s {
  coords_t * cs;            /* Reference to coordinate system */
  int nref;                 /* Reference count */

  /* Global parameters */
  int    nplanes;           /* Total number of planes */
  int    type;              /* Shear type */
  int    period;            /* for oscillatory */
  int    nt0;               /* time0 (input as integer) */
  double uy;                /* u[Y] for all planes */
  double dx_min;            /* Position first plane */
  double dx_sep;            /* Plane separation */
  double omega;             /* u_y = u_le cos (omega t) for oscillatory */  
  double time0;             /* time offset */

  /* Local parameters */
  int nlocal;               /* Number of planes local domain */
  int nxbuffer;             /* Size of buffer region in x */
  int index_real_nbuffer;
  int * index_buffer_to_real;
  int * index_real_to_buffer;
  int * buffer_duy;

  MPI_Comm  le_comm;        /* 1-d communicator */
  MPI_Comm  le_plane_comm;  /* 2-d communicator */
};

/* PENDING */
extern le_t * le_stat;

#endif
