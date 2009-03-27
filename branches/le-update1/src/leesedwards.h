/*****************************************************************************
 *
 *  leesedwards.h
 *
 *  $Id: leesedwards.h,v 1.3.4.4 2009-03-27 16:35:50 kevin Exp $
 *
 *  Edinburgh Soft Matter and Statistical Physics Group and
 *  Edinburgh Parallel Computing Centre
 *  (c) The University of Edinburgh (2009)
 *  Kevin Stratford (kevin@epcc.ed.ac.uk)
 *
 *****************************************************************************/

#ifndef _LEESEDWARDS_H
#define _LEESEDWARDS_H

void le_init(void);
void le_finish(void);

int le_get_nxbuffer(void);
int le_index_real_to_buffer(const int, const int);
int le_index_buffer_to_real(const int);
int le_site_index(const int, const int, const int);
int le_plane_location(const int);
int le_get_nplane_total(void);
int le_get_nplane_local(void);

double    le_buffer_displacement(const int, const double);
double    le_get_block_uy(int);
double    le_get_steady_uy(const int); 
double    le_plane_uy(const double);
double    le_plane_uy_max(void);
double    le_shear_rate(void);
MPI_Comm  le_communicator(void);
void      le_displacement_ranks(const double, int[2], int[2]);
void      le_set_oscillatory(const double);

/* Address macro. For performance purposes, -DNDEBUG replaces
 * calls to ADDR, ie., le_site_index() with a macro, which requires that
 * the local system size be available as "nlocal". nhalo_ is
 * const, and available from coords.h, as are {X,Y,Z}  */

#ifdef NDEBUG
#define ADDR(ic,jc,kc) \
((nlocal[Y]+2*nhalo_)*(nlocal[Z]+2*nhalo_)*(nhalo_+(ic)-1) + \
                      (nlocal[Z]+2*nhalo_)*(nhalo_+(jc)-1) + \
                                           (nhalo_+(kc)-1))
#else
#define ADDR le_site_index
#endif

#endif
