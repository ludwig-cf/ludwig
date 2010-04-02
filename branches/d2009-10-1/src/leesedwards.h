/*****************************************************************************
 *
 *  leesedwards.h
 *
 *  $Id: leesedwards.h,v 1.5.4.1 2010-04-02 07:56:02 kevin Exp $
 *
 *  Edinburgh Soft Matter and Statistical Physics Group and
 *  Edinburgh Parallel Computing Centre
 *
 *  Kevin Stratford (kevin@epcc.ed.ac.uk)
 *  (c) 2010 The University of Edinburgh
 *
 *****************************************************************************/

#ifndef LEESEDWARDS_H
#define LEESEDWARDS_H

void le_init(void);
void le_finish(void);

int le_get_nxbuffer(void);
int le_index_real_to_buffer(const int, const int);
int le_index_buffer_to_real(const int);
int le_site_index(const int, const int, const int);
int le_plane_location(const int);
int le_get_nplane_total(void);
int le_get_nplane_local(void);
int le_nsites(void);

double    le_buffer_displacement(const int, const double);
double    le_get_block_uy(int);
double    le_get_steady_uy(const int); 
double    le_plane_uy(const double);
double    le_plane_uy_max(void);
double    le_shear_rate(void);
MPI_Comm  le_communicator(void);
void      le_displacement_ranks(const double, int[2], int[2]);
void      le_jstart_to_ranks(const int, int send[2], int recv[2]);
void      le_set_oscillatory(const double);
void      le_set_nplane_total(const int nplane);
void      le_set_plane_uymax(const double uy);

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
