/*****************************************************************************
 *
 *  coords.h
 *
 *  $Id: coords.h,v 1.3.16.5 2010-03-27 05:56:49 kevin Exp $
 *
 *  Edinburgh Soft Matter and Statistical Physics Group and
 *  Edinburgh Parallel Computing Centre
 *
 *  Kevin Stratford (kevin@epcc.ed.ac.uk)
 *  (c) The University of Edinburgh (2008)
 *
 *****************************************************************************/

#ifndef _COORDS_H
#define _COORDS_H

enum cartesian_directions {X, Y, Z};
enum cartesian_neighbours {FORWARD, BACKWARD};

extern int nhalo_;

void   coords_init(void);
void   coords_finish(void);
int    N_total(const int);
int    is_periodic(const int);
double L(const int);
double Lmin(const int);
int    cart_rank(void);
int    cart_size(const int);
int    cart_coords(const int);
int    cart_neighb(const int direction, const int dimension);
void   get_N_local(int []);
void   get_N_offset(int []);
int    get_site_index(const int, const int, const int);

MPI_Comm cart_comm(void);

void   coords_nlocal(int n[3]);
void   coords_nhalo_set(const int nhalo);
int    coords_nhalo(void);
void   coords_ntotal_set(const int n[3]);
void   coords_decomposition_set(const int p[3]);
void   coords_reorder_set(const int);
void   coords_periodicity_set(const int p[3]);
int    coords_nsites(void);
int    coords_index(const int ic, const int jc, const int kc);

#endif
