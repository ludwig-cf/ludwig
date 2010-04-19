/*****************************************************************************
 *
 *  coords.h
 *
 *  $Id: coords.h,v 1.3.16.8 2010-04-19 10:31:29 kevin Exp $
 *
 *  Edinburgh Soft Matter and Statistical Physics Group and
 *  Edinburgh Parallel Computing Centre
 *
 *  Kevin Stratford (kevin@epcc.ed.ac.uk)
 *  (c) 2010 The University of Edinburgh
 *
 *****************************************************************************/

#ifndef COORDS_H
#define COORDS_H

enum cartesian_directions {X, Y, Z};
enum cartesian_neighbours {FORWARD, BACKWARD};
enum upper_triangle {XX, XY, XZ, YY, YZ, ZZ};

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

MPI_Comm cart_comm(void);

void   coords_nlocal(int n[3]);
void   coords_nlocal_offset(int n[3]);
void   coords_nhalo_set(const int nhalo);
int    coords_nhalo(void);
void   coords_ntotal_set(const int n[3]);
void   coords_decomposition_set(const int p[3]);
void   coords_reorder_set(const int);
void   coords_periodicity_set(const int p[3]);
int    coords_nsites(void);
int    coords_index(const int ic, const int jc, const int kc);
void   coords_minimum_distance(const double r1[3], const double r2[3],
			       double r12[3]);
void   coords_index_to_ijk(const int index, int coords[3]);

#endif
