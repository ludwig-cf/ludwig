/*****************************************************************************
 *
 *  coords.h
 *
 *  $Id: coords.h,v 1.2 2006-10-12 14:09:18 kevin Exp $
 *
 *  Kevin Stratford (kevin@epcc.ed.ac.uk)
 *
 *****************************************************************************/

#ifndef _COORDS_H
#define _COORDS_H

enum cartesian_directions {X, Y, Z};
enum cartesian_neighbours {FORWARD, BACKWARD};

void   coords_init(void);
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

#ifdef _MPI_
MPI_Comm cart_comm(void);
#endif

#endif
