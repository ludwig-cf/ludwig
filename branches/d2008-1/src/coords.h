/*****************************************************************************
 *
 *  coords.h
 *
 *  $Id: coords.h,v 1.2.4.1 2008-01-07 17:32:29 kevin Exp $
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

MPI_Comm cart_comm(void);

#endif
