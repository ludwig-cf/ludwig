/*****************************************************************************
 *
 *  cartesian.h
 *
 *****************************************************************************/

#ifndef _CARTESIAN_H
#define _CARTESIAN_H

enum cartesian_direction {FORWARD, BACKWARD};

void cart_init(void);
int  cart_rank(void);
int  cart_size(const int);
int  cart_coords(const int);
int  cart_neighb(const int direction, const int dimension);
void get_N_local(int []);
void get_N_offset(int []);

#ifdef _MPI_
MPI_Comm cart_comm(void);
#endif

#endif
