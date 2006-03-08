/*****************************************************************************
 *
 *  coords.h
 *
 *  Kevin Stratford (kevin@epcc.ed.ac.uk)
 *
 *****************************************************************************/

#ifndef _COORDS_H
#define _COORDS_H

enum cartesian_directions {X, Y, Z};

void   coords_init(void);
int    N_total(const int);
int    is_periodic(const int);
double L(const int);
double Lmin(const int);

#endif
