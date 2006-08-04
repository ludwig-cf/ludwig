/***************************************************************************
 *
 *  lattice.h
 *
 *  Kevin Stratford (kevin@epcc.ed.ac.uk)
 *
 ***************************************************************************/

#ifndef _LATTICE_H_
#define _LATTICE_H_

#ifdef _D3Q19_
  #include "d3q19.h"
#else
  #include "d3q15.h"
#endif

void LATT_allocate_force(const int);
void LATT_allocate_phi(const int);
void LATT_allocate_sites(const int);
void latt_allocate_velocity(const int);

struct vector {double c[3];};

#endif /* _LATTICE_H_ */
