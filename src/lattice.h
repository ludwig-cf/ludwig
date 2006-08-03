/***************************************************************************
 *
 *  lattice.h
 *
 *  Kevin Stratford (kevin@epcc.ed.ac.uk)
 *
 ***************************************************************************/

#ifndef _LATTICE_H_
#define _LATTICE_H_

void LATT_allocate_force(const int);
void LATT_allocate_phi(const int);
void LATT_allocate_sites(const int);

struct vector {double c[3];};

#endif /* _LATTICE_H_ */
