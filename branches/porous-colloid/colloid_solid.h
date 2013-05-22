/*****************************************************************************
 *
 *  colloid_solid_lubrication.h
 *
 *  Responsible for colloid-solid lubrication forces
 *  
 *
 *  $Id: colloid_solid_lubrication.h $
 *
 *  Juho
 *
 *****************************************************************************/

#ifndef COLLOID_SOLID_H
#define COLLOID_SOLID_H
 
#include <stdio.h>

void solid_lubrication(colloid_t * pc, double force[3]);
double cylinder_lubrication(const int dim, const double r[3], const double ah);

#endif
