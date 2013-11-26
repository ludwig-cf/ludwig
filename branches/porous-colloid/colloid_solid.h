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
double cylinder_lubrication_force(const double r[3], const double ah, double rhatij[3]);
double porous_wall_lubrication_force(const int dim, const double r[3], const double ah);
void cylinder_wall_soft_force(const double r[3], const double ah, double force[3]);
void cylinder_bottom_soft_force(const double r[3], const double ah, double force[3]);

#endif
