/*****************************************************************************
 *
 *  subgrid.h
 *
 *  $Id: subgrid.h,v 1.2 2010-10-15 12:40:03 kevin Exp $
 *
 *  Edinburgh Soft Matter and Statistical Physics Group and
 *  Edinburgh Parallel Computing Centre
 *
 *  Kevin Stratford (kevin@epcc.ed.ac.uk)
 *  (c) 2010 The University of Edinburgh
 *
 *****************************************************************************/

#ifndef SUBGRID_H
#define SUBGRID_H

void subgrid_update(void);
void subgrid_force_from_particles(void);
void subgrid_on_set(void);
int  subgrid_on(void);

#endif
