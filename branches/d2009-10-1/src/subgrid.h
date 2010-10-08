/*****************************************************************************
 *
 *  subgrid.h
 *
 *  $Id: subgrid.h,v 1.1.22.1 2010-10-08 15:06:16 kevin Exp $
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
