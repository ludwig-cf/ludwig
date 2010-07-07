/*****************************************************************************
 *
 *  brownian.h
 *
 *  Brownian dynamics is disabled at the moment.
 *
 *  $Id: brownian.h,v 1.2.20.1 2010-07-07 09:02:36 kevin Exp $
 *
 *  Edinburgh Soft Matter and Statistical Physics Group and
 *  Edinburgh Parallel Computing Centre
 *
 *  Kevin Stratford (kevin@epcc.ed.ac.uk)
 *  (c) 2010 The University of Edinburgh
 *
 *****************************************************************************/

#ifndef BROWNIAN_H
#define BROWNIAN_H

void brownian_step_no_inertia(void);
void brownian_set_random(void);

#endif
