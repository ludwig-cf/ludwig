/*****************************************************************************
 *
 *  stats_turbulent.h
 *
 *  $Id: stats_turbulent.h,v 1.1.2.1 2008-07-29 14:23:08 kevin Exp $
 *
 *  Edinburgh Soft Matter and Statistical Physics Group and
 *  Edinburgh Parallel Computing Centre
 *
 *  Kevin Stratford (kevin@epcc.ed.ac.uk)
 *  (c) 2008 The University of Edinburgh
 *
 *****************************************************************************/

#ifndef STATS_TURBULENT_H
#define STATS_TURBULENT_H

void stats_turbulent_init(void);
void stats_turbulent_finish(void);
void stats_turbulent_ubar_zero(void);
void stats_turbulent_ubar_accumulate(void);
void stats_turbulent_ubar_output(const char *);

#endif
