/*****************************************************************************
 *
 *  stats_calibration.c
 *
 *  $Id$
 *
 *  Edinburgh Soft Matter and Statistical Physics Group and
 *  Edinburgh Parallel Computing Centre
 *
 *  Kevin Stratford (kevin@epcc.ed.ac.uk)
 *  (c) 2011 The University of Edinburgh
 *
 *****************************************************************************/

#ifndef STATS_CALIBRATION_H
#define STATS_CALIBRATION_H

void stats_calibration_init(void);
void stats_calibration_accululate(int ntimestep);
void stats_calibration_finish(void);

#endif
