/*****************************************************************************
 *
 *  colloid_sums.h
 *
 *  $Id: colloid_sums.h,v 1.1.2.2 2010-09-23 17:09:12 kevin Exp $
 *
 *  Edinburgh Soft Matter and Statistical Physics Group and
 *  Edinburgh Parallel Computing Centre
 *
 *  Kevin Stratford (kevin@epcc.ed.ac.uk)
 *  (c) 2010 The University of Edinburgh
 *
 *****************************************************************************/

#ifndef COLLOID_SUMS_H
#define COLLOID_SUMS_H

enum message_type {COLLOID_SUM_STRUCTURE = 0,
		   COLLOID_SUM_DYNAMICS = 1,
		   COLLOID_SUM_ACTIVE = 2};

void colloid_sums_halo(const int message_type);
void colloid_sums_dim(const int dim, const int message_type);

#endif
