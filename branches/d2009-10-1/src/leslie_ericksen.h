/*****************************************************************************
 *
 *  leslie_ericksen.h
 *
 *  $Id: leslie_ericksen.h,v 1.1.2.2 2010-03-26 08:39:12 kevin Exp $
 *
 *  Edinburgh Soft Matter and Statistical Physics Group
 *  and Edinburgh Parallel Computing Centre
 *
 *  Kevin Stratford (kevin@epcc.ed.ac.uk)
 *  (c) 2010 The University of Edinburgh
 *
 *****************************************************************************/

#ifndef LESLIE_ERICKSEN_H
#define LESLIE_ERICKSEN_H

void leslie_ericksen_update(void);
void leslie_ericksen_gamma_set(const double gamma);
void leslie_ericksen_swim_set(const double gamma);

#endif
