/*****************************************************************************
 *
 *  wall.h
 *
 *  Interface for the wall information.
 *
 *  $Id: wall.h,v 1.4.4.1 2010-01-06 17:25:28 kevin Exp $
 *
 *  Kevin Stratford (kevin@epcc.ed.ac.uk)
 *
 *****************************************************************************/

#ifndef _WALL_H
#define _WALL_H

void wall_init(void);
void wall_bounce_back(void);
void wall_finish(void);
void wall_update(void);

void wall_accumulate_force(const double f[3]);
void wall_net_momentum(double g[3]);

#endif
