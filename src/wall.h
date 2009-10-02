/*****************************************************************************
 *
 *  wall.h
 *
 *  Interface for the wall information.
 *
 *  $Id: wall.h,v 1.4 2009-10-02 13:50:18 kevin Exp $
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
void wall_force(void);

#endif
