/*****************************************************************************
 *
 *  wall.h
 *
 *  Interface for the wall information.
 *
 *  $Id: wall.h,v 1.2 2006-10-12 14:09:18 kevin Exp $
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

#endif
