/*****************************************************************************
 *
 *  cio.h
 *
 *  $Id: cio.h,v 1.1.1.1.22.1 2010-03-27 11:18:16 kevin Exp $
 *
 *  Edinburgh Soft Matter and Statistical Physics Group and
 *  Edinburgh Parallel Computing Centre
 *
 *  Kevin Stratford (kevin@epcc.ed.ac.uk)
 *  (c) 2010 The University of Edinburgh
 *
 *****************************************************************************/

#ifndef CIO_H
#define CIO_H

void colloid_io_init(void);
void colliod_io_finish(void);

void colloid_io_read(const char * filename);
void colloid_io_write(const char * filename);

#endif
