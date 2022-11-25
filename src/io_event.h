/*****************************************************************************
 *
 *  io_event.h
 *
 *  This is currently a placeholder to allow additional information
 *  to be provided to specific i/o read/write instances.
 *
 *
 *  Edinburgh Soft Matter and Statistical Physics Group and
 *  Edinburgh Parallel Computing Centre
 *
 *  (c) 2022 The University of Edinburgh
 *
 *  Kevin Stratford (kevin@epcc.ed.ac.uk)
 *
 *****************************************************************************/

#ifndef LUDWIG_IO_EVENT_H
#define LUDWIG_IO_EVENT_H

typedef struct io_event_s io_event_t;

struct io_event_s {
  int isconfig;        /* Is this a configuration step */
};

#endif
