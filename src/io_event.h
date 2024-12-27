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
 *  (c) 2022-2024 The University of Edinburgh
 *
 *  Kevin Stratford (kevin@epcc.ed.ac.uk)
 *
 *****************************************************************************/

#ifndef LUDWIG_IO_EVENT_H
#define LUDWIG_IO_EVENT_H

#include "io_metadata.h"

enum io_event_record_enum {
  IO_EVENT_AGGR = 0,
  IO_EVENT_DISAGGR,
  IO_EVENT_READ,
  IO_EVENT_WRITE,
  IO_EVENT_REPORT,
  IO_EVENT_MAX
};

typedef enum io_event_record_enum io_event_record_t;
typedef struct io_event_s io_event_t;

struct io_event_s {
  const char * extra_name;      /* Extra metadata name */
  cJSON * extra_json;           /* Extra JSON section */
  double time[IO_EVENT_MAX];    /* MPI_Wtime()s */
};

int io_event_record(io_event_t * event, io_event_record_t iorec);
int io_event_report(io_event_t * event, const io_metadata_t * metadata,
		    const char * name, io_event_record_t iorec);
int io_event_report_read(io_event_t * event, const io_metadata_t * metadata,
			 const char * name);
int io_event_report_write(io_event_t * event, const io_metadata_t * metadata,
			  const char * name);

#endif
