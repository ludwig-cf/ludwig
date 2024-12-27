/*****************************************************************************
 *
 *  io_event.c
 *
 *  Used to form reports on individual i/o events.
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

#include <assert.h>

#include "io_event.h"

/*****************************************************************************
 *
 *  io_event_record
 *
 *****************************************************************************/

int io_event_record(io_event_t * event, io_event_record_t iorec) {

  assert(event);
  assert(0 <= iorec && iorec <= IO_EVENT_MAX);

  event->time[iorec] = MPI_Wtime();

  return 0;
}

/*****************************************************************************
 *
 *  io_event_report
 *
 *  Report for an i/o event in parallel.
 *  The aggregation report might want the local data size (currently total).
 *  The file write is the total data size of the file.
 *
 *  Some refinement might be wanted for multiple files.
 *
 *****************************************************************************/

int io_event_report(io_event_t * event, const io_metadata_t * metadata,
		    const char * name, io_event_record_t iorec) {

  assert(event);
  assert(metadata);
  assert(name);

  switch (iorec) {
  case IO_EVENT_READ:
    io_event_report_read(event, metadata, name);
    break;
  case IO_EVENT_WRITE:
    io_event_report_write(event, metadata, name);
    break;
  default:
    /* Internal error. */
    pe_exit(metadata->cs->pe, "Bad io event in report\n");
  }

  return 0;
}

/*****************************************************************************
 *
 *  io_event_report_read
 *
 *  Report for an i/o event in parallel.
 *  The aggregation report might want the local data size (currently total).
 *  The file write is the total data size of the file.
 *
 *  Some refinement might be wanted for multiple files.
 *
 *****************************************************************************/

int io_event_report_read(io_event_t * event, const io_metadata_t * metadata,
			 const char * name) {

  assert(event);
  assert(metadata);
  assert(name);

  /* End of event (for reporting purposes) */
  event->time[IO_EVENT_REPORT] = MPI_Wtime();

  if (metadata->options.report) {

    pe_t * pe = metadata->cs->pe;

    const char * units = NULL;
    double dunit6 = 1.0e+06; /* Units of data size are MB */
    double dunit9 = 1.0e+09; /* Units of data size are GB */

    /* Times (we assume these have been collected correctly! */
    /* Read, then disaggregate, then report */

    double tr = event->time[IO_EVENT_DISAGGR] - event->time[IO_EVENT_READ];
    double ta = event->time[IO_EVENT_REPORT] - event->time[IO_EVENT_DISAGGR];

    /* Record size and total file size. */

    double dr = metadata->element.datasize*metadata->element.count;
    double ds = metadata->subfile.sizes[X]*metadata->subfile.sizes[Y]*
                metadata->subfile.sizes[Z];
    double db = dr*ds;

    if (db > dunit9) {
      /* Use GB */
      units = "GB";
      db = db/dunit9;
    }
    else {
      /* Use MB */
      units = "MB";
      db = db/dunit6;
    }
    pe_info(pe, "- %10s read          %7.3f %2s in %7.3f seconds\n",
	    name, db, units, tr);
    pe_info(pe, "- %10s disaggregated %7.3f %2s in %7.3f seconds\n",
	    name, db, units, ta);
    pe_info(pe, "- %10s rate          %7.3f GB per second\n",
	    name, dr*ds/dunit9/tr);
  }

  return 0;
}

/*****************************************************************************
 *
 *  io_event_report_write
 *
 *  The aggregation report might want the local data size (currently total).
 *  The file write is the total data size of the file.
 *
 *  Some refinement might be wanted for multiple files.
 *
 *****************************************************************************/

int io_event_report_write(io_event_t * event, const io_metadata_t * metadata,
			  const char * name) {

  assert(event);
  assert(metadata);
  assert(name);


  /* End of event (for reporting purposes) */
  event->time[IO_EVENT_REPORT] = MPI_Wtime();

  if (metadata->options.report) {

    pe_t * pe = metadata->cs->pe;

    const char * units = NULL;
    double dunit6 = 1.0e+06; /* Units of data size are MB */
    double dunit9 = 1.0e+09; /* Units of data size are GB */

    /* Times (we assume these have been collected correctly! */
    /* Write: aggr is first, write is second, report last */

    double ta = event->time[IO_EVENT_WRITE]  - event->time[IO_EVENT_AGGR];
    double tw = event->time[IO_EVENT_REPORT] - event->time[IO_EVENT_WRITE];

    /* Record size and total file size. */

    double dr = metadata->element.datasize*metadata->element.count;
    double ds = metadata->subfile.sizes[X]*metadata->subfile.sizes[Y]*
                metadata->subfile.sizes[Z];
    double db = dr*ds;

    if (db > dunit9) {
      /* Use GB */
      units = "GB";
      db = db/dunit9;
    }
    else {
      /* Use MB */
      units = "MB";
      db = db/dunit6;
    }
    pe_info(pe, "- %10s aggregated %7.3f %2s in %7.3f seconds\n",
	    name, db, units, ta);
    pe_info(pe, "- %10s wrote      %7.3f %2s in %7.3f seconds\n",
	    name, db, units, tw);
    pe_info(pe, "- %10s rate       %7.3f GB per second\n",
	    name, dr*ds/dunit9/tw);
  }

  return 0;
}
