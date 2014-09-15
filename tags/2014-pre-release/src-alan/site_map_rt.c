/*****************************************************************************
 *
 *  site_map_rt.c
 *
 *  Run time initialisation of the site map.
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

#include <stdio.h>
#include <string.h>

#include "pe.h"
#include "runtime.h"
#include "io_harness.h"
#include "site_map.h"

/*****************************************************************************
 *
 *  site_map_run_time
 *
 *****************************************************************************/

void site_map_run_time(void) {

  int n;
  int grd[3] = {1, 1, 1};
  char filename[FILENAME_MAX];
  char key[FILENAME_MAX];


  n = RUN_get_string_parameter("porous_media_file", filename, FILENAME_MAX);

  if (n == 0) {
    site_map_init_grid(grd);
  }
  else {

    /* A porous media file is present. */
    /* By default, this is expected to be a single serial file in BINARY
     * with site status information (but no wetting parameters). */

    info("\n");
    info("Porous media\n");
    info("------------\n");
    info("Porous media file requested: %s\n", filename);

    RUN_get_int_parameter_vector("porous_media_io_grid", grd);
    site_map_init_grid(grd);

    /* Always a single file read at the moment */
    io_info_single_file_set(io_info_site_map);

    RUN_get_string_parameter("porous_media_type", key, FILENAME_MAX);

    if (strcmp(key, "status_with_h") == 0) {
      info("Porous media type:           status_with_h\n");
      site_map_io_status_with_h();
    }
    else {
      info("Porous media type:           status_only\n");
    }

    RUN_get_string_parameter("porous_media_format", key, FILENAME_MAX);

    if (strcmp(key, "ASCII") == 0) {
      info("Porous media format:         ASCII\n");
      io_info_set_format_ascii(io_info_site_map);
    }
    else {
      info("Porous media format:         BINARY\n");
    }

    info("Porous media io grid:       %2d %2d %2d\n", grd[0], grd[1], grd[2]);
    info("Reading file...\n");

    io_read(filename, io_info_site_map);
    site_map_halo();
  }

  return;
}
