
/* KS note: this is superceded by io-harness. Colloid I/O
 * remains to be updated. */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "pe.h"
#include "runtime.h"
#include "coords.h"

#include "utilities.h"
#include "communicate.h"

IO_Param     io_grp;      /* Parameters of current IO group */

int          input_format;     /* Default input format is ASCII */
int          output_format;    /* Default output format is binary */

char         input_config[256] = "EMPTY";
char         output_config[256] = "config.out";

MPI_Comm     IO_Comm;     /* Communicator for parallel IO groups */

/*---------------------------------------------------------------------------*\
 * void COM_init( int argc, char **argv )                                    *
 *                                                                           *
 * Initialises communication routines                                        *
 *                                                                           *
 * Version: 2.0                                                              *
 * Options: _MPI_, _TRACE_                                                   *
 *                                                                           *
 * Arguments                                                                 *
 * - argc:    same as the main() routine's argc                              *
 * - argv:    same as the main() routine's argv                              *
 *                                                                           *
 * Last Updated: 06/01/2002 by JCD                                           *
\*---------------------------------------------------------------------------*/

void COM_init() {

  char tmp[256];

#ifdef _MPI_ /* Parallel (MPI) section */

  /* Set-up parallel IO parameters (rank and root) */

  io_grp.n_io = 1; /* Default */
  RUN_get_int_parameter("n_io_nodes", &(io_grp.n_io));

  io_grp.size = pe_size() / io_grp.n_io;

  if((cart_rank()%io_grp.size) == 0) {
    io_grp.root = 1;
  }
  else {
    io_grp.root = 0;
  }

  /* Set-up filename suffix for each parallel IO file */

  io_grp.file_ext = (char *) malloc(16*sizeof(char));

  if (io_grp.file_ext == NULL) fatal("malloc(io_grp.file_ext) failed\n");

  io_grp.index = cart_rank()/io_grp.size + 1;   /* Start IO indexing at 1 */
  sprintf(io_grp.file_ext, ".%d-%d", io_grp.n_io, io_grp.index);

  /* Create communicator for each IO group, and get rank within IO group */
  MPI_Comm_split(cart_comm(), io_grp.index, cart_rank(), &IO_Comm);
  MPI_Comm_rank(IO_Comm, &io_grp.rank);

  MPI_Barrier(cart_comm());

#else /* Serial section */

  /* Serial definition of io_grp (used in cio) */
  io_grp.root  = 1;
  io_grp.n_io  = 1;
  io_grp.size  = 1;
  io_grp.index = 0;
  io_grp.rank  = 0;
  io_grp.file_ext = (char *) malloc(16*sizeof(char));
  if (io_grp.file_ext == NULL) fatal("malloc(io_grp.file_ext) failed\n");
  sprintf(io_grp.file_ext, "%s", ""); /* Nothing required in serial*/

#endif /* _MPI_ */

  /* Everybody */

  /* I/O */
  strcpy(input_config, "EMPTY");
  strcpy(output_config, "config.out");

  input_format = BINARY;
  output_format = BINARY;

  RUN_get_string_parameter("input_config", input_config, 256);
  RUN_get_string_parameter("output_config", output_config, 256);

  RUN_get_string_parameter("input_format", tmp, 256);
  if (strncmp("ASCII",  tmp, 5) == 0 ) input_format = ASCII;
  if (strncmp("ASCII_SERIAL",  tmp, 12) == 0 ) input_format = ASCII_SERIAL;
  if (strncmp("BINARY", tmp, 6) == 0 ) input_format = BINARY;

  RUN_get_string_parameter("output_format", tmp, 256);
  if (strncmp("ASCII",  tmp, 5) == 0 ) output_format = ASCII;
  if (strncmp("BINARY", tmp, 6) == 0 ) output_format = BINARY;

  /* Set input routines: point to ASCII/binary routine depending on current 
     settings */
}

/*****************************************************************************
 *
 *  get_output_config_filename
 *
 *  Return conifguration file name stub for time "step"
 *
 *****************************************************************************/

void get_output_config_filename(char * stub, const int step) {

  sprintf(stub, "%s%8.8d", output_config, step);

  return;
}

/*****************************************************************************
 *
 *  get_input_config_filename
 *
 *  Return configuration file name (where for historical reasons,
 *  input_config holds the whole name). "step is ignored.
 *
 *****************************************************************************/

void get_input_config_filename(char * stub, const int step) {

  /* But use this... */
  sprintf(stub, "%s", input_config);

  return;
}
