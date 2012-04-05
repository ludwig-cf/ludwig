/*****************************************************************************
 *
 *  colloid_file.c
 *
 *  Produce a file of colloid information suitable for reading into
 *  the main code in serial or in parallel.
 *
 *  First decide how many colloids are required and with what
 *  properties. The format of the output should be one 4-byte
 *  integer which is the number of colloids in the file,
 *  followed by this number of colloid_state_t structs. This
 *  is set out below.
 *
 *  The output is always in a single file, and can be ascii or
 *  binary. Use the standard functions in colloid.c to do the
 *  I/O as below.
 *
 *  Compile with, e.g., cc -I../src ../src/colloid.c colloid_file.c
 *
 *  The file ../src/colloid.h gives the full description of the
 *  colloid state.
 *
 *  $Id: colloid_file.c,v 1.2 2010-10-15 11:42:06 kevin Exp $
 *
 *  Edinburgh Soft Matter and Statistical Physics Group and
 *  Edinburgh Parallel Computing Centre
 *
 *  Kevin Stratford (kevin@epcc.ed.ac.uk)
 *  (c) 2010 The University of Edinburgh
 *
 *****************************************************************************/

#include <assert.h>
#include <stdio.h>
#include <stdlib.h>

#include "../src/colloid.h"

/*****************************************************************************
 *
 *  main
 *
 *****************************************************************************/

int main(int argc, char ** argv) {

  int index;
  int ncolloid = 1;  /* Total number of colloids */
  int binary = 0;    /* Switch for binary output */
 
  const char * filename = "config.cds00000000.001-001";
  FILE * fp;

  colloid_state_t state;        /* State to write */
  colloid_state_t * state_zero; /* State all zeros */

  fp = fopen(filename, "w");
  if (fp == NULL) {
    printf("Could not open %s\n", filename);
    exit(0);
  }

  /* Set the zero state using calloc() */
  state_zero = (colloid_state_t *) calloc(1, sizeof(colloid_state_t));
  assert(state_zero != NULL);

  /* Header: the format is consistent with ../src/cio.c */

  if (binary) {
    fwrite(&ncolloid, sizeof(int), 1, fp);
  }
  else {
    fprintf(fp, "%22d\n", ncolloid);
  }

  /* Colloids */

  for (index = 1; index <= ncolloid; index++) {

    state = *state_zero;
    state.index = index;

    /* Set the appropriate state here: the minimum requirements are
     *
     *    an index (as above, starting at 1 and MUST be unique) 
     *    an input radius a0
     *    a hydrodynamic radius ah
     *    a position r [Lmin < r < Lmin + ntotal]
     *    where Lmin is 0.5 by default. Everything else may be safely
     *    set to zero (as is done above). */

    state.a0 = 2.3;
    state.ah = 2.3;
    state.r[0] = 1.0; /* X position */
    state.r[1] = 1.0; /* Y position */
    state.r[2] = 1.0; /* Z position */

    if (binary) {
      colloid_state_write_binary(state, fp);
    }
    else {
      colloid_state_write_ascii(state, fp);
    }

  }

  fclose(fp);
  free(state_zero);

  return 0;
}
