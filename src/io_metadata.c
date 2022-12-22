/*****************************************************************************
 *
 *  io_metadata.c
 *
 *  Lattice quantity i/o metadata. 
 *
 *  Aggregate information on i/o, that is the run time options, the data
 *  element type, and the lattice structure, and provide a communicator.
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
#include <stdlib.h>

#include "io_metadata.h"
#include "util_fopen.h"

/*****************************************************************************
 *
 *  io_metadata_create
 *
 *  Note this can fail and return non-zero if there is no decomposition
 *  available for the given coordinate system/options.
 *
 *  Success returns 0.
 *
 *****************************************************************************/

int io_metadata_create(cs_t * cs,
		       const io_options_t * options,
		       const io_element_t * element,
		       io_metadata_t ** metadata) {

  io_metadata_t * meta = NULL;

  assert(cs);
  assert(options);
  assert(element);
  assert(metadata);

  meta = (io_metadata_t *) calloc(1, sizeof(io_metadata_t));
  assert(meta);
  if (meta == NULL) goto err;

  if (io_metadata_initialise(cs, options, element, meta) != 0) goto err;

  *metadata = meta;

  return 0;

 err:

  if (meta) free(meta);
  return -1;
}

/*****************************************************************************
 *
 *  io_metadata_free
 *
 *****************************************************************************/

int io_metadata_free(io_metadata_t ** metadata) {

  assert(metadata);

  io_metadata_finalise(*metadata);
  free(*metadata);
  *metadata = NULL;

  return 0;
}

/*****************************************************************************
 *
 *  io_metadata_initialise
 *
 *  Generate once only per run as the cost of the MPI_Comm_split()
 *  should not be repeated.
 *
 *****************************************************************************/

int io_metadata_initialise(cs_t * cs,
			   const io_options_t * options,
			   const io_element_t * element,
			   io_metadata_t * meta) {
  assert(cs);
  assert(options);
  assert(element);
  assert(meta);

  *meta = (io_metadata_t) {0};

  meta->cs = cs;
  cs_cart_comm(cs, &meta->parent);

  {
    /* Store the limits as a convenience */
    int nlocal[3] = {0};
    cs_nlocal(cs, nlocal);
    meta->limits.imin = 1; meta->limits.imax = nlocal[X];
    meta->limits.jmin = 1; meta->limits.jmax = nlocal[Y];
    meta->limits.kmin = 1; meta->limits.kmax = nlocal[Z];
  }

  meta->options = *options;
  meta->element = *element;

  {
    /* Must have a decomposition... */
    int ifail = io_subfile_create(cs, options->iogrid, &meta->subfile);
    if (ifail != 0) return -1;
  }

  /* Split communicator for the file. */
  /* One could revisit assigning the rank ... */
  {
    int rank = -1;
    MPI_Comm_rank(meta->parent, &rank);
    MPI_Comm_split(meta->parent, meta->subfile.index, rank, &meta->comm);
  }

  return 0;
}

/*****************************************************************************
 *
 *  io_metadata_finalise
 *
 *****************************************************************************/

int io_metadata_finalise(io_metadata_t * meta) {

  assert(meta);

  MPI_Comm_free(&meta->comm);
  *meta = (io_metadata_t) {0};
  meta->comm = MPI_COMM_NULL;

  return 0;
}

/*****************************************************************************
 *
 *  io_metadata_to_json
 *
 *  The "non-MPI" content can be translated meaningfully.
 *
 *****************************************************************************/

int io_metadata_to_json(const io_metadata_t * meta, cJSON ** json) {

  int ifail = 0;

  assert(meta);

  if (json == NULL || *json != NULL) {
    ifail = -1;
  }
  else {
    cJSON * myjson = cJSON_CreateObject();

    {
      /* coords (including Lees Edwards information) */
      cJSON * jtmp = NULL;
      ifail = cs_to_json(meta->cs, &jtmp);
      if (ifail == 0) cJSON_AddItemToObject(myjson, "coords", jtmp);
    }
    {
      /* options */
      cJSON * jtmp = NULL;
      ifail = io_options_to_json(&meta->options, &jtmp);
      if (ifail == 0) cJSON_AddItemToObject(myjson, "io_options", jtmp);
    }
    {
      /* element */
      cJSON * jtmp = NULL;
      ifail = io_element_to_json(&meta->element, &jtmp);
      if (ifail == 0) cJSON_AddItemToObject(myjson, "io_element", jtmp);
    }
    {
      /* subfile */
      cJSON * jtmp = NULL;
      ifail = io_subfile_to_json(&meta->subfile, &jtmp);
      if (ifail == 0) cJSON_AddItemToObject(myjson, "io_subfile", jtmp);
    }
    *json = myjson;
  }

  return ifail;
}

/*****************************************************************************
 *
 *  io_metadata_from_json
 *
 *  Assumes we have an existing coordinate system.
 *  FIXME: However, Lees Edwards options cs->leopts is initialised here.
 *  The question of initialising cs_t from JSON is PENDING.
 *
 *****************************************************************************/

int io_metadata_from_json(cs_t * cs, const cJSON * json, io_metadata_t * m) {

  int ifail = 0;

  assert(cs);
  assert(json);
  assert(cs->leopts.nplanes == 0); /* No exsiting information */

  if (m == NULL) {
    ifail = -1;
  }
  else {
    /* Generate options, element, from json, and initialise ... */
    /* This initialises cs->leopts */
    io_options_t options      = {IO_MODE_INVALID};
    io_element_t element      = {0};

    cJSON * jopt = cJSON_GetObjectItemCaseSensitive(json, "io_options");
    cJSON * jelm = cJSON_GetObjectItemCaseSensitive(json, "io_element");
    ifail = io_options_from_json(jopt, &options);
    ifail = io_element_from_json(jelm, &element);
    ifail = io_metadata_initialise(cs, &options, &element, m);
  }

  return ifail;
}

/*****************************************************************************
 *
 *  io_metadata_write
 *
 *  Driver to write to file with an optional extra comment block.
 *
 *****************************************************************************/

int io_metadata_write(const io_metadata_t * metadata,
		      const char * stub,
		      const cJSON * comments) {

  cJSON * json = NULL;
  char filename[BUFSIZ] = {0};

  assert(metadata);
  assert(stub);

  if (metadata->iswriten) return 0;

  /* Generate a json with the header inserted (gets deleted below) */

  io_metadata_to_json(metadata, &json);
  if (comments) {
    cJSON * jtmp = cJSON_Duplicate(comments, 1);
    cJSON_AddItemToObject(json, "comments", jtmp);
  }

  /* The extension uses indices in natural numbers 001-002 etc. */
  sprintf(filename, "%s-metadata.%3.3d-%3.3d", stub,
	  1 + metadata->subfile.index, metadata->subfile.nfile);

  {
    /* Write to file (root only) */
    int rank = -1;
    MPI_Comm_rank(metadata->comm, &rank);
    if (rank == 0) {
      FILE * fp = NULL;
      char * str = cJSON_Print(json);
      fp = util_fopen(filename, "w");
      fprintf(fp, "%s\n", str);
      fclose(fp);
      free(str);
    }
  }

  cJSON_Delete(json);

  return 0;
}
