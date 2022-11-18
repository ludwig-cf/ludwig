/*****************************************************************************
 *
 *  io_subfile.c
 *
 *  File decomposition description.
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
#include <string.h>

#include "coords_s.h"
#include "io_subfile.h"

/*****************************************************************************
 *
 *  io_subfile_default
 *
 *  Return decomposition with iogrid {1,1,1}, which cannot fail.
 *
 *****************************************************************************/

io_subfile_t io_subfile_default(cs_t * cs) {

  io_subfile_t subfile = {0};
  int iogrid[3] = {1, 1, 1};

  io_subfile_create(cs, iogrid, &subfile);

  return subfile;
}

/*****************************************************************************
 *
 *  io_subfile_create
 *
 *  Returns zero on success. Non-zero indicates no decompsoition is
 *  available.
 *
 *****************************************************************************/

int io_subfile_create(cs_t * cs, const int iogrid[3], io_subfile_t * subfile) {

  MPI_Comm parent = MPI_COMM_NULL;

  assert(cs);
  assert(subfile);

  /* Check we can make a decomposition ... */

  for (int i = 0; i < 3; i++) {
    if (cs->param->mpi_cartsz[i] % iogrid[i] != 0) goto err;
  }

  /* Note ndims is fixed at the moment ...*/

  subfile->ndims = 3;
  cs_cart_comm(cs, &parent);

  for (int i = 0; i < 3; i++) {
    /* Some integer arithmetic to form blocks */
    int isz = cs->param->mpi_cartsz[i];
    int icoord = cs->param->mpi_cartcoords[i];
    int ioffset = icoord / (isz/iogrid[i]);

    subfile->iosize[i] = iogrid[i];
    subfile->coords[i] = iogrid[i]*icoord/isz;
    subfile->offset[i] = cs->listnoffset[i][ioffset];

    /* sizes must be accumulated allowing for non-uniform decomposition */
    subfile->sizes[i] = 0;
    for (int j = ioffset; j < ioffset + (isz/iogrid[i]); j++) {
      subfile->sizes[i] += cs->listnlocal[i][j];
    }
  }

  /* Index/nfile */

  subfile->nfile = iogrid[X]*iogrid[Y]*iogrid[Z];
  subfile->index = subfile->coords[X]
                 + subfile->coords[Y]*iogrid[X]
                 + subfile->coords[Z]*iogrid[X]*iogrid[Y];
  return 0;

 err:
  return -1;
}

/*****************************************************************************
 *
 *  io_subfile_to_json
 *
 *****************************************************************************/

int io_subfile_to_json(const io_subfile_t * subfile, cJSON ** json) {

  int ifail = 0;

  if (json == NULL || *json != NULL) {
    ifail = -1;
  }
  else {

    cJSON * myjson = cJSON_CreateObject();
    cJSON * iosize = cJSON_CreateIntArray(subfile->iosize, 3);
    cJSON * coords = cJSON_CreateIntArray(subfile->coords, 3);
    cJSON * offset = cJSON_CreateIntArray(subfile->offset, 3);
    cJSON * sizes  = cJSON_CreateIntArray(subfile->sizes,  3);

    cJSON_AddNumberToObject(myjson, "Number of files", subfile->nfile);
    cJSON_AddNumberToObject(myjson, "File index", subfile->index);
    cJSON_AddItemToObject(myjson, "Topology", iosize);
    cJSON_AddItemToObject(myjson, "Coordinate", coords);

    cJSON_AddNumberToObject(myjson, "Data ndims", subfile->ndims);
    cJSON_AddItemToObject(myjson, "File size (sites)", sizes);
    cJSON_AddItemToObject(myjson, "File offset (sites)", offset);

    *json = myjson;
  }

  return ifail;
}

/*****************************************************************************
 *
 *  io_subfile_from_json
 *
 *  Return zero indicates success.
 *
 *****************************************************************************/

int io_subfile_from_json(const cJSON * json, io_subfile_t * subfile) {

  int ifail = 0;

  if (json == NULL || subfile == NULL) {
    ifail = -1;
  }
  else {
    cJSON * nfile = cJSON_GetObjectItemCaseSensitive(json, "Number of files");
    cJSON * index = cJSON_GetObjectItemCaseSensitive(json, "File index");
    cJSON * ndims = cJSON_GetObjectItemCaseSensitive(json, "Data ndims");
    cJSON * iosz  = cJSON_GetObjectItemCaseSensitive(json, "Topology");
    cJSON * coord = cJSON_GetObjectItemCaseSensitive(json, "Coordinate");
    cJSON * sizes = cJSON_GetObjectItemCaseSensitive(json, "File size (sites)");
    cJSON * offst = cJSON_GetObjectItemCaseSensitive(json, "File offset (sites)");

    if (nfile) subfile->nfile = cJSON_GetNumberValue(nfile);
    if (index) subfile->index = cJSON_GetNumberValue(index);
    if (ndims) subfile->ndims = cJSON_GetNumberValue(ndims);

    if (nfile == NULL) ifail += 1;
    if (index == NULL) ifail += 2;
    if (ndims == NULL) ifail += 4;

    if (3 != util_json_to_int_array(iosz,  subfile->iosize, 3)) ifail +=  8;
    if (3 != util_json_to_int_array(coord, subfile->coords, 3)) ifail += 16;
    if (3 != util_json_to_int_array(sizes, subfile->sizes,  3)) ifail += 32;
    if (3 != util_json_to_int_array(offst, subfile->offset, 3)) ifail += 64;
  }

  return ifail;
}

/*****************************************************************************
 *
 *  io_subfile_name
 *
 *  Return a standardised data file name e.g., "stub-tttttttt.iii-nnn"
 *  being read as timestep t, part iii of nnn parts.
 *
 *  Parts are counted in natural numbers 1, 2, 3... which is (1+index).
 *
 *  bufsz is the maximum length of filename.
 *
 *****************************************************************************/

int io_subfile_name(const io_subfile_t * subfile, const char * stub, int it,
		    char * filename, size_t bufsz) {

  int ifail = -1;

  assert(subfile);
  assert(stub);
  assert(filename);

  assert(0 <= it && it < 1000*1000*1000); /* Format error */

  if (bufsz > strlen(stub) + 8) {
    sprintf(filename, "%s-%9.9d.%3.3d-%3.3d", stub, it, 1 + subfile->index,
	    subfile->nfile);
    ifail = 0;
  }

  return ifail;
}
