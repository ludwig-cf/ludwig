/*****************************************************************************
 *
 *  map.c
 *
 *  Map of fluid lattice sites containing status (fluid, solid, etc)
 *  and with space for additional data such as wetting parameters.
 *
 *  $Id$
 *
 *  Edinburgh Soft Matter and Statistical Physics Group and
 *  Edinburgh Parallel Computing Centre
 *
 *  Kevin Stratford (kevin@epcc.ed.ac.uk)
 *  (c) 2012 The University of Edinburgh
 *
 *****************************************************************************/

#include <assert.h>
#include <stdio.h>
#include <stdlib.h>

#include "pe.h"
#include "coords.h"
#include "coords_field.h"
#include "map_s.h"

static int map_read(FILE * fp, int index, void * self);
static int map_write(FILE * fp, int index, void * self);
static int map_read_ascii(FILE * fp, int index, void * self);
static int map_write_ascii(FILE * fp, int index, void * self);

/*****************************************************************************
 *
 *  map_create
 *
 *  Allocate the map object, the status field, and any additional
 *  data required.
 *
 *****************************************************************************/

int map_create(int ndata, map_t ** pobj) {

  int nsites;
  int nhalo;
  map_t * obj = NULL;

  assert(ndata >= 0);
  assert(pobj);

  nsites = coords_nsites();
  nhalo = coords_nhalo();

  obj = calloc(1, sizeof(map_t));
  if (obj == NULL) fatal("calloc(map_t) failed\n");

  obj->status = calloc(nsites, sizeof(char));
  if (obj->status == NULL) fatal("calloc(map->status) failed\n");

 /* allocate target copy */
  //targetCalloc((void **) &obj->t_status, nsites*sizeof(char));

  obj->ndata = ndata;

  /* Could be zero-sized array */

  obj->data = calloc(ndata*nsites, sizeof(double));
  if (ndata > 0 && obj->data == NULL) fatal("calloc(map->data) failed\n");

  coords_field_init_mpi_indexed(nhalo, 1, MPI_CHAR, obj->halostatus);
  if (obj->ndata) {
    coords_field_init_mpi_indexed(nhalo, obj->ndata, MPI_DOUBLE,
				  obj->halodata);
  }


  /* allocate target copy of structure */
  targetMalloc((void**) &(obj->tcopy),sizeof(map_t));

  /* allocate data space on target */
  char* tmpptr;
  map_t* t_obj = obj->tcopy;
  targetCalloc((void**) &tmpptr,nsites*sizeof(char));
  copyToTarget(&(t_obj->status),&tmpptr,sizeof(char*)); 

  copyToTarget(&(t_obj->is_porous_media),&(obj->is_porous_media),sizeof(int)); 
  copyToTarget(&(t_obj->ndata),&(obj->ndata),sizeof(int)); 

  obj->t_status= tmpptr; //DEPRECATED direct access to target data.



  *pobj = obj;

  return 0;
}

/*****************************************************************************
 *
 *  map_free
 *
 *****************************************************************************/

void map_free(map_t * obj) {

  assert(obj);

  MPI_Type_free(&obj->halostatus[X]);
  MPI_Type_free(&obj->halostatus[Y]);
  MPI_Type_free(&obj->halostatus[Z]);

  if (obj->ndata > 0) {
    free(obj->data);
    MPI_Type_free(&obj->halodata[X]);
    MPI_Type_free(&obj->halodata[Y]);
    MPI_Type_free(&obj->halodata[Z]);
  }

  if (obj->info) io_info_destroy(obj->info);
  free(obj->status);
  //  targetFree(obj->t_status);

 if (obj->tcopy) {

    //free data space on target 
    char* tmpptr;
    map_t* t_obj = obj->tcopy;
    copyFromTarget(&tmpptr,&(t_obj->status),sizeof(char*)); 
    targetFree(tmpptr);
    
    //free target copy of structure
    targetFree(obj->tcopy);
  }

  free(obj);

  return;
}

/*****************************************************************************
 *
 *  map_init_io_info
 *
 *****************************************************************************/

int map_init_io_info(map_t * obj, int grid[3], int form_in, int form_out) {

  size_t sz;

  assert(obj);
  assert(grid);
  assert(obj->info == NULL);

  obj->info = io_info_create_with_grid(grid);
  if (obj->info == NULL) fatal("io_info_create(map) failed\n");

  io_info_set_name(obj->info, "map");
  io_info_write_set(obj->info, IO_FORMAT_BINARY, map_write);
  io_info_write_set(obj->info, IO_FORMAT_ASCII, map_write_ascii);
  io_info_read_set(obj->info, IO_FORMAT_BINARY, map_read);
  io_info_read_set(obj->info, IO_FORMAT_ASCII, map_read_ascii);

  sz = sizeof(char) + obj->ndata*sizeof(double);
  io_info_set_bytesize(obj->info, sz);

  io_info_format_set(obj->info, form_in, form_out);
  io_info_metadata_filestub_set(obj->info, "map");

  return 0;
}

/*****************************************************************************
 *
 *  map_io_info
 *
 *****************************************************************************/

int map_io_info(map_t * obj, io_info_t ** info) {

  assert(obj);
  assert(info);

  *info = obj->info;

  return 0;
}

/*****************************************************************************
 *
 *  map_halo
 *
 *  This should only be required when porous media read from file.
 *
 *****************************************************************************/

int map_halo(map_t * obj) {

  int nhalo;
  assert(obj);

  nhalo = coords_nhalo();

  coords_field_halo(nhalo, 1, obj->status, MPI_CHAR, obj->halostatus);
  if (obj->ndata) {
    coords_field_halo(nhalo, obj->ndata, obj->data, MPI_DOUBLE,
		      obj->halodata);
  }

  return 0;
}

/*****************************************************************************
 *
 *  map_status
 *
 *  Return map status as an integer.
 *
 *****************************************************************************/

int map_status(map_t * obj, int index, int * status) {

  assert(obj);
  assert(status);

  *status = (int) obj->status[index];

  return 0;
}

/*****************************************************************************
 *
 *  map_status_set
 *
 *****************************************************************************/

int map_status_set(map_t * obj, int index, int status) {

  assert(obj);
  assert(status >= 0);
  assert(status < MAP_STATUS_MAX);

  obj->status[index] = status;

  return 0;
}

/*****************************************************************************
 *
 *  map_ndata
 *
 *****************************************************************************/

int map_ndata(map_t * obj, int * ndata) {

  assert(obj);
  assert(ndata);

  *ndata = obj->ndata;

  return 0;
}

/*****************************************************************************
 *
 *  map_data
 *
 *****************************************************************************/

int map_data(map_t * obj, int index, double * data) {

  int n;

  assert(obj);
  assert(data);

  for (n = 0; n < obj->ndata; n++) {
    data[n] = obj->data[obj->ndata*index + n];
  }

  return 0;
}

/*****************************************************************************
 *
 *  map_data_set
 *
 *****************************************************************************/

int map_data_set(map_t * obj, int index, double * data) {

  int n;

  assert(obj);
  assert(data);

  for (n = 0; n < obj->ndata; n++) {
    obj->data[obj->ndata*index + n] = data[n];
  }

  return 0;
}

/*****************************************************************************
 *
 *  map_pm
 *
 *****************************************************************************/

int map_pm(map_t * obj, int * flag) {

  assert(obj);
  assert(flag);

  *flag = obj->is_porous_media;

  return 0;
}

/*****************************************************************************
 *
 *  map_pm_set
 *
 *****************************************************************************/

int map_pm_set(map_t * obj, int flag) {

  assert(obj);

  obj->is_porous_media = flag;

  return 0;
}

/*****************************************************************************
 *
 *  map_volume_allreduce
 *
 *****************************************************************************/

int map_volume_allreduce(map_t * obj, int status, int * volume) {

  int vol_local;
  MPI_Comm comm;

  assert(obj);
  assert(volume);

  map_volume_local(obj, status, &vol_local);

  comm = cart_comm();
  MPI_Allreduce(&vol_local, volume, 1, MPI_INT, MPI_SUM, comm);

  return 0;
}

/*****************************************************************************
 *
 *  map_volume_local
 *
 *****************************************************************************/

int map_volume_local(map_t * obj, int status_wanted, int * volume) {

  int nlocal[3];
  int ic, jc, kc, index;
  int vol_local;
  int status;

  assert(obj);
  assert(volume);

  coords_nlocal(nlocal);
  vol_local = 0;

  for (ic = 1; ic <= nlocal[X]; ic++) {
    for (jc = 1; jc <= nlocal[Y]; jc++) {
      for (kc = 1; kc <= nlocal[Z]; kc++) {

	index = coords_index(ic, jc, kc);
	map_status(obj, index, &status);
	if (status == status_wanted) vol_local += 1;
      }
    }
  }

  *volume = vol_local;

  return 0;
}

/*****************************************************************************
 *
 *  map_write
 *
 *****************************************************************************/

static int map_write(FILE * fp, int index, void * self) {

  int nw;
  int indexf;
  map_t * obj = self;

  assert(fp);
  assert(obj);

  nw = fwrite(&obj->status[index], sizeof(char), 1, fp);
  if (nw != 1) fatal("fwrite(map->status) failed\n");

  if (obj->ndata > 0) {
    coords_field_index(index, 0, obj->ndata, &indexf);
    nw = fwrite(&obj->data[indexf], sizeof(double), obj->ndata, fp);
    if (nw != obj->ndata) fatal("fwrite(map->data) failed\n");
  }

  return 0;
}

/*****************************************************************************
 *
 *  map_read
 *
 *****************************************************************************/

static int map_read(FILE * fp, int index, void * self) {

  int nr;
  int indexf;
  map_t * obj = self;

  assert(fp);
  assert(obj);

  nr = fread(&obj->status[index], sizeof(char), 1, fp);
  if (nr != 1) fatal("fread(map->status) failed");

  if (obj->ndata > 0) {
    coords_field_index(index, 0, obj->ndata, &indexf);
    nr = fread(&obj->data[indexf], sizeof(double), obj->ndata, fp);
    if (nr != obj->ndata) fatal("fread(map->data) failed\n");
  }

  return 0;
}

/*****************************************************************************
 *
 *  map_write_ascii
 *
 *****************************************************************************/

static int map_write_ascii(FILE * fp, int index, void * self) {

  int n, nw;
  int indexf;
  int status;
  map_t * obj = self;

  assert(fp);
  assert(obj);

  status = obj->status[index];
  assert(status < 99); /* Fixed format check. */

  nw = fprintf(fp, "%2d", status);
  if (nw != 2) fatal("fprintf(map->status) failed\n");

  for (n = 0; n < obj->ndata; n++) {
    coords_field_index(index, n, obj->ndata, &indexf);
    fprintf(fp, " %22.15e", obj->data[indexf]);
  }

  nw = fprintf(fp, "\n");
  if (nw != 1) fatal("fprintf(map) failed\n");

  return 0;
}

/*****************************************************************************
 *
 *  map_read_ascii
 *
 *****************************************************************************/

static int map_read_ascii(FILE * fp, int index, void * self) {

  int n, nr;
  int indexf;
  int status;
  map_t * obj = self;

  assert(fp);
  assert(obj);

  nr = fscanf(fp, "%2d", &status);
  if (nr != 1) fatal("fscanf(map->status) failed\n");
  obj->status[index] = status;

  for (n = 0; n < obj->ndata; n++) {
    coords_field_index(index, n, obj->ndata, &indexf);
    nr = fscanf(fp, " %le", obj->data + indexf);
    if (nr != 1) fatal("fscanf(map->data[%d]) failed\n", n);
  }

  return 0;
}
