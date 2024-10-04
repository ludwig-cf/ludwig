/*****************************************************************************
 *
 *  map.c
 *
 *  Map of fluid lattice sites containing status (fluid, solid, etc)
 *  and with space for additional data such as wetting parameters.
 *
 *  Edinburgh Soft Matter and Statistical Physics Group and
 *  Edinburgh Parallel Computing Centre
 *
 *  (c) 2012-2024 The University of Edinburgh
 *
 *  Contributing authors:
 *  Kevin Stratford (kevin@epcc.ed.ac.uk)
 *
 *****************************************************************************/

#include <assert.h>
#include <limits.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "pe.h"
#include "coords.h"
#include "map.h"
#include "util.h"

static const int nchar_per_ascii_double_ = MAP_DATA_RECORD_LENGTH_ASCII;

int map_halo_impl(cs_t * cs, int nall, int na, void * buf,
		  MPI_Datatype mpidata);

/*****************************************************************************
 *
 *  map_create
 *
 *****************************************************************************/

int map_create(pe_t * pe, cs_t * cs, const map_options_t * options,
	       map_t ** map) {

  map_t * obj = NULL;

  assert(pe);
  assert(cs);
  assert(options);
  assert(map);

  obj = (map_t *) calloc(1, sizeof(map_t));
  if (obj == NULL) goto err;

  if (map_initialise(pe, cs, options, obj) != 0) goto err;

  *map = obj;

  return 0;

 err:

  if (obj) free(obj);
  return -1;
}

/*****************************************************************************
 *
 *  map_free
 *
 *****************************************************************************/

int map_free(map_t ** map) {

  assert(map);

  map_finalise(*map);
  free(*map);
  *map = NULL;

  return 0;
}

/*****************************************************************************
 *
 *  map_initialise
 *
 *****************************************************************************/

int map_initialise(pe_t * pe, cs_t * cs, const map_options_t * options,
		   map_t * map) {
  int ndevice = 0;

  assert(pe);
  assert(cs);
  assert(options);
  assert(map);

  if (map_options_valid(options) == 0) {
    pe_warn(pe, "map_initialise: options invalid\n");
    goto err;
  }

  map->pe = pe;
  map->cs = cs;
  map->nsite = cs->param->nsites;
  map->ndata = options->ndata;

  map->status = (char *) calloc(map->nsite, sizeof(char));
  assert(map->status);
  if (map->status == NULL) {
    pe_warn(pe, "map_initialise: calloc(map->status) failed\n");
    goto err;
  }

  /* Avoid overflow in allocation */
  if (INT_MAX/(map->nsite*imax(1, options->ndata)) < 1) {
    pe_warn(pe, "map_initialise: failure in int32_t allocation\n");
    goto err;
  }

  /* ndata may be zero, but avoid zero-sized allocations */

  if (map->ndata > 0) {
    map->data = (double *) calloc((size_t) map->ndata*map->nsite,
				  sizeof(double));
    assert(map->data);
    if (map->data == NULL) {
      pe_warn(pe, "map_initialise: calloc(map->data) failed\n");
      goto err;
    }
  }
    map->options = *options;

  {
    /* i/o metadata */
    int ifail = 0;
    io_element_t element = {0};

    /* Count for ascii is "%3d" for status, " %22.15e" for each data item
     * ie., 3 char, plus 23 char(s), plus a new line */
    io_element_t ascii = {
      .datatype = MPI_CHAR,
      .datasize = sizeof(char),
      .count    = 3 + nchar_per_ascii_double_*map->ndata + 1,
      .endian   = io_endianness()
    };
    io_element_t binary = {
      .datatype = MPI_CHAR,
      .datasize = sizeof(char),
      .count    = 1 + (int) sizeof(double)*map->ndata,
      .endian   = io_endianness()
    };

    map->ascii = ascii;
    map->binary = binary;
    map->filestub = options->filestub;

    if (options->iodata.input.iorformat == IO_RECORD_ASCII)  element = ascii;
    if (options->iodata.input.iorformat == IO_RECORD_BINARY) element = binary;
    ifail = io_metadata_initialise(cs, &options->iodata.input, &element,
				   &map->input);
    if (ifail != 0) {
      pe_warn(pe, "Failed to initialise map input metadata\n");
      goto err;
    }

    if (options->iodata.output.iorformat == IO_RECORD_ASCII)  element = ascii;
    if (options->iodata.output.iorformat == IO_RECORD_BINARY) element = binary;
    ifail = io_metadata_initialise(cs, &options->iodata.output, &element,
				   &map->output);
    if (ifail != 0) {
      pe_warn(pe, "Failed to initialise map output metadata\n");
      goto err;
    }
  }

  /* Allocate target copy of structure (or alias) */

  tdpAssert( tdpGetDeviceCount(&ndevice) );

  if (ndevice == 0) {
    map->target = map;
  }
  else {
    char * status = NULL;

    tdpAssert( tdpMalloc((void **) &map->target, sizeof(map_t)) );
    tdpAssert( tdpMemset(map->target, 0, sizeof(map_t)) );

    tdpAssert( tdpMalloc((void **) &status, map->nsite*sizeof(char)) );
    tdpAssert( tdpMemset(status, 0, map->nsite*sizeof(char)) );
    tdpAssert( tdpMemcpy(&map->target->status, &status, sizeof(char *),
			 tdpMemcpyHostToDevice) );

    tdpAssert( tdpMemcpy(&map->target->is_porous_media, &map->is_porous_media,
			 sizeof(int), tdpMemcpyHostToDevice) );
    tdpAssert( tdpMemcpy(&map->target->nsite, &map->nsite, sizeof(int),
			 tdpMemcpyHostToDevice) );
    tdpAssert( tdpMemcpy(&map->target->ndata, &map->ndata, sizeof(int),
			 tdpMemcpyHostToDevice) );

    /* Data */
    if (map->ndata > 0) {
      size_t nsz = (size_t) map->ndata*map->nsite*sizeof(double);
      double * data = NULL;
      tdpAssert( tdpMalloc((void **) &data, nsz));
      tdpAssert( tdpMemset(data, 0, nsz));
      tdpAssert( tdpMemcpy(&map->target->data, &data, sizeof(double *),
			   tdpMemcpyHostToDevice) );
    }
  }

  return 0;

 err:
  /* All failures are before any device memory is involved ... */
  if (map->input.cs) io_metadata_finalise(&map->input);
  if (map->output.cs) io_metadata_finalise(&map->output);
  if (map->data) free(map->data);
  if (map->status) free(map->status);

  *map = (map_t) {0};

  return -1;
}

/*****************************************************************************
 *
 *  map_finalise
 *
 *****************************************************************************/

int map_finalise(map_t * map) {

  assert(map);
  assert(map->status);

  int ndevice = 0;

  tdpAssert( tdpGetDeviceCount(&ndevice) );

  if (ndevice > 0) {
    char * status = NULL;

    if (map->ndata > 0) {
      double * data = NULL;
      tdpAssert( tdpMemcpy(&data, &map->target->data, sizeof(double *),
			   tdpMemcpyDeviceToHost) );
      tdpAssert( tdpFree(data) );
    }

    tdpAssert( tdpMemcpy(&status, &map->target->status, sizeof(char *),
			 tdpMemcpyDeviceToHost) );
    tdpAssert( tdpFree(status) );
    tdpAssert( tdpFree(map->target) );
  }

  io_metadata_finalise(&map->input);
  io_metadata_finalise(&map->output);

  if (map->data) free(map->data);
  if (map->status) free(map->status);

  *map = (map_t) {0};

  return 0;
}

/*****************************************************************************
 *
 *  map_memcpy
 *
 *****************************************************************************/

int map_memcpy(map_t * map, tdpMemcpyKind flag) {

  int ndevice;
  char * tmp;

  assert(map);

  tdpAssert( tdpGetDeviceCount(&ndevice) );

  if (ndevice == 0) {
    /* Ensure we alias */
    assert(map->target == map);
  }
  else {
    tdpAssert( tdpMemcpy(&tmp, &map->target->status, sizeof(char *),
			 tdpMemcpyDeviceToHost) );

    switch (flag) {
    case tdpMemcpyHostToDevice:
      tdpAssert( tdpMemcpy(tmp, map->status, map->nsite*sizeof(char), flag) );
      break;
    case tdpMemcpyDeviceToHost:
      tdpAssert( tdpMemcpy(map->status, tmp, map->nsite*sizeof(char), flag) );
      break;
    default:
      pe_fatal(map->pe, "Bad flag in map_memcpy()\n");
    }
  }

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

  assert(obj);

  map_halo_impl(obj->cs, obj->nsite, 1, obj->status, MPI_CHAR);

  if (obj->ndata) {
    map_halo_impl(obj->cs, obj->nsite, obj->ndata, obj->data, MPI_DOUBLE);
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

  cs_cart_comm(obj->cs, &comm);
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

  cs_nlocal(obj->cs, nlocal);
  vol_local = 0;

  for (ic = 1; ic <= nlocal[X]; ic++) {
    for (jc = 1; jc <= nlocal[Y]; jc++) {
      for (kc = 1; kc <= nlocal[Z]; kc++) {

	index = cs_index(obj->cs, ic, jc, kc);
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
 *  map_read_buf
 *
 *****************************************************************************/

int map_read_buf(map_t * map, int index, const char * buf) {

  assert(map);
  assert(buf);

  int iaddr = addr_rank0(map->nsite, index);
  memcpy(map->status + iaddr, buf, sizeof(char));

  for (int id = 0; id < map->ndata; id++) {
    size_t ioff = sizeof(char) + sizeof(double)*id;
    iaddr = addr_rank1(map->nsite, map->ndata, index, id);
    memcpy(map->data + iaddr, buf + ioff, sizeof(double));
  }

  return 0;
}

/*****************************************************************************
 *
 *  map_write_buf
 *
 *****************************************************************************/

int map_write_buf(map_t * map, int index, char * buf) {

  int ifail = 0;

  assert(map);
  assert(buf);

  int iaddr = addr_rank0(map->nsite, index);
  memcpy(buf, map->status + iaddr, sizeof(char));

  for (int id = 0; id < map->ndata; id++) {
    size_t ioff = sizeof(char) + sizeof(double)*id;
    iaddr = addr_rank1(map->nsite, map->ndata, index, id);
    memcpy(buf + ioff, map->data + iaddr, sizeof(double));
  }

  return ifail;
}

/*****************************************************************************
 *
 *  map_write_buf_ascii
 *
 *****************************************************************************/

int map_write_buf_ascii(map_t * map, int index, char * buf) {

  const int nbyte = nchar_per_ascii_double_;
  int ifail = 0;

  assert(map);
  assert(buf);

  {
    /* Status: three characters (plus the `\0`) */
    char tmp[BUFSIZ] = {0};
    int iaddr = addr_rank0(map->nsite, index);
    int nr = snprintf(tmp, 4, "%3d", map->status[iaddr]);
    if (nr != 3) ifail = -1;
    memcpy(buf, tmp, 3*sizeof(char));
  }

  for (int id = 0; id < map->ndata; id++) {
    char tmp[BUFSIZ] = {0};
    int iaddr = addr_rank1(map->nsite, map->ndata, index, id);
    int nr = snprintf(tmp, nbyte + 1, " %22.15e", map->data[iaddr]);
    if (nr != nbyte) ifail = -2;
    memcpy(buf + (3 + id*nbyte)*sizeof(char), tmp, nbyte*sizeof(char));
  }

  /* Add new line */
  {
    char tmp[BUFSIZ] = {0};
    int nr = snprintf(tmp, 2, "\n");
    if (nr != 1) ifail = -3;
    memcpy(buf + (3 + map->ndata*nbyte)*sizeof(char), tmp, sizeof(char));
  }

  return ifail;
}

/*****************************************************************************
 *
 *  map_read_buf_ascii
 *
 *****************************************************************************/

int map_read_buf_ascii(map_t * map, int index, const char * buf) {

  const int nbyte = nchar_per_ascii_double_;
  int ifail = 0;
  int nr = 0;

  assert(map);
  assert(buf);

  /* Status */
  {
    char tmp[BUFSIZ] = {0};
    int iaddr = addr_rank0(map->nsite, index);
    int status = -1;
    memcpy(tmp, buf, 3*sizeof(char));
    nr = sscanf(tmp, "%d", &status);
    if (nr != 1) ifail = -1;
    map->status[iaddr] = status;
  }

  /* Wetting data */
  for (int id = 0; id < map->ndata; id++) {
    char tmp[BUFSIZ] = {0};
    int iaddr = addr_rank1(map->nsite, map->ndata, index, id);
    memcpy(tmp, buf + (3 + id*nbyte)*sizeof(char), nbyte*sizeof(char));
    nr = sscanf(tmp, "%le", map->data + iaddr);
    if (nr != 1) ifail = 2;
  }

  return ifail;
}

/*****************************************************************************
 *
 *  map_io_aggr_pack
 *
 *****************************************************************************/

int map_io_aggr_pack(map_t * map, io_aggregator_t * aggr) {

  assert(map);
  assert(aggr);
  assert(aggr->buf);

  #pragma omp parallel
  {
    int iasc = map->options.iodata.output.iorformat == IO_RECORD_ASCII;
    int ibin = map->options.iodata.output.iorformat == IO_RECORD_BINARY;
    assert(iasc ^ ibin); /* one or other */

    #pragma omp for
    for (int ib = 0; ib < cs_limits_size(aggr->lim); ib++) {
      int ic = cs_limits_ic(aggr->lim, ib);
      int jc = cs_limits_jc(aggr->lim, ib);
      int kc = cs_limits_kc(aggr->lim, ib);

      /* Write at (ic, jc, kc) */
      int index = cs_index(map->cs, ic, jc, kc);
      size_t offset = ib*aggr->szelement;
      if (iasc) map_write_buf_ascii(map, index, aggr->buf + offset);
      if (ibin) map_write_buf(map, index, aggr->buf + offset);
    }
  }

  return 0;
}

/*****************************************************************************
 *
 *  map_io_aggr_unpack
 *
 *****************************************************************************/

int map_io_aggr_unpack(map_t * map, const io_aggregator_t * aggr) {

  assert(map);
  assert(aggr);
  assert(aggr->buf);

  #pragma omp parallel
  {
    int iasc = map->options.iodata.input.iorformat == IO_RECORD_ASCII;
    int ibin = map->options.iodata.input.iorformat == IO_RECORD_BINARY;
    assert(iasc ^ ibin);

    #pragma omp for
    for (int ib = 0; ib < cs_limits_size(aggr->lim); ib++) {
      int ic = cs_limits_ic(aggr->lim, ib);
      int jc = cs_limits_jc(aggr->lim, ib);
      int kc = cs_limits_kc(aggr->lim, ib);

      /* Read for (ic, jc, kc) */
      int index = cs_index(map->cs, ic, jc, kc);
      size_t offset = ib*aggr->szelement;
      if (iasc) map_read_buf_ascii(map, index, aggr->buf + offset);
      if (ibin) map_read_buf(map, index, aggr->buf + offset);
    }
  }

  return 0;
}

/*****************************************************************************
 *
 *  map_io_read
 *
 *****************************************************************************/

int map_io_read(map_t * map, int timestep) {

  int ifail = 0;

  assert(map);

  {
    io_metadata_t * meta = &map->input;
    io_impl_t * io = NULL;
    char filename[BUFSIZ] = {0};

    io_subfile_name(&meta->subfile, map->filestub, timestep, filename, BUFSIZ);

    ifail = io_impl_create(meta, &io);

    if (ifail == 0) {
      ifail = io->impl->read(io, filename);
      map_io_aggr_unpack(map, io->aggr);
      io->impl->free(&io);
    }
  }

  return ifail;
}

/*****************************************************************************
 *
 *  map_io_write
 *
 *****************************************************************************/

int map_io_write(map_t * map, int timestep) {

  int ifail = 0;

  assert(map);

  {
    const io_metadata_t * meta = &map->output;

    io_impl_t * io = NULL;
    char filename[BUFSIZ] = {0};

    io_subfile_name(&meta->subfile, map->filestub, timestep, filename, BUFSIZ);
    ifail = io_impl_create(meta, &io);

    if (ifail == 0) {
      map_io_aggr_pack(map, io->aggr);
      ifail = io->impl->write(io, filename);
      io->impl->free(&io);
    }
  }

  return ifail;
}

/*****************************************************************************
 *
 *  map_halo_impl
 *
 *****************************************************************************/

int map_halo_impl(cs_t * cs, int nall, int na, void * mbuf,
		  MPI_Datatype mpidata) {
  size_t sz = 0;
  int ic, jc, kc;
  int ia, index;
  int nh;
  int ireal, ihalo;
  int icount, nsend;
  int pforw, pback;
  int nlocal[3];
  int nhcomm = 0;

  unsigned char * buf;
  unsigned char * sendforw;
  unsigned char * sendback;
  unsigned char * recvforw;
  unsigned char * recvback;

  MPI_Comm comm;
  MPI_Request req[4];
  MPI_Status status[2];

  const int tagf = 2015;
  const int tagb = 2016;

  assert(cs);
  assert(mbuf);
  assert(mpidata == MPI_CHAR || mpidata == MPI_DOUBLE);

  buf = (unsigned char *) mbuf;

  comm = cs->commcart;
  if (mpidata == MPI_CHAR) sz = sizeof(char);
  if (mpidata == MPI_DOUBLE) sz = sizeof(double);
  assert(sz != 0);

  cs_nlocal(cs, nlocal);
  cs_nhalo(cs, &nhcomm);

  /* X-direction */

  nsend = nhcomm*na*nlocal[Y]*nlocal[Z];
  sendforw = (unsigned char *) malloc(nsend*sz);
  sendback = (unsigned char *) malloc(nsend*sz);
  recvforw = (unsigned char *) malloc(nsend*sz);
  recvback = (unsigned char *) malloc(nsend*sz);
  assert(sendforw && sendback);
  assert(recvforw && recvback);
  if (sendforw == NULL) pe_fatal(cs->pe, "malloc(sendforw) failed\n");
  if (sendback == NULL) pe_fatal(cs->pe, "malloc(sendback) failed\n");
  if (recvforw == NULL) pe_fatal(cs->pe, "malloc(recvforw) failed\n");
  if (recvback == NULL) pe_fatal(cs->pe, "malloc(recvback) failed\n");

  /* Load send buffers */

  icount = 0;

  for (nh = 0; nh < nhcomm; nh++) {
    for (jc = 1; jc <= nlocal[Y]; jc++) {
      for (kc = 1; kc <= nlocal[Z]; kc++) {
	for (ia = 0; ia < na; ia++) {
	  /* Backward going... */
	  index = cs_index(cs, 1 + nh, jc, kc);
	  ireal = addr_rank1(nall, na, index, ia);
	  memcpy(sendback + icount*sz, buf + ireal*sz, sz);
	  /* ...and forward going. */
	  index = cs_index(cs, nlocal[X] - nh, jc, kc);
	  ireal = addr_rank1(nall, na, index, ia);
	  memcpy(sendforw + icount*sz, buf + ireal*sz, sz);
	  icount += 1;
	}
      }
    }
  }

  assert(icount == nsend);

  if (cs->param->mpi_cartsz[X] == 1) {
    memcpy(recvback, sendforw, nsend*sz);
    memcpy(recvforw, sendback, nsend*sz);
    req[2] = MPI_REQUEST_NULL;
    req[3] = MPI_REQUEST_NULL;
  }
  else {
    pforw = cs->mpi_cart_neighbours[CS_FORW][X];
    pback = cs->mpi_cart_neighbours[CS_BACK][X];
    MPI_Irecv(recvforw, nsend, mpidata, pforw, tagb, comm, req);
    MPI_Irecv(recvback, nsend, mpidata, pback, tagf, comm, req + 1);
    MPI_Issend(sendback, nsend, mpidata, pback, tagb, comm, req + 2);
    MPI_Issend(sendforw, nsend, mpidata, pforw, tagf, comm, req + 3);
    /* Wait for receives */
    MPI_Waitall(2, req, status);
  }

  /* Unload */

  icount = 0;

  for (nh = 0; nh < nhcomm; nh++) {
    for (jc = 1; jc <= nlocal[Y]; jc++) {
      for (kc = 1; kc <= nlocal[Z]; kc++) {
	for (ia = 0; ia < na; ia++) {
	  index = cs_index(cs, nlocal[X] + 1 + nh, jc, kc);
	  ihalo = addr_rank1(nall, na, index, ia);
	  memcpy(buf + ihalo*sz, recvforw + icount*sz, sz);
	  index = cs_index(cs, 0 - nh, jc, kc);
	  ihalo = addr_rank1(nall, na, index, ia);
	  memcpy(buf + ihalo*sz, recvback + icount*sz, sz);
	  icount += 1;
	}
      }
    }
  }

  assert(icount == nsend);

  free(recvback);
  free(recvforw);

  MPI_Waitall(2, req + 2, status);

  free(sendback);
  free(sendforw);

  /* Y direction */

  nsend = nhcomm*na*(nlocal[X] + 2*nhcomm)*nlocal[Z];
  sendforw = (unsigned char *) malloc(nsend*sz);
  sendback = (unsigned char *) malloc(nsend*sz);
  recvforw = (unsigned char *) malloc(nsend*sz);
  recvback = (unsigned char *) malloc(nsend*sz);
  assert(sendforw && sendback);
  assert(recvforw && recvback);
  if (sendforw == NULL) pe_fatal(cs->pe, "malloc(sendforw) failed\n");
  if (sendback == NULL) pe_fatal(cs->pe, "malloc(sendback) failed\n");
  if (recvforw == NULL) pe_fatal(cs->pe, "malloc(recvforw) failed\n");
  if (recvback == NULL) pe_fatal(cs->pe, "malloc(recvback) failed\n");

  /* Load buffers */

  icount = 0;

  for (nh = 0; nh < nhcomm; nh++) {
    for (ic = 1 - nhcomm; ic <= nlocal[X] + nhcomm; ic++) {
      for (kc = 1; kc <= nlocal[Z]; kc++) {
	for (ia = 0; ia < na; ia++) {
	  index = cs_index(cs, ic, 1 + nh, kc);
	  ireal = addr_rank1(nall, na, index, ia);
	  memcpy(sendback + icount*sz, buf + ireal*sz, sz);
	  index = cs_index(cs, ic, nlocal[Y] - nh, kc);
	  ireal = addr_rank1(nall, na, index, ia);
	  memcpy(sendforw + icount*sz, buf + ireal*sz, sz);
	  icount += 1;
	}
      }
    }
  }

  assert(icount == nsend);

  if (cs->param->mpi_cartsz[Y] == 1) {
    memcpy(recvback, sendforw, nsend*sz);
    memcpy(recvforw, sendback, nsend*sz);
    req[2] = MPI_REQUEST_NULL;
    req[3] = MPI_REQUEST_NULL;
  }
  else {
    pforw = cs->mpi_cart_neighbours[CS_FORW][Y];
    pback = cs->mpi_cart_neighbours[CS_BACK][Y];
    MPI_Irecv(recvforw, nsend, mpidata, pforw, tagb, comm, req);
    MPI_Irecv(recvback, nsend, mpidata, pback, tagf, comm, req + 1);
    MPI_Issend(sendback, nsend, mpidata, pback, tagb, comm, req + 2);
    MPI_Issend(sendforw, nsend, mpidata, pforw, tagf, comm, req + 3);
    /* Wait for receives */
    MPI_Waitall(2, req, status);
  }

  /* Unload */

  icount = 0;

  for (nh = 0; nh < nhcomm; nh++) {
    for (ic = 1 - nhcomm; ic <= nlocal[X] + nhcomm; ic++) {
      for (kc = 1; kc <= nlocal[Z]; kc++) {
	for (ia = 0; ia < na; ia++) {
	  index = cs_index(cs, ic, 0 - nh, kc);
	  ihalo = addr_rank1(nall, na, index, ia);
	  memcpy(buf + ihalo*sz, recvback + icount*sz, sz);
	  index = cs_index(cs, ic, nlocal[Y] + 1 + nh, kc);
	  ihalo = addr_rank1(nall, na, index, ia);
	  memcpy(buf + ihalo*sz, recvforw + icount*sz, sz);
	  icount += 1;
	}
      }
    }
  }

  assert(icount == nsend);

  free(recvback);
  free(recvforw);

  MPI_Waitall(2, req + 2, status);

  free(sendback);
  free(sendforw);

  /* Z direction */

  nsend = nhcomm*na*(nlocal[X] + 2*nhcomm)*(nlocal[Y] + 2*nhcomm);
  sendforw = (unsigned char *) malloc(nsend*sz);
  sendback = (unsigned char *) malloc(nsend*sz);
  recvforw = (unsigned char *) malloc(nsend*sz);
  recvback = (unsigned char *) malloc(nsend*sz);
  assert(sendforw && sendback);
  assert(recvforw && recvback);
  if (sendforw == NULL) pe_fatal(cs->pe, "malloc(sendforw) failed\n");
  if (sendback == NULL) pe_fatal(cs->pe, "malloc(sendback) failed\n");
  if (recvforw == NULL) pe_fatal(cs->pe, "malloc(recvforw) failed\n");
  if (recvback == NULL) pe_fatal(cs->pe, "malloc(recvback) failed\n");

  /* Load */
  /* Some adjustment in the load required for 2d systems (X-Y) */

  icount = 0;

  for (nh = 0; nh < nhcomm; nh++) {
    for (ic = 1 - nhcomm; ic <= nlocal[X] + nhcomm; ic++) {
      for (jc = 1 - nhcomm; jc <= nlocal[Y] + nhcomm; jc++) {
	for (ia = 0; ia < na; ia++) {
	  kc = imin(1 + nh, nlocal[Z]);
	  index = cs_index(cs, ic, jc, kc);
	  ireal = addr_rank1(nall, na, index, ia);
	  memcpy(sendback + icount*sz, buf + ireal*sz, sz);
	  kc = imax(nlocal[Z] - nh, 1);
	  index = cs_index(cs, ic, jc, kc);
	  ireal = addr_rank1(nall, na, index, ia);
	  memcpy(sendforw + icount*sz, buf + ireal*sz, sz);
	  icount += 1;
	}
      }
    }
  }

  assert(icount == nsend);

  if (cs->param->mpi_cartsz[Z] == 1) {
    memcpy(recvback, sendforw, nsend*sz);
    memcpy(recvforw, sendback, nsend*sz);
    req[2] = MPI_REQUEST_NULL;
    req[3] = MPI_REQUEST_NULL;
  }
  else {
    pforw = cs->mpi_cart_neighbours[CS_FORW][Z];
    pback = cs->mpi_cart_neighbours[CS_BACK][Z];
    MPI_Irecv(recvforw, nsend, mpidata, pforw, tagb, comm, req);
    MPI_Irecv(recvback, nsend, mpidata, pback, tagf, comm, req + 1);
    MPI_Issend(sendback, nsend, mpidata, pback, tagb, comm, req + 2);
    MPI_Issend(sendforw, nsend, mpidata, pforw, tagf, comm, req + 3);
    /* Wait before unloading */
    MPI_Waitall(2, req, status);
  }

  /* Unload */

  icount = 0;

  for (nh = 0; nh < nhcomm; nh++) {
    for (ic = 1 - nhcomm; ic <= nlocal[X] + nhcomm; ic++) {
      for (jc = 1 - nhcomm; jc <= nlocal[Y] + nhcomm; jc++) {
	for (ia = 0; ia < na; ia++) {
	  index = cs_index(cs, ic, jc, 0 - nh);
	  ihalo = addr_rank1(nall, na, index, ia);
	  memcpy(buf + ihalo*sz, recvback + icount*sz, sz);
	  index = cs_index(cs, ic, jc, nlocal[Z] + 1 + nh);
	  ihalo = addr_rank1(nall, na, index, ia);
	  memcpy(buf + ihalo*sz, recvforw + icount*sz, sz);
	  icount += 1;
	}
      }
    }
  }

  assert(icount == nsend);

  free(recvback);
  free(recvforw);

  MPI_Waitall(2, req + 2, status);

  free(sendback);
  free(sendforw);

  return 0;
}
