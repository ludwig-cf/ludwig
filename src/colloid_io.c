/*****************************************************************************
 *
 *  colloid_io.c
 *
 *  Colloid parallel I/O driver.
 *
 *  Edinburgh Soft Matter and Statistical Physics Group and
 *  Edinburgh Parallel Computing Centre
 *
 *  (c) 2010-2020 The University of Edinburgh
 *
 *  Contributing authors:
 *  Kevin Stratford (kevin@epcc.ed.ac.uk)
 *
 *****************************************************************************/

#include <assert.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "pe.h"
#include "coords.h"
#include "colloid_io.h"

struct colloid_io_s {
  int n_io;                      /* Number of parallel IO group */
  int size;                      /* Size (in PEs) of each IO group */
  int index;                     /* Index of current IO group */
  int rank;                      /* Rank of PE in IO group */
  int single_file_read;          /* 'serial' input flag */
  int nd[3];                     /* Cartesian dimensions */
  int coords[3];                 /* Cartesian position of this group */

  MPI_Comm comm;                 /* Communicator */
  MPI_Comm xcomm;                /* Communicator between groups */

  int (* f_header_write) (FILE * fp, int nfile);
  int (* f_header_read)  (FILE * fp, int * nfile);
  int (* f_list_write)   (colloid_io_t * cio, int, int, int, FILE * fp);
  int (* f_list_read)    (colloid_io_t * cio, int ndata, FILE * fp);
  int (* f_buffer_write) (FILE * fp, int nc, colloid_state_t * buf);

  pe_t * pe;                     /* Parallel environment */
  cs_t * cs;                     /* Coordinate system */
  colloids_info_t * info;        /* Keep a reference to this */
};

int colloid_io_pack_buffer(colloid_io_t * cio, int nc, colloid_state_t * buf);
int colloid_io_write_buffer_ascii(FILE * fp, int nc, colloid_state_t * buf);
int colloid_io_write_buffer_binary(FILE * fp, int nc, colloid_state_t * buf);
int colloid_io_count_colloids(colloid_io_t * cio, int * ngroup);

static int colloid_io_read_header_binary(FILE * fp, int * nfile);
static int colloid_io_read_header_ascii(FILE * fp, int * nfile);
static int colloid_io_write_header_binary(FILE *, int nfile);
static int colloid_io_write_header_ascii(FILE *, int nfile);

static int colloid_io_read_list_ascii(colloid_io_t * cio, int nc, FILE * fp);
static int colloid_io_read_list_binary(colloid_io_t * cio, int nc, FILE * fp);
static int colloid_io_write_list_ascii(colloid_io_t * cio,
				       int ic , int jc, int kc, FILE * fp);
static int colloid_io_write_list_binary(colloid_io_t * cio,
					int ic , int jc, int kc, FILE * fp);

static int colloid_io_filename(colloid_io_t * cio, char * filename,
			       const char * stub);
static int colloid_io_check_read(colloid_io_t * cio, int ngroup);

/*****************************************************************************
 *
 *  colloid_io_create
 *
 *  This split is the same as that which occurs in io_harness, which
 *  means the spatial decomposition is the same as that of the
 *  lattice quantities.
 *
 *  The io_grid[3] input is that requested; we return what actually
 *  gets allocated, given the constraints.
 *
 *****************************************************************************/

int colloid_io_create(pe_t * pe, cs_t * cs, int io_grid[3],
		      colloids_info_t * info,
		      colloid_io_t ** pcio) {
  int ia;
  int ncart;
  int mpicartsz[3];
  int mpicoords[3];
  colloid_io_t * obj = NULL;
  MPI_Comm cartcomm;

  assert(pe);
  assert(cs);
  assert(info);
  assert(pcio);

  obj = (colloid_io_t*) calloc(1, sizeof(colloid_io_t));
  assert(obj);
  if (obj == NULL) pe_fatal(pe, "calloc(colloid_io_t) failed\n");

  obj->pe = pe;
  obj->cs = cs;
  obj->n_io = 1;

  cs_cart_comm(cs, &cartcomm);
  cs_cartsz(cs, mpicartsz);
  cs_cart_coords(cs, mpicoords);

  for (ia = 0; ia < 3; ia++) {
    ncart = mpicartsz[ia];

    if (io_grid[ia] > ncart) io_grid[ia] = ncart;

    if (ncart % io_grid[ia] != 0) {
      pe_fatal(pe, "Bad colloid io grid (dim %d = %d)\n", ia, io_grid[ia]);
    }

    obj->nd[ia] = io_grid[ia];
    obj->n_io *= io_grid[ia];
    obj->coords[ia] = io_grid[ia]*mpicoords[ia]/ncart;
  }

  obj->index = obj->coords[X] + obj->coords[Y]*io_grid[X]
    + obj->coords[Z]*io_grid[X]*io_grid[Y];

  MPI_Comm_split(cartcomm, obj->index, cs_cart_rank(cs), &obj->comm);
  MPI_Comm_rank(obj->comm, &obj->rank);
  MPI_Comm_size(obj->comm, &obj->size);

  /* 'Cross' communicator between same rank in different groups. */

  MPI_Comm_split(cartcomm, obj->rank, cs_cart_rank(cs), &obj->xcomm);

  obj->info = info;
  *pcio = obj;

  return 0;
}

/*****************************************************************************
 *
 *  colloid_io_finish
 *
 *****************************************************************************/

int colloid_io_free(colloid_io_t * cio) {

  assert(cio);

  MPI_Comm_free(&cio->xcomm);
  MPI_Comm_free(&cio->comm);

  free(cio);

  return 0;
}

/*****************************************************************************
 *
 *  colloid_io_info_set
 *
 *****************************************************************************/

int colloid_io_info_set(colloid_io_t * cio, colloids_info_t * info) {

  assert(cio);
  assert(info);

  cio->info = info;

  return 0;
}

/*****************************************************************************
 *
 *  colloid_io_info
 *
 *  Send current state in human-readable form to given file.
 *
 *****************************************************************************/

int colloid_io_info(colloid_io_t * cio) {

  char fin[BUFSIZ];
  char fout[BUFSIZ];

  assert(cio);

  strcpy(fin, "unset");
  if (cio->f_header_read == colloid_io_read_header_ascii) strcpy(fin, "ascii");
  if (cio->f_header_read == colloid_io_read_header_binary) strcpy(fin, "binary");
  strcpy(fout, "unset");
  if (cio->f_header_write == colloid_io_write_header_ascii) strcpy(fout, "ascii");
  if (cio->f_header_write == colloid_io_write_header_binary) strcpy(fout, "binary");

  pe_info(cio->pe, "\n");
  pe_info(cio->pe, "Colloid I/O settings\n");
  pe_info(cio->pe, "--------------------\n");
  pe_info(cio->pe, "Decomposition:               %2d %2d %2d\n",
	  cio->nd[X], cio->nd[Y], cio->nd[Z]);
  pe_info(cio->pe, "Number of files:              %d\n", cio->n_io);
  pe_info(cio->pe, "Input format:                 %s\n", fin);
  pe_info(cio->pe, "Output format:                %s\n", fout);
  pe_info(cio->pe, "Single file read flag:        %d\n", cio->single_file_read);
  pe_info(cio->pe, "\n");

  return 0;
}

/*****************************************************************************
 *
 *  colloid_io_count_colloids
 *
 *  Count the local number of (physical, not halo) colloids and
 *  add up across the group. The root process of the group needs
 *  to know the total.
 *
 *****************************************************************************/

int colloid_io_count_colloids(colloid_io_t * cio, int * ngroup) {

  int nlocal;

  assert(cio);
  assert(ngroup);

  colloids_info_nlocal(cio->info, &nlocal);
  MPI_Reduce(&nlocal, ngroup, 1, MPI_INT, MPI_SUM, 0, cio->comm);

  return 0;
}

/*****************************************************************************
 *
 *  colloid_io_write
 *
 *  Write information on the colloids in the local domain proper (not
 *  including the halos) to the specified file.
 *
 *  In parallel, one file per io group is used. Processes within a
 *  group aggregate data to root, and root writes.
 *
 *****************************************************************************/

int colloid_io_write(colloid_io_t * cio, const char * filename) {

  int ntotal;
  int n, nc;
  int * nclist = NULL;
  int * displ = NULL;

  char filename_io[FILENAME_MAX];
  colloid_state_t * cbuf = NULL;
  colloid_state_t * rbuf = NULL;
  FILE * fp_state = NULL;

  assert(cio);

  colloids_info_ntotal(cio->info, &ntotal);
  if (ntotal == 0) return 0;

  assert(cio->f_header_write);
  assert(cio->f_buffer_write);

  /* Set the filename */

  colloid_io_filename(cio, filename_io, filename);

  pe_info(cio->pe, "\n");
  pe_info(cio->pe, "colloid_io_write:\n");
  pe_info(cio->pe, "writing colloid information to %s etc\n", filename_io);

  /* Gather a list of colloid numbers from each rank in the group */

  nclist = (int *) calloc(cio->size, sizeof(int));
  assert(nclist);
  if (nclist == NULL) pe_fatal(cio->pe, "malloc(nclist) failed\n");

  colloids_info_nlocal(cio->info, &nc);
  MPI_Gather(&nc, 1, MPI_INT, nclist, 1, MPI_INT, 0, cio->comm);

  /* Allocate local buffer, pack. */

  cbuf = (colloid_state_t *) malloc(nc*sizeof(colloid_state_t));
  assert(cbuf);
  if (cbuf == NULL) pe_fatal(cio->pe, "malloc(cbuf) failed\n");

  colloid_io_pack_buffer(cio, nc, cbuf);

  if (cio->rank == 0) {

    /* Work out displacements and the total */
    /* Allocate total recieve buffer */

    displ = (int *) malloc(cio->size*sizeof(int));
    assert(displ);
    if (displ == NULL) pe_fatal(cio->pe, "malloc(displ) failed\n");

    displ[0] = 0;
    for (n = 1; n < cio->size; n++) {
      displ[n] = displ[n-1] + nclist[n-1];
    }

    ntotal = 0;
    for (n = 0; n < cio->size; n++) {
      ntotal += nclist[n];
      displ[n] *= sizeof(colloid_state_t);   /* to bytes */
      nclist[n] *= sizeof(colloid_state_t);  /* ditto */
    }

    assert(ntotal > 0); /* STATIC ANALYSIS */
    rbuf = (colloid_state_t *) malloc(ntotal*sizeof(colloid_state_t));
    assert(rbuf);
    if (rbuf == NULL) pe_fatal(cio->pe, "malloc(rbuf) failed\n");
  }

  MPI_Gatherv(cbuf, nc*sizeof(colloid_state_t), MPI_BYTE, rbuf, nclist,
	      displ, MPI_BYTE, 0, cio->comm);

  if (cio->rank == 0) {

    fp_state = fopen(filename_io, "w");
    if (fp_state == NULL) {
      pe_fatal(cio->pe, "Failed to open %s\n", filename_io);
    }
    cio->f_header_write(fp_state, ntotal);
    cio->f_buffer_write(fp_state, ntotal, rbuf);

    if (ferror(fp_state)) {
      perror("perror: ");
      pe_fatal(cio->pe, "Error on writing file %s\n", filename_io);
    }

    fclose(fp_state);
    free(rbuf);
    free(displ);
  }

  free(cbuf);
  free(nclist);

  return 0;
}

/*****************************************************************************
 *
 *  colloid_io_pack_buffer
 *
 *  Transfer list content to contiguous buffer (nc colloids).
 *
 *****************************************************************************/

int colloid_io_pack_buffer(colloid_io_t * cio, int nc, colloid_state_t * buf) {

  int n = 0;
  colloid_t * pc = NULL;

  assert(cio);
  assert(buf);

  colloids_info_local_head(cio->info, &pc);

  while (pc) {
    assert(n < nc);
    memcpy(buf + n, &pc->s, sizeof(colloid_state_t));
    n += 1;
    pc = pc->nextlocal;
  }

  /* Check the size did match exactly. */
  if (n != nc) pe_fatal(cio->pe, "Internal error in cio_pack_buffer\n");

  return 0;
}

/*****************************************************************************
 *
 *  colloid_io_write_buffer_ascii
 *
 *  Write nc states to file. Return number of bytes written.
 *
 *****************************************************************************/

int colloid_io_write_buffer_ascii(FILE * fp, int nc, colloid_state_t * buf) {

  int n;
  int ifail = 0;

  assert(fp);
  assert(buf);

  for (n = 0; n < nc; n++) {
    ifail += colloid_state_write_ascii(buf + n, fp);
  }

  return ifail;
}

/*****************************************************************************
 *
 *  colloid_io_write_buffer_binary
 *
 *  Write nc states to file.
 *
 *****************************************************************************/

int colloid_io_write_buffer_binary(FILE * fp, int nc, colloid_state_t * buf) {

  assert(fp);
  assert(buf);

  fwrite(buf, sizeof(colloid_state_t), nc, fp);

  return 0;
}

/*****************************************************************************
 *
 *  colloid_io_read
 *
 *  This is the driver routine to read colloid information from file.
 *
 *  The read is a free-for-all in which all processes in the group
 *  read the entire file.
 *
 *****************************************************************************/

int colloid_io_read(colloid_io_t * cio, const char * filename) {

  int    ngroup;
  char   filename_io[FILENAME_MAX];
  FILE * fp_state;

  assert(cio->f_header_read);
  assert(cio->f_list_read);

  /* Set the filename from the stub and the extension */

  colloid_io_filename(cio, filename_io, filename);

  if (cio->single_file_read) {
    /* All groups read for single 'serial' file */
    sprintf(filename_io, "%s.%3.3d-%3.3d", filename, 1, 1);
    pe_info(cio->pe, "colloid_io_read: reading from single file %s\n", filename_io);
  }
  else {
    pe_info(cio->pe, "colloid_io_read: reading from %s etc\n", filename_io);
  }

  /* Open the file and read the information */

  fp_state = fopen(filename_io, "r");
  if (fp_state == NULL) pe_fatal(cio->pe, "Failed to open %s\n", filename_io);

  cio->f_header_read(fp_state, &ngroup);
  cio->f_list_read(cio, ngroup, fp_state);

  if (ferror(fp_state)) {
    perror("perror: ");
    pe_fatal(cio->pe, "Error on reading %s\n", filename_io);
  }

  fclose(fp_state);

  colloid_io_check_read(cio, ngroup);

  return 0;
}

/*****************************************************************************
 *
 *  colloid_io_write_header_ascii
 *
 *  Write colloid information to file in human-readable form.
 *
 *****************************************************************************/

static int colloid_io_write_header_ascii(FILE * fp, int ngroup) {

  fprintf(fp, "%22d\n", ngroup);

  return 0;
}

/*****************************************************************************
 *
 *  colloid_io_read_header_ascii
 *
 *  Reads the number of particles in the file.
 *
 *****************************************************************************/

static int colloid_io_read_header_ascii(FILE * fp, int * nfile) {

  int nr;
  
  assert(nfile);

  nr = fscanf(fp, "%22d\n",  nfile);
  if (nr != 1) return -1; 
  
  return 0;
}

/*****************************************************************************
 *
 *  colloid_io_write_header_binary
 *
 *  Write the colloid header information to file.
 *
 *****************************************************************************/

static int colloid_io_write_header_binary(FILE * fp, int ngroup) {

  fwrite(&ngroup, sizeof(int), 1, fp);

  return 0;
}

/*****************************************************************************
 *
 *  colloid_io_read_header_binary
 *
 *  The read equivalent of the above.
 *
 *****************************************************************************/

static int colloid_io_read_header_binary(FILE * fp, int * nfile) {

  size_t nr;
  
  assert(fp);
  assert(nfile);

  nr = fread(nfile, sizeof(int), 1, fp);
  if (nr != 1) return -1;
  
  return 0;
}

/*****************************************************************************
 *
 *  colloid_io_write_list_ascii
 *
 *****************************************************************************/

static int colloid_io_write_list_ascii(colloid_io_t * cio,
				       int ic, int jc, int kc, FILE * fp) {
  int ifail = 0;
  colloid_t * pc = NULL;

  assert(cio);
  assert(fp);

  colloids_info_cell_list_head(cio->info, ic, jc, kc, &pc);

  while (pc) {
    ifail += colloid_state_write_ascii(&pc->s, fp);
    pc = pc->next;
  }

  return ifail;
}

/*****************************************************************************
 *
 *  colloid_io_read_list_ascii
 *
 *****************************************************************************/

static int colloid_io_read_list_ascii(colloid_io_t * cio, int ndata,
				      FILE * fp) {
  int ifail = 0;
  int nread;
  int nlocal = 0;

  colloid_state_t s;
  colloid_t * p_colloid;

  assert(cio);
  assert(fp);

  for (nread = 0; nread < ndata; nread++) {
    ifail += colloid_state_read_ascii(&s, fp);
    colloids_info_add_local(cio->info, s.index, s.r, &p_colloid);
    if (p_colloid) {
      p_colloid->s = s;
      nlocal++;
    }
  }

  /* Deliver a warning, but continue */
  if (ifail) {
    pe_verbose(cio->pe, "Possible error in colloid_io_read_list_ascii\n");
  }

  return ifail;
}

/*****************************************************************************
 *
 *  colloid_io_write_list_binary
 *
 *****************************************************************************/

static int colloid_io_write_list_binary(colloid_io_t * cio,
					int ic, int jc, int kc, FILE * fp) {
  colloid_t * pc = NULL;

  assert(cio);
  assert(fp);

  colloids_info_cell_list_head(cio->info, ic, jc, kc, &pc);

  while (pc) {
    colloid_state_write_binary(&pc->s, fp);
    pc = pc->next;
  }

  return 0;
}

/*****************************************************************************
 *
 *  colloid_io_read_list_binary
 *
 *****************************************************************************/

static int colloid_io_read_list_binary(colloid_io_t * cio, int ndata,
				       FILE * fp) {

  int nread;
  colloid_state_t s;
  colloid_t * pc = NULL;

  assert(cio);
  assert(fp);

  for (nread = 0; nread < ndata; nread++) {
    colloid_state_read_binary(&s, fp);
    colloids_info_add_local(cio->info, s.index, s.r, &pc);
    if (pc)  pc->s = s;
  }

  return 0;
}

/*****************************************************************************
 *
 *  colloid_io_format_input_ascii_set
 *
 *****************************************************************************/

int colloid_io_format_input_ascii_set(colloid_io_t * cio) {

  assert(cio);

  cio->f_header_read = colloid_io_read_header_ascii;
  cio->f_list_read   = colloid_io_read_list_ascii;

  return 0;
}

/*****************************************************************************
 *
 *  colloid_io_format_input_binary_set
 *
 *****************************************************************************/

int colloid_io_format_input_binary_set(colloid_io_t * cio) {

  assert(cio);

  cio->f_header_read = colloid_io_read_header_binary;
  cio->f_list_read   = colloid_io_read_list_binary;

  return 0;
}

/*****************************************************************************
 *
 *  colloid_io_format_input_serial_set
 *
 *****************************************************************************/

int colloid_io_format_input_serial_set(colloid_io_t * cio) {

  assert(cio);

  cio->single_file_read = 1;

  return 0;
}

/*****************************************************************************
 *
 *  colloid_io_format_output_ascii_set
 *
 *****************************************************************************/

int colloid_io_format_output_ascii_set(colloid_io_t * cio) {

  assert(cio);

  cio->f_buffer_write = colloid_io_write_buffer_ascii;
  cio->f_header_write = colloid_io_write_header_ascii;
  cio->f_list_write   = colloid_io_write_list_ascii;

  return 0;
}

/*****************************************************************************
 *
 *  colloid_io_format_output_binary_set
 *
 *****************************************************************************/

int colloid_io_format_output_binary_set(colloid_io_t * cio) {

  cio->f_buffer_write = colloid_io_write_buffer_binary;
  cio->f_header_write = colloid_io_write_header_binary;
  cio->f_list_write   = colloid_io_write_list_binary;

  return 0;
}

/*****************************************************************************
 *
 *  colloid_io_filename
 *
 *  Add the extension to the supplied stub.
 *
 *****************************************************************************/

static int colloid_io_filename(colloid_io_t * cio, char * filename,
			       const char * stub) {
  assert(cio);
  assert(stub);
  assert(strlen(stub) < FILENAME_MAX/2);  /* Check stub not too long */

  if (cio->index >= 1000) {
    pe_fatal(cio->pe, "Format botch for cio stub %s\n", stub);
  }

  sprintf(filename, "%s.%3.3d-%3.3d", stub, cio->n_io, cio->index + 1);

  return 0;
}

/*****************************************************************************
 *
 *  colloid_io_check_read
 *
 *  Check the number of colloids in the list is consistent with that
 *  in the file (ngroup), and set the total.
 *
 *  If we haven't lost any particles, we can set the global total and
 *  proceed.
 *
 *****************************************************************************/

static int colloid_io_check_read(colloid_io_t * cio, int ngroup) {

  int nlocal;
  int ntotal;

  assert(cio);

  colloids_info_nlocal(cio->info, &nlocal);

  MPI_Reduce(&nlocal, &ntotal, 1, MPI_INT, MPI_SUM, 0, cio->comm);

  if (cio->single_file_read) {
    /* Only the global total can be compared (ngroup is ntotal). */
    nlocal = ntotal;
    MPI_Allreduce(&nlocal, &ntotal, 1, MPI_INT, MPI_SUM, cio->xcomm);
  }

  if (cio->rank == 0) {
    if (ntotal != ngroup) {
      pe_verbose(cio->pe, "Colloid I/O group %d\n", cio->index);
      pe_verbose(cio->pe, "Colloids in file: %d Got %d\n", ngroup, ntotal);
      pe_fatal(cio->pe, "Total number of colloids not consistent with file\n");
    }
  }

  colloids_info_ntotal_set(cio->info);
  colloids_info_ntotal(cio->info, &ntotal);
  pe_info(cio->pe, "Read a total of %d colloids from file\n", ntotal);

  return 0;
}
