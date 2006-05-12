/*****************************************************************************
 *
 *  cio.c
 *
 *  Colloid I/O, serial and parallel.
 *
 *  Kevin Stratford (kevin@epcc.ed.ac.uk)
 *
 *****************************************************************************/

#include "utilities.h"
#include "colloids.h"
#include "cells.h"
#include "cio.h"

#include "pe.h"

#ifdef _MPI_
#define         TAG_IO 2000
extern MPI_Comm IO_Comm;
#endif

void CIO_read_list_ascii(FILE *);
void CIO_read_list_binary(FILE *);
void CIO_read_header_ascii(FILE *);
void CIO_read_header_binary(FILE *);
void CIO_write_header_ascii(FILE *);
void CIO_write_header_binary(FILE *);
void CIO_write_header_null(FILE *);
int  CIO_write_list_ascii(FILE *, int, int, int);
int  CIO_write_list_binary(FILE *, int, int, int);
int  CIO_write_xu_binary(FILE *, int, int, int);

extern IO_Param io_grp;                  /* From communicate.c */

static void (* CIO_write_header)(FILE *);
static int  (* CIO_write_list)(FILE *, int, int, int);
static void (* CIO_read_header)(FILE *);
static void (* CIO_read_list)(FILE *);

/*****************************************************************************
 *
 *  CIO_write_state
 *
 *  Write information on the colloids in the local domain proper (not
 *  including the halos) to the specified file.
 *
 *  In parallel, one file per io group is used. Processes within a
 *  group write to the same file on reciept of the token.
 *
 *****************************************************************************/

void CIO_write_state(const char * filename) {

  FILE *    fp_state;
  char      filename_io[FILENAME_MAX];
  int       token = 0;

  int       ic, jc, kc;
  IVector   ncell;
  void      CMPI_count_colloids(void);

#ifdef _MPI_
  MPI_Status status;
#endif

#ifdef _COLLOIDS_

  /* Set the filename */

  info("CIO_write_state\n");
  sprintf(filename_io, "%s%s", filename, io_grp.file_ext);

  /* Make sure everyone has their current number of particles
   * up-to-date */

  CMPI_count_colloids();

  if (io_grp.root) {
    /* Open the file, and write the header, followed by own colloids.
     * When this is done, pass the token to the next processs in the
     * group. */

    fp_state = fopen(filename_io, "w+b");
    if (fp_state == NULL) fatal("Failed to open %s\n", filename_io);
  }
  else {
    /* Non-io-root process. Block until we receive the token from the
     * previous process, after which we can go ahead and append our own
     * colloids to the file. */

#ifdef _MPI_
    MPI_Recv(&token, 1, MPI_INT, io_grp.rank - 1, TAG_IO, IO_Comm, &status);
#endif
    fp_state = fopen(filename_io, "a+b");
    if (fp_state == NULL) fatal("Failed to open %s\n", filename_io);
  }

  /* Write the local colloid state, consisting of header plus data. */

  CIO_write_header(fp_state);

  ncell = Global_Colloid.Ncell;

  for (ic = 1; ic <= ncell.x; ic++) {
    for (jc = 1; jc <= ncell.y; jc++) {
      for (kc = 1; kc <= ncell.z; kc++) {
	token += CIO_write_list(fp_state, ic, jc, kc);
      }
    }
  }

  fclose(fp_state);

  /* This process has finished, so we can pass the token to the
   * next process in the group. If this is the last process in the
   * group, we've completed the file. */

  if (io_grp.rank < io_grp.size - 1) {
    /* Send the token, which happens to be the accumulated number of
     * particles. */
#ifdef _MPI_
    MPI_Ssend(&token, 1, MPI_INT, io_grp.rank + 1, TAG_IO, IO_Comm);
#endif
  }
  else {
    /* Last  process ... */
    VERBOSE(("IO group %d wrote %d colloids to %s\n", io_grp.index,
	     token, filename_io));
  }

#endif

  return;
}


/*****************************************************************************
 *
 *  CIO_read_state
 *
 *  This is the driver routine to read colloid information from file.
 *
 *****************************************************************************/

void CIO_read_state(const char * filename) {

  FILE *    fp_state;
  char      filename_io[FILENAME_MAX];
  long int  token = 0;

#ifdef _MPI_
  MPI_Status status;
#endif

  /* Set the filename from the stub and the extension */

  info("CIO_read_state\n");
  sprintf(filename_io, "%s%s", filename, io_grp.file_ext);

  if (io_grp.root) {
    /* Open the file and read the header information */
    fp_state = fopen(filename_io, "r");
    if (fp_state == NULL) fatal("Failed to open %s\n", filename_io);
  }
  else {
    /* Block until we receive the token which allows us to proceed.
     * Then, open the file and move to the appropriate position before
     * starting to read. */
#ifdef _MPI_
    MPI_Recv(&token, 1, MPI_LONG, io_grp.rank - 1, TAG_IO, IO_Comm, &status);
#endif
    fp_state = fopen(filename_io, "r");
    if (fp_state == NULL) fatal("Failed to open %s\n", filename_io);
    rewind(fp_state);
    fseek(fp_state, token, SEEK_SET);
  }

  /* Read the data */

  CIO_read_header(fp_state);
  CIO_read_list(fp_state);

  token = ftell(fp_state);
  fclose(fp_state);

  /* Pass on the token, which is the current offset in the file */

  if (io_grp.rank < io_grp.size - 1) {
#ifdef _MPI_
    MPI_Ssend(&token, 1, MPI_LONG, io_grp.rank + 1, TAG_IO, IO_Comm);
#endif
  }
  else {
    VERBOSE(("IO group %d read %d colloids from %s\n", io_grp.index,
	     1, filename_io));
  }

  return;
}


/*****************************************************************************
 *
 *  CIO_write_header_ascii
 *
 *  Write colloid information to file in human-readable form.
 *
 *****************************************************************************/

void CIO_write_header_ascii(FILE * fp) {

  fprintf(fp, "I/O n_io:  %22d\n",    io_grp.n_io);
  fprintf(fp, "I/O index: %22d\n",    io_grp.index);
  fprintf(fp, "N_colloid: %22d\n",    Global_Colloid.N_colloid);
  fprintf(fp, "nlocal:    %22d\n",    Global_Colloid.nlocal);
  fprintf(fp, "rebuild:   %22d\n",    Global_Colloid.fr);
  fprintf(fp, "a0:        %22.15e\n", Global_Colloid.a0);
  fprintf(fp, "ah:        %22.15e\n", Global_Colloid.ah);
  fprintf(fp, "vf:        %22.15e\n", Global_Colloid.vf);
  fprintf(fp, "rho:       %22.15e\n", Global_Colloid.rho);
  fprintf(fp, "deltaf:    %22.15e\n", Global_Colloid.deltaf);
  fprintf(fp, "deltag:    %22.15e\n", Global_Colloid.deltag);
  fprintf(fp, "r_lu_n:    %22.15e\n", Global_Colloid.r_lu_n);
  fprintf(fp, "r_lu_t:    %22.15e\n", Global_Colloid.r_lu_t);
  fprintf(fp, "r_lu_r:    %22.15e\n", Global_Colloid.r_lu_r);
  fprintf(fp, "r_ssph:    %22.15e\n", Global_Colloid.r_ssph);
  fprintf(fp, "r_clus:    %22.15e\n", Global_Colloid.r_clus);
  fprintf(fp, "Ncell.x:   %22d\n",    Global_Colloid.Ncell.x);
  fprintf(fp, "Ncell.y:   %22d\n",    Global_Colloid.Ncell.y);
  fprintf(fp, "Ncell.z:   %22d\n",    Global_Colloid.Ncell.z);
  fprintf(fp, "Lcell.x:   %22.15e\n", Global_Colloid.Lcell.x);
  fprintf(fp, "Lcell.y:   %22.15e\n", Global_Colloid.Lcell.y);
  fprintf(fp, "Lcell.z:   %22.15e\n", Global_Colloid.Lcell.z);
  fprintf(fp, "F.x:       %22.15e\n", Global_Colloid.F.x);
  fprintf(fp, "F.y:       %22.15e\n", Global_Colloid.F.y);
  fprintf(fp, "F.z:       %22.15e\n", Global_Colloid.F.z);
  fprintf(fp, "pid:       %22d\n",    Global_Colloid.pid);
  fprintf(fp, "drop_in_p1:%22.15e\n", Global_Colloid.drop_in_p1);
  fprintf(fp, "drop_in_p2:%22.15e\n", Global_Colloid.drop_in_p2);
  fprintf(fp, "drop_in_p3:%22.15e\n", Global_Colloid.drop_in_p3);

  return;
}


/*****************************************************************************
 *
 *  CIO_read_header_ascii
 *
 *  Everybody reads their own header, which should include the
 *  local number of particles.
 *
 *  Issues
 *    A large nuumber of these values are ignored at the moment,
 *    meaning that run time input must agree. There is an outstanding
 *    issue as to what to do if one wants to override run time input.
 *
 *****************************************************************************/

void CIO_read_header_ascii(FILE * fp) {

  int    read_int;
  Float  read_float;

  info("Colloid file header information has been filtered\n");

  fscanf(fp, "I/O n_io:  %22d\n",  &read_int);
  fscanf(fp, "I/O index: %22d\n",  &read_int);
  fscanf(fp, "N_colloid: %22d\n",  &(Global_Colloid.N_colloid));
  fscanf(fp, "nlocal:    %22d\n",  &(Global_Colloid.nlocal));
  fscanf(fp, "rebuild:   %22d\n",  &read_int);
  fscanf(fp, "a0:        %22lg\n", &read_float);
  fscanf(fp, "ah:        %22lg\n", &read_float);

  fscanf(fp, "vf:        %22lg\n", &read_float);
  fscanf(fp, "rho:       %22lg\n", &read_float);
  fscanf(fp, "deltaf:    %22lg\n", &read_float);
  fscanf(fp, "deltag:    %22lg\n", &read_float);
  fscanf(fp, "r_lu_n:    %22lg\n", &read_float);
  fscanf(fp, "r_lu_t:    %22lg\n", &read_float);
  fscanf(fp, "r_lu_r:    %22lg\n", &read_float);
  fscanf(fp, "r_ssph:    %22lg\n", &read_float);
  fscanf(fp, "r_clus:    %22lg\n", &read_float);
  fscanf(fp, "Ncell.x:   %22d\n",  &read_int);
  fscanf(fp, "Ncell.y:   %22d\n",  &read_int);
  fscanf(fp, "Ncell.z:   %22d\n",  &read_int);

  fscanf(fp, "Lcell.x:   %22lg\n", &read_float);
  fscanf(fp, "Lcell.y:   %22lg\n", &read_float);
  fscanf(fp, "Lcell.z:   %22lg\n", &read_float);
  fscanf(fp, "F.x:       %22lg\n", &read_float);
  fscanf(fp, "F.y:       %22lg\n", &read_float);
  fscanf(fp, "F.z:       %22lg\n", &read_float);
  fscanf(fp, "pid:       %22d\n",  &read_int);
  fscanf(fp, "drop_in_p1:%22lg\n", &read_float);
  fscanf(fp, "drop_in_p2:%22lg\n", &read_float);
  fscanf(fp, "drop_in_p3:%22lg\n", &read_float);

  return;
}


/*****************************************************************************
 *
 *  CIO_write_header_binary
 *
 *  Write the colloid header information to file.
 *
 *****************************************************************************/

void CIO_write_header_binary(FILE * fp) {

  fwrite(&(io_grp.n_io),               sizeof(int),     1, fp);
  fwrite(&(io_grp.index),              sizeof(int),     1, fp);
  fwrite(&(Global_Colloid.N_colloid),  sizeof(int),     1, fp);
  fwrite(&(Global_Colloid.nlocal),     sizeof(int),     1, fp);
  fwrite(&(Global_Colloid.fr),         sizeof(int),     1, fp);

  fwrite(&(Global_Colloid.a0),         sizeof(Float),   1, fp);
  fwrite(&(Global_Colloid.ah),         sizeof(Float),   1, fp);
  fwrite(&(Global_Colloid.vf),         sizeof(Float),   1, fp);
  fwrite(&(Global_Colloid.rho),        sizeof(Float),   1, fp);
  fwrite(&(Global_Colloid.deltaf),     sizeof(Float),   1, fp);
  fwrite(&(Global_Colloid.deltag),     sizeof(Float),   1, fp);
  fwrite(&(Global_Colloid.r_lu_n),     sizeof(Float),   1, fp);
  fwrite(&(Global_Colloid.r_lu_t),     sizeof(Float),   1, fp);
  fwrite(&(Global_Colloid.r_lu_r),     sizeof(Float),   1, fp);
  fwrite(&(Global_Colloid.r_ssph),     sizeof(Float),   1, fp);
  fwrite(&(Global_Colloid.r_clus),     sizeof(Float),   1, fp);

  fwrite(&(Global_Colloid.Ncell),      sizeof(IVector), 1, fp);
  fwrite(&(Global_Colloid.Lcell),      sizeof(FVector), 1, fp);
  fwrite(&(Global_Colloid.F),          sizeof(FVector), 1, fp);
  fwrite(&(Global_Colloid.pid),        sizeof(int),     1, fp);
  fwrite(&(Global_Colloid.drop_in_p1), sizeof(Float),   1, fp);
  fwrite(&(Global_Colloid.drop_in_p2), sizeof(Float),   1, fp);
  fwrite(&(Global_Colloid.drop_in_p3), sizeof(Float),   1, fp);

  return;
}

void CIO_write_header_null(FILE * fp) {

  fwrite(&(Global_Colloid.nlocal), sizeof(int), 1, fp);

  return;
}

/*****************************************************************************
 *
 *  CIO_read_header_binary
 *
 *  The read equivalent of the above.
 *
 *****************************************************************************/

void CIO_read_header_binary(FILE * fp) {

  int     read_int;
  IVector read_ivector;
  Float   read_float;
  FVector read_fvector;

  info("Colloid file header information has been filtered\n");

  fread(&(read_int),                  sizeof(int),     1, fp); /* n_io */
  fread(&(read_int),                  sizeof(int),     1, fp);
  fread(&(Global_Colloid.N_colloid),  sizeof(int),     1, fp);
  fread(&(Global_Colloid.nlocal),     sizeof(int),     1, fp);
  fread(&(read_int),                  sizeof(int),     1, fp);

  fread(&(read_float),                sizeof(Float),   1, fp); /* a0 */
  fread(&(read_float),                sizeof(Float),   1, fp);
  fread(&(read_float),                sizeof(Float),   1, fp);
  fread(&(read_float),                sizeof(Float),   1, fp);
  fread(&(read_float),                sizeof(Float),   1, fp);
  fread(&(read_float),                sizeof(Float),   1, fp);
  fread(&(read_float),                sizeof(Float),   1, fp);
  fread(&(read_float),                sizeof(Float),   1, fp);
  fread(&(read_float),                sizeof(Float),   1, fp);
  fread(&(read_float),                sizeof(Float),   1, fp);
  fread(&(read_float),                sizeof(Float),   1, fp);

  fread(&(read_ivector),              sizeof(IVector), 1, fp);
  fread(&(read_fvector),              sizeof(FVector), 1, fp);
  fread(&(read_fvector),              sizeof(FVector), 1, fp);
  fread(&(read_int),                  sizeof(int),     1, fp);
  fread(&(read_float),                sizeof(Float),   1, fp);
  fread(&(read_float),                sizeof(Float),   1, fp);
  fread(&(read_float),                sizeof(Float),   1, fp);

  return;
}

/*****************************************************************************
 *
 *  CIO_write_list_ascii
 *
 *****************************************************************************/

int CIO_write_list_ascii(FILE * fp, int ic, int jc, int kc) {

  Colloid * p_colloid;
  int       nwrite = 0;

  p_colloid = CELL_get_head_of_list(ic, jc, kc);

  while (p_colloid) {
    nwrite++;

    fprintf(fp, "%22.15e %22.15e %d\n", p_colloid->a0, p_colloid->ah,
	    p_colloid->index);
    fprintf(fp, "%22.15e %22.15e %22.15e\n", p_colloid->r.x, p_colloid->r.y,
	    p_colloid->r.z);
    fprintf(fp, "%22.15e %22.15e %22.15e\n", p_colloid->v.x, p_colloid->v.y,
	    p_colloid->v.z);
    fprintf(fp, "%22.15e %22.15e %22.15e\n", p_colloid->omega.x,
	    p_colloid->omega.y, p_colloid->omega.z);
    fprintf(fp, "%22.15e\n", p_colloid->deltaphi);
 
    /* Next colloid */
    p_colloid = p_colloid->next;
  }

  return nwrite;
}


/*****************************************************************************
 *
 *  CIO_read_ascii
 *
 *  The colloid data from file an add a new particle to the local
 *  list.
 *
 *****************************************************************************/

void CIO_read_list_ascii(FILE * fp) {

  int       nread;
  int       read_index;
  Float     read_a0;
  Float     read_ah;
  FVector   read_r, read_v, read_o;
  Float     read_deltaphi;
  Colloid * p_colloid;

  for (nread = 0; nread < Global_Colloid.nlocal; nread++) {

    fscanf(fp, "%22le %22le %22d\n",  &(read_a0), &(read_ah), &(read_index));
    fscanf(fp, "%22le %22le %22le\n", &(read_r.x), &(read_r.y), &(read_r.z));
    fscanf(fp, "%22le %22le %22le\n", &(read_v.x), &(read_v.y), &(read_v.z));
    fscanf(fp, "%22le %22le %22le\n", &(read_o.x), &(read_o.y), &(read_o.z));
    fscanf(fp, "%22le\n",             &(read_deltaphi));

    p_colloid = COLL_add_colloid(read_index, read_a0, read_ah, read_r,
				 read_v, read_o);

    if (p_colloid) {
      p_colloid->deltaphi = read_deltaphi;
    }
    else {
      /* This didn't go into the cell list */
      fatal("Colloid information doesn't tally with read position\n");
    }
  }

  return;
}

/*****************************************************************************
 *
 *  CIO_write_list_binary
 *
 *  Write out the colliod information (if any) for the specified list.
 *
 *****************************************************************************/

int CIO_write_list_binary(FILE * fp, int ic, int jc, int kc) {

  Colloid * p_colloid;
  int       nwrite = 0;

  p_colloid = CELL_get_head_of_list(ic, jc, kc);

  while (p_colloid) {
    nwrite++;
    fwrite(&(p_colloid->a0),       sizeof(Float),   1, fp);
    fwrite(&(p_colloid->ah),       sizeof(Float),   1, fp);
    fwrite(&(p_colloid->index),    sizeof(int),     1, fp);
    fwrite(&(p_colloid->r),        sizeof(FVector), 1, fp);
    fwrite(&(p_colloid->v),        sizeof(FVector), 1, fp);
    fwrite(&(p_colloid->omega),    sizeof(FVector), 1, fp);
    fwrite(&(p_colloid->deltaphi), sizeof(Float),   1, fp);

    /* Next colloid */
    p_colloid = p_colloid->next;
  }

  return nwrite;
}


/*****************************************************************************
 *
 *  CIO_read_binary
 *
 *  Write me.
 *
 *****************************************************************************/

void CIO_read_list_binary(FILE * fp) {

  int       nread;
  int       read_index;
  Float     read_a0;
  Float     read_ah;
  FVector   read_r, read_v, read_o;
  Float     read_deltaphi;
  Colloid * p_colloid;

  for (nread = 0; nread < Global_Colloid.nlocal; nread++) {

    fread(&read_a0,       sizeof(Float),   1, fp);
    fread(&read_ah,       sizeof(Float),   1, fp);
    fread(&read_index,    sizeof(int),     1, fp);
    fread(&read_r,        sizeof(FVector), 1, fp);
    fread(&read_v,        sizeof(FVector), 1, fp);
    fread(&read_o,        sizeof(FVector), 1, fp);
    fread(&read_deltaphi, sizeof(Float),   1, fp);

    p_colloid = COLL_add_colloid(read_index, read_a0, read_ah, read_r,
				 read_v, read_o);

    if (p_colloid) {
      p_colloid->deltaphi = read_deltaphi;
    }
    else {
      /* This didn't go into the cell list */
      fatal("Colloid information doesn't tally with read position\n");
    }
  }

  return;
}

/*****************************************************************************
 *
 *  CIO_write_xu_binary
 *
 *  Write the position and velocity alone.
 *
 *****************************************************************************/

int CIO_write_xu_binary(FILE * fp, int ic, int jc, int kc) {

  Colloid * p_colloid;
  int       nwrite = 0;

  p_colloid = CELL_get_head_of_list(ic, jc, kc);

  while (p_colloid) {
    nwrite++;
    fwrite(&(p_colloid->index),    sizeof(int),     1, fp);
    fwrite(&(p_colloid->r),        sizeof(FVector), 1, fp);
    fwrite(&(p_colloid->v),        sizeof(FVector), 1, fp);

    /* Next colloid */
    p_colloid = p_colloid->next;
  }

  return nwrite;
}


/*****************************************************************************
 *
 *  CIO_set_cio_format
 *
 *  Set the format for IO {BINARY|ASCII}.
 *
 *****************************************************************************/

void CIO_set_cio_format(int io_intype, int io_outtype) {

  switch (io_intype) {
  case BINARY:
    CIO_read_header  = CIO_read_header_binary;
    CIO_read_list    = CIO_read_list_binary;
    break;
  case ASCII:
    CIO_read_header  = CIO_read_header_ascii;
    CIO_read_list    = CIO_read_list_ascii;
    break;
  default:
    fatal("Invalid colloid input format (value %d)\n", io_intype);
  }

  switch (io_outtype) {
  case BINARY:
    CIO_write_header = CIO_write_header_binary;
    CIO_write_list   = CIO_write_list_binary;
    break;
  case ASCII:
    CIO_write_header = CIO_write_header_ascii;
    CIO_write_list   = CIO_write_list_ascii;
    break;
  default:
    fatal("Invalid colloid output format (value %d)\n", io_outtype);
  }

#ifdef _CONSORTIUM_
  CIO_write_header = CIO_write_header_null;
  CIO_write_list   = CIO_write_xu_binary;
#endif

  return;
}
