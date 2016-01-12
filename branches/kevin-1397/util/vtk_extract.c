/****************************************************************************
 *
 *  vtk_extract.c
 *
 *  This program deals with the consolidation of parallel I/O
 *  (and in serial with Lees-Edwards planes). This program is
 *  serial.
 *
 *  To process/recombined parallel I/O into a single file with the
 *  correct order (ie., z running fastest, then y, then x) run
 *
 *  ./vtk_extract meta-file data-file
 *
 *  where 'meta-file' is the first meta information file in the series,
 *  and 'data-file' is the first data file for a given time step. All
 *  values are expected to be of data-type double - this will be
 *  preserved on output.
 *
 *  Clearly, the two input files must match, or results will be
 *  unpredictable.
 *  
 *  Flags need to be set for post-processing and output of the
 *  scalar and biaxial order parameter and director field.
 *
 *  Compile with $(CC) vtk_extract.c -lm
 *
 *  $Id$
 *
 *  Edinburgh Soft Matter and Statistical Physics Group and
 *  Edinburgh Parallel Computing Centre
 *
 *  Oliver Henrich (ohenrich@epcc.ed.ac.uk)
 *  Kevin Stratford (kevin@epcc.ed.ac.uk)
 *  (c) 2014 The University of Edinburgh
 *
 ****************************************************************************/



#include <assert.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <float.h>

int nlocal[3];
int rank, cx, cy, cz, nlx, nly, nlz, nox, noy, noz;

int ntotal[3];
int ntargets[3]; /* Output target */
int pe_[3];
int nplanes_ = 0;
int nio_;
int nrec_ = 1;
int npp_ = 0;
int input_isbigendian_ = -1;   /* May need to deal with endianness */
int reverse_byte_order_ = 0;   /* Switch for bigendian input */
int input_binary_  = 0;        /* Switch for format of input */
int output_binary_ = 0;        /* Switch for format of final output */

int output_lc_op_   = 1;       /* Switch for LC order parameter output */
int output_lc_dir_  = 1;       /* Switch for LC director output */
int output_lc_biop_ = 1;       /* Switch for LC biaxial order parameter output */

int output_ek_psi_ = 1;        /* Switch for EK potenital output */
int output_ek_elc_ = 1;        /* Switch for EK electric charge density */
int output_ek_elf_ = 1;        /* Switch for EK electric field */
int output_ek_rho_ = 1;        /* Switch for EK number densities output */

int is_velocity_ = 0;          /* Switch to identify velocity field */

int output_index_ = 0;         /* For ASCII output, include (i,j,k) indices */
int vtk_header = 1;            /* For visualisation with Paraview */
int output_cmf_ = 1;           /* Flag for output in column-major format */

double e0[3] = {0.0, 0.0, 0.0};  /* External electric field for potential jump */

int le_t0_ = 0;                /* LE offset start time (time steps) */ 

double le_speed_ = 0.0;
double le_displace_ = 0.0;
double * le_displacements_;
double * le_duy_;

char stub_[FILENAME_MAX];

void read_meta_data_file(const char *);
int  read_data_file_name(const char *);

int site_index(int, int, int, const int *);
void read_data(FILE *, int *, double *);
void write_data(FILE *, int *, double *);
void write_data_cmf(FILE *, int *, double *);
void write_data_q(FILE *, FILE *, FILE *, int *, double *);
void write_data_q_cmf(FILE *, FILE *, FILE *, int *, double *);
void write_data_psi(FILE *, FILE *, FILE *, FILE *, FILE *, int *, double *, double *);
void write_data_psi_cmf(FILE *, FILE *, FILE *, FILE *, FILE *, int *, double *, double *);
int copy_data(double *, double *);
int le_displacement(int, int);
void le_set_displacements(void);
void le_unroll(double *);
double reverse_byte_order_double(char *);

void electric_field(double *, double *);

void order_parameters(double *);
void calculate_scalar_biaxial_order_parameter_director(double q[5], double qs[5]);
int  util_jacobi_sort(double a[3][3], double vals[3], double vecs[3][3]);
int  util_jacobi(double a[3][3], double vals[3], double vecs[3][3]);
void util_swap(int ia, int ib, double a[3], double b[3][3]);

int is_bigendian(void);

int main(int argc, char ** argv) {

  int ntime;
  int i, n, p, pe_per_io;
  int ifail;

  double * datalocal;
  double * datasection;
  double * ppsection;
  char io_metadata[FILENAME_MAX];
  char io_data[FILENAME_MAX];
  char line[FILENAME_MAX];

  char opfilename[60];
  char dirfilename[60];
  char biopfilename[60];

  char psi_filename[60];
  char rho0_filename[60];
  char rho1_filename[60];
  char elc_filename[60];
  char elf_filename[60];

  char genfilename[60];

  FILE * fp_metadata;
  FILE * fp_data;
  FILE * fg_out;
  FILE * fp_out, * fd_out, * fb_out;
  FILE * fp_psi_out, * fp_rho0_out, * fp_rho1_out;
  FILE * fp_elf_out, * fp_elc_out;


  /* Check the command line, then parse the meta data information,
   * and sort out the data file name  */

  if (argc != 3) {
    printf("Usage %s meta-file data-file\n", argv[0]);
    exit(-1);
  }

  read_meta_data_file(argv[1]);
  ntime = read_data_file_name(argv[2]);

  /* Work out parallel decompsoition that was used and the number of
   * processors per I/O group. */

  for (i = 0; i < 3; i++) {
    nlocal[i] = ntotal[i]/pe_[i];
  }

  pe_per_io = pe_[0]*pe_[1]*pe_[2]/nio_;


  /* Allocate storage */

  n = nrec_*nlocal[0]*nlocal[1]*nlocal[2];
  datalocal = (double *) calloc(n, sizeof(double));
  if (datalocal == NULL) printf("calloc(datalocal) failed\n");
    
  /* No. sites in the target section (always original at moment) */

  for (i = 0; i < 3; i++) {
    ntargets[i] = ntotal[i];
  }
 
  n = nrec_*ntargets[0]*ntargets[1]*ntargets[2];
  datasection = (double *) calloc(n, sizeof(double));
  if (datasection == NULL) printf("calloc(datasection) failed\n");

  /* For electrokinetics post-processing cannot be done in place */
  /* and we need to allocate a section dedicated to post-processing */

  if (output_ek_elf_ == 1) npp_ += 3;

  n = npp_*ntargets[0]*ntargets[1]*ntargets[2];

  ppsection = (double *) calloc(n, sizeof(double));
  if (ppsection == NULL) printf("calloc(ppsection) failed\n");

  /* LE displacements as function of x */
  le_displace_ = le_speed_*(double) (ntime - le_t0_);
  le_displacements_ = (double *) malloc(ntotal[0]*sizeof(double));
  le_duy_ = (double *) malloc(ntotal[0]*sizeof(double));
  if (le_displacements_ == NULL) printf("malloc(le_displacements_)\n");
  if (le_duy_ == NULL) printf("malloc(le_duy_) failed\n");
  le_set_displacements();


  /* Main loop */

  for (n = 1; n <= nio_; n++) {

    /* Open metadata file and skip 12 lines to get to the
     * decomposition inofmration */

    sprintf(io_metadata, "%s.%3.3d-%3.3d.meta", stub_, nio_, n);
    printf("Reading metadata file ... %s ", io_metadata);

    fp_metadata = fopen(io_metadata, "r");
    if (fp_metadata == NULL) printf("fopen(%s) failed\n", io_metadata);

    for (p = 0; p < 12; p++) {
      fgets(line, FILENAME_MAX, fp_metadata);
      printf("%s", line);
    }

    /* Open the current data file */

    sprintf(io_data, "%s-%8.8d.%3.3d-%3.3d", stub_, ntime, nio_, n);
    printf("-> %s\n", io_data);

    fp_data = fopen(io_data, "r+b");
    if (fp_data == NULL) printf("fopen(%s) failed\n", io_data);

    /* Read data file based on offsets recorded in the metadata,
     * then copy this section to the global array */

    for (p = 0; p < pe_per_io; p++) {

      ifail = fscanf(fp_metadata, "%d %d %d %d %d %d %d %d %d %d",
		     &rank, &cx, &cy, &cz, &nlx, &nly, &nlz, &nox, &noy, &noz);
      assert(ifail == 10);

      read_data(fp_data, nlocal, datalocal);
      copy_data(datalocal, datasection);
    }

    fclose(fp_data);
    fclose(fp_metadata);
  }

  /* Unroll the data if Lees Edwards planes are present */

  if (nplanes_ > 0) {
    printf("Unrolling LE planes from centre (displacement %f)\n",
	   le_displace_);
    le_unroll(datasection);
  }


  /* Write output */

  /* Electrokinetic quantities */
  if (nrec_ == 4 && strncmp(stub_, "psi", 3) == 0) {

    if (output_ek_elc_ != 0 || output_ek_elf_ != 0) {
      electric_field(datasection, ppsection);
    }

    /* Write separately the five quantities in five different files
       Change the flags at the top in order to select the output  */

    if (output_ek_psi_ != 0) {
      sprintf(psi_filename,"psi-%s-%8.8d.vtk", stub_, ntime);
      fp_psi_out = fopen(psi_filename, "w");
      if (fp_psi_out == NULL) printf("fopen(%s) failed\n", psi_filename);
      if(vtk_header) {
	fprintf(fp_psi_out, "# vtk DataFile Version 2.0\n");
	fprintf(fp_psi_out, "Generated by vtk_extract.c\n");
	fprintf(fp_psi_out, "ASCII\n");
	fprintf(fp_psi_out, "DATASET STRUCTURED_POINTS\n");
	fprintf(fp_psi_out, "DIMENSIONS %d %d %d\n", ntotal[0], ntotal[1], ntotal[2]);
	fprintf(fp_psi_out, "ORIGIN %d %d %d\n", 0, 0, 0);
	fprintf(fp_psi_out, "SPACING %d %d %d\n", 1, 1, 1);
	fprintf(fp_psi_out, "POINT_DATA %d\n", ntotal[0]*ntotal[1]*ntotal[2]);
	fprintf(fp_psi_out, "SCALARS el_potential float %d\n", 1);
	fprintf(fp_psi_out, "LOOKUP_TABLE default\n");
      }
      printf("... Writing result to %s\n", psi_filename);
    }

    if (output_ek_rho_ != 0) {

      sprintf(rho0_filename,"rho0-%s-%8.8d.vtk", stub_, ntime);
      fp_rho0_out = fopen(rho0_filename, "w");
      if (fp_rho0_out == NULL) printf("fopen(%s) failed\n", rho0_filename);
      if(vtk_header) {
	fprintf(fp_rho0_out, "# vtk DataFile Version 2.0\n");
	fprintf(fp_rho0_out, "Generated by vtk_extract.c\n");
	fprintf(fp_rho0_out, "ASCII\n");
	fprintf(fp_rho0_out, "DATASET STRUCTURED_POINTS\n");
	fprintf(fp_rho0_out, "DIMENSIONS %d %d %d\n", ntotal[0], ntotal[1], ntotal[2]);
	fprintf(fp_rho0_out, "ORIGIN %d %d %d\n", 0, 0, 0);
	fprintf(fp_rho0_out, "SPACING %d %d %d\n", 1, 1, 1);
	fprintf(fp_rho0_out, "POINT_DATA %d\n", ntotal[0]*ntotal[1]*ntotal[2]);
	fprintf(fp_rho0_out, "SCALARS rho0 float %d\n", 1);
	fprintf(fp_rho0_out, "LOOKUP_TABLE default\n");
      }
      printf("... Writing result to %s\n", rho0_filename);

      sprintf(rho1_filename,"rho1-%s-%8.8d.vtk", stub_, ntime);
      fp_rho1_out = fopen(rho1_filename, "w");
      if (fp_rho1_out == NULL) printf("fopen(%s) failed\n", rho1_filename);
      if(vtk_header) {
	fprintf(fp_rho1_out, "# vtk DataFile Version 2.0\n");
	fprintf(fp_rho1_out, "Generated by vtk_extract.c\n");
	fprintf(fp_rho1_out, "ASCII\n");
	fprintf(fp_rho1_out, "DATASET STRUCTURED_POINTS\n");
	fprintf(fp_rho1_out, "DIMENSIONS %d %d %d\n", ntotal[0], ntotal[1], ntotal[2]);
	fprintf(fp_rho1_out, "ORIGIN %d %d %d\n", 0, 0, 0);
	fprintf(fp_rho1_out, "SPACING %d %d %d\n", 1, 1, 1);
	fprintf(fp_rho1_out, "POINT_DATA %d\n", ntotal[0]*ntotal[1]*ntotal[2]);
	fprintf(fp_rho1_out, "SCALARS rho1 float %d\n", 1);
	fprintf(fp_rho1_out, "LOOKUP_TABLE default\n");
      }
      printf("... Writing result to %s\n", rho1_filename);

    }

    if (output_ek_elc_ != 0) {
      sprintf(elc_filename,"elc-%s-%8.8d.vtk", stub_, ntime);
      fp_elc_out = fopen(elc_filename, "w");
      if (fp_elc_out == NULL) printf("fopen(%s) failed\n", elc_filename);
      if(vtk_header) {
	fprintf(fp_elc_out, "# vtk DataFile Version 2.0\n");
	fprintf(fp_elc_out, "Generated by vtk_extract.c\n");
	fprintf(fp_elc_out, "ASCII\n");
	fprintf(fp_elc_out, "DATASET STRUCTURED_POINTS\n");
	fprintf(fp_elc_out, "DIMENSIONS %d %d %d\n", ntotal[0], ntotal[1], ntotal[2]);
	fprintf(fp_elc_out, "ORIGIN %d %d %d\n", 0, 0, 0);
	fprintf(fp_elc_out, "SPACING %d %d %d\n", 1, 1, 1);
	fprintf(fp_elc_out, "POINT_DATA %d\n", ntotal[0]*ntotal[1]*ntotal[2]);
	fprintf(fp_elc_out, "SCALARS elc float %d\n", 1);
	fprintf(fp_elc_out, "LOOKUP_TABLE default\n");
      }
      printf("... Writing result to %s\n", elc_filename);
    }

    if (output_ek_elf_ != 0) {
      sprintf(elf_filename,"elf-%s-%8.8d.vtk", stub_, ntime);
      fp_elf_out = fopen(elf_filename, "w");
      if (fp_elf_out == NULL) printf("fopen(%s) failed\n", elf_filename);
      if(vtk_header) {
    	fprintf(fp_elf_out, "# vtk DataFile Version 2.0\n");
	fprintf(fp_elf_out, "Generated by vtk_extract.c\n");
	fprintf(fp_elf_out, "ASCII\n");
	fprintf(fp_elf_out, "DATASET STRUCTURED_POINTS\n");
	fprintf(fp_elf_out, "DIMENSIONS %d %d %d\n", ntotal[0], ntotal[1], ntotal[2]);
	fprintf(fp_elf_out, "ORIGIN %d %d %d\n", 0, 0, 0);
	fprintf(fp_elf_out, "SPACING %d %d %d\n", 1, 1, 1);
	fprintf(fp_elf_out, "POINT_DATA %d\n", ntotal[0]*ntotal[1]*ntotal[2]);
	fprintf(fp_elf_out, "VECTORS elf float\n");
      }
      printf("... Writing result to %s\n", elf_filename);
    }

    if(output_cmf_ == 0) write_data_psi(fp_psi_out, fp_rho0_out, fp_rho1_out,
	fp_elc_out, fp_elf_out, ntargets, datasection, ppsection);
    if(output_cmf_ == 1) write_data_psi_cmf(fp_psi_out, fp_rho0_out, fp_rho1_out, 
	fp_elc_out, fp_elf_out, ntargets, datasection, ppsection);

    if (output_ek_psi_ != 0) fclose(fp_psi_out);
    if (output_ek_rho_ != 0) {
			     fclose(fp_rho0_out);
			     fclose(fp_rho1_out);
    }
    if (output_ek_elc_ != 0) fclose(fp_elc_out);
    if (output_ek_elf_ != 0) fclose(fp_elf_out);

  }

  /* Tensor order parameter */
  else if (nrec_ == 5 && strncmp(stub_, "q", 1) == 0) {

  /* Diagonalization of the Q matrix and calculation of scalar director biaxial op */

  order_parameters(datasection);

  /* Write separately the three quantities in three different files
     Change the flags at the top in order to select the output  */

    if (output_lc_op_ != 0) {
      sprintf(opfilename,"op-%s-%8.8d.vtk", stub_, ntime);
      fp_out = fopen(opfilename, "w");
      if (fp_out == NULL) printf("fopen(%s) failed\n", opfilename);
      if(vtk_header) {
	fprintf(fp_out, "# vtk DataFile Version 2.0\n");
	fprintf(fp_out, "Generated by vtk_extract.c\n");
	fprintf(fp_out, "ASCII\n");
	fprintf(fp_out, "DATASET STRUCTURED_POINTS\n");
	fprintf(fp_out, "DIMENSIONS %d %d %d\n", ntotal[0], ntotal[1], ntotal[2]);
	fprintf(fp_out, "ORIGIN %d %d %d\n", 0, 0, 0);
	fprintf(fp_out, "SPACING %d %d %d\n", 1, 1, 1);
	fprintf(fp_out, "POINT_DATA %d\n", ntotal[0]*ntotal[1]*ntotal[2]);
	fprintf(fp_out, "SCALARS scalar_op float %d\n", 1);
	fprintf(fp_out, "LOOKUP_TABLE default\n");
      }
      printf("... Writing result to %s\n", opfilename);
    }

    if (output_lc_dir_ != 0) {
      sprintf(dirfilename,"dir-%s-%8.8d.vtk", stub_, ntime);
      fd_out = fopen(dirfilename, "w");
      if (fd_out == NULL) printf("fopen(%s) failed\n", dirfilename);
      if(vtk_header) {
	fprintf(fd_out, "# vtk DataFile Version 2.0\n");
	fprintf(fd_out, "Generated by vtk_extract.c\n");
	fprintf(fd_out, "ASCII\n");
	fprintf(fd_out, "DATASET STRUCTURED_POINTS\n");
	fprintf(fd_out, "DIMENSIONS %d %d %d\n", ntotal[0], ntotal[1], ntotal[2]);
	fprintf(fd_out, "ORIGIN %d %d %d\n", 0, 0, 0);
	fprintf(fd_out, "SPACING %d %d %d\n", 1, 1, 1);
	fprintf(fd_out, "POINT_DATA %d\n", ntotal[0]*ntotal[1]*ntotal[2]);
	fprintf(fd_out, "VECTORS director float\n");
      }
      printf("... Writing result to %s\n", dirfilename);
    }

    if (output_lc_biop_ != 0) {
      sprintf(biopfilename,"biop-%s-%8.8d.vtk", stub_, ntime);
      fb_out = fopen(biopfilename, "w");
      if (fb_out == NULL) printf("fopen(%s) failed\n", biopfilename);
      if(vtk_header) {
	fprintf(fb_out, "# vtk DataFile Version 2.0\n");
	fprintf(fb_out, "Generated by vtk_extract.c\n");
	fprintf(fb_out, "ASCII\n");
	fprintf(fb_out, "DATASET STRUCTURED_POINTS\n");
	fprintf(fb_out, "DIMENSIONS %d %d %d\n", ntotal[0], ntotal[1], ntotal[2]);
	fprintf(fb_out, "ORIGIN %d %d %d\n", 0, 0, 0);
	fprintf(fb_out, "SPACING %d %d %d\n", 1, 1, 1);
	fprintf(fb_out, "POINT_DATA %d\n", ntotal[0]*ntotal[1]*ntotal[2]);
	fprintf(fb_out, "SCALARS biaxial_op float %d\n", 1);
	fprintf(fb_out, "LOOKUP_TABLE default\n");
      }
      printf("... Writing result to %s\n", biopfilename);
    }

    if(output_cmf_ == 0) write_data_q(fp_out, fd_out, fb_out, ntargets, datasection);
    if(output_cmf_ == 1) write_data_q_cmf(fp_out, fd_out, fb_out, ntargets, datasection);

    if (output_lc_op_ != 0) fclose(fp_out);
    if (output_lc_dir_ != 0) fclose(fd_out);
    if (output_lc_biop_ != 0) fclose(fb_out);

  }

  /* Generic data file */
  else {

    sprintf(genfilename, "%s-%8.8d.vtk", stub_, ntime);

    fg_out = fopen(genfilename, "w");

    if (fg_out == NULL) printf("fopen(%s) failed\n", genfilename);
    if(vtk_header) {
      fprintf(fg_out, "# vtk DataFile Version 2.0\n");
      fprintf(fg_out, "Generated by vtk_extract.c\n");
      fprintf(fg_out, "ASCII\n");
      fprintf(fg_out, "DATASET STRUCTURED_POINTS\n");
      fprintf(fg_out, "DIMENSIONS %d %d %d\n", ntotal[0], ntotal[1], ntotal[2]);
      fprintf(fg_out, "ORIGIN %d %d %d\n", 0, 0, 0);
      fprintf(fg_out, "SPACING %d %d %d\n", 1, 1, 1);
      fprintf(fg_out, "POINT_DATA %d\n", ntotal[0]*ntotal[1]*ntotal[2]);
      if (nrec_ == 1 && strncmp(stub_, "phi", 1) == 0) {
	fprintf(fg_out, "SCALARS something float %d\n", nrec_);
	fprintf(fg_out, "LOOKUP_TABLE default\n");
      }
      if (nrec_ == 3 && strncmp(stub_, "vel", 3) == 0) {
	fprintf(fg_out, "VECTORS %s float\n", stub_);
      }
    }

    printf("... Writing result to %s\n", genfilename);

    if(output_cmf_ == 0) write_data(fg_out, ntargets, datasection);
    if(output_cmf_ == 1) write_data_cmf(fg_out, ntargets, datasection);

    fclose(fg_out);

  }

  free(datalocal);
  free(le_displacements_);

  return 0;

}

/****************************************************************************
 *
 *  read_meta_data_file
 *
 *  This sets a number of the global variable from the meta data file.
 *
 ****************************************************************************/

void read_meta_data_file(const char * filename) {

  int npe, nrbyte;
  int ifail;
  char tmp[FILENAME_MAX];
  FILE * fp_meta;
  const int ncharoffset = 33;

  fp_meta = fopen(filename, "r");
  if (fp_meta == NULL) {
    printf("fopen(%s) failed\n", filename);
    exit(-1);
  }

  fgets(tmp, FILENAME_MAX, fp_meta);
  ifail = sscanf(tmp+ncharoffset, "%s\n", stub_);
  assert(ifail == 1);
  printf("Read stub: %s\n", stub_);
  fgets(tmp, FILENAME_MAX, fp_meta);

  fgets(tmp, FILENAME_MAX, fp_meta);
  ifail = sscanf(tmp+ncharoffset, "%d\n", &nrbyte);
  assert(ifail == 1);
  printf("Record size (bytes): %d\n", nrbyte);
  assert((nrbyte % 8) == 0);
  nrec_ = nrbyte/8;

  fgets(tmp, FILENAME_MAX, fp_meta);
  ifail = sscanf(tmp+ncharoffset, "%d", &input_isbigendian_);
  assert(ifail == 1);
  assert(input_isbigendian_ == 0 || input_isbigendian_ == 1);

  if(input_isbigendian_ != is_bigendian())reverse_byte_order_ = 1;
  
  fgets(tmp, FILENAME_MAX, fp_meta);
  ifail = sscanf(tmp+ncharoffset, "%d\n", &npe);
  assert(ifail == 1);
  printf("Total number of processors %d\n", npe);

  fgets(tmp, FILENAME_MAX, fp_meta);
  ifail = sscanf(tmp+ncharoffset, "%d %d %d", pe_, pe_+1, pe_+2);
  assert(ifail == 3);
  printf("Decomposition is %d %d %d\n", pe_[0], pe_[1], pe_[2]);
  assert(npe == pe_[0]*pe_[1]*pe_[2]);

  fgets(tmp, FILENAME_MAX, fp_meta);
  ifail = sscanf(tmp+ncharoffset, "%d %d %d", ntotal, ntotal+1, ntotal+2);
  assert(ifail == 3);
  printf("System size is %d %d %d\n", ntotal[0], ntotal[1], ntotal[2]);

  fgets(tmp, FILENAME_MAX, fp_meta);
  ifail = sscanf(tmp+ncharoffset, "%d", &nplanes_);
  assert(ifail == 1);
  assert(nplanes_ >= 0);
  printf("Number of Lees Edwards planes %d\n", nplanes_);
  fgets(tmp, FILENAME_MAX, fp_meta);
  sscanf(tmp+ncharoffset, "%lf", &le_speed_);
  printf("Lees Edwards speed: %f\n", le_speed_);

  /* Number of I/O groups */
  fgets(tmp, FILENAME_MAX, fp_meta);
  sscanf(tmp+ncharoffset, "%d", &nio_);
  printf("Number of I/O groups: %d\n", nio_);

  fclose(fp_meta);

  /* Is this the velocity field? */

  if (nrec_ == 3 && strncmp(filename, "vel", 3) == 0) {
    is_velocity_ = 1;
    printf("\nThis is a velocity field\n");
  }

  return;
}

/****************************************************************************
 *
 *  read_data_file_name
 *
 *  This replies on a filename of the form stub-nnnnnn.001-001
 *  to identify the time step ('nnnnnnn').
 *
 ****************************************************************************/

int read_data_file_name(const char * filename) {

  int ntime = -1;
  char * tmp;
  
  tmp = strchr(filename, '-');
  if (tmp) {
    sscanf(tmp+1, "%d.", &ntime);
  }

  assert (ntime >= 0);
  return ntime;
}

/****************************************************************************
 *
 *  copy_data
 *
 *  Copy the local data into the global array in the correct place.
 *
 ****************************************************************************/

int copy_data(double * datalocal, double * datasection) {

  int ic, jc, kc, ic_g, jc_g, kc_g;
  int index_l, index_s, nr;

  /* Sweep through the local sites for this pe */

  for (ic = 1; ic <= nlocal[0]; ic++) {
    ic_g = nox + ic;
    for (jc = 1; jc <= nlocal[1]; jc++) {
      jc_g = noy + jc;
      for (kc = 1; kc <= nlocal[2]; kc++) {
	kc_g = noz + kc;
	index_s = nrec_*site_index(ic_g, jc_g, kc_g, ntargets);

	for (nr = 0; nr < nrec_; nr++) {
	  index_l = nrec_*site_index(ic, jc, kc, nlocal);
	  *(datasection + index_s + nr) = *(datalocal + index_l + nr);
	}
      }
    }
  }

  return 0;
}


/****************************************************************************
 *
 *  read_data
 *
 *  Read data block relevant for one pe
 *
 ****************************************************************************/

void read_data(FILE * fp_data, int n[3], double * data) {

  int ic, jc, kc, index, nr;
  double phi;
  double revphi;


  if (input_binary_) {
    for (ic = 1; ic <= n[0]; ic++) {
      for (jc = 1; jc <= n[1]; jc++) {
	for (kc = 1; kc <= n[2]; kc++) {
	  index = site_index(ic, jc, kc, nlocal);

	  for (nr = 0; nr < nrec_; nr++) {
	    fread(&phi, sizeof(double), 1, fp_data);
	    if(reverse_byte_order_){
	       revphi = reverse_byte_order_double((char *) &phi); 
	       phi = revphi;
	    }
	    *(data + nrec_*index + nr) = phi;
	  }
	}
      }
    }
  }
  else {
    for (ic = 1; ic <= n[0]; ic++) {
      for (jc = 1; jc <= n[1]; jc++) {
	for (kc = 1; kc <= n[2]; kc++) {
	  index = site_index(ic, jc, kc, nlocal);

	  for (nr = 0; nr < nrec_; nr++) {
	    fscanf(fp_data, "%le", data + nrec_*index + nr);
	  }
	}
      }
    }
  }

  return;
}


/****************************************************************************
 *
 *  write_data
 *
 *  Write contiguous block of (float) data.
 *
 ****************************************************************************/

void write_data(FILE * fp_data, int n[3], double * data) {

  int ic, jc, kc, index, nr;

  index = 0;

  if (output_binary_) {
    for (ic = 0; ic < n[0]; ic++) {
      for (jc = 0; jc < n[1]; jc++) {
        for (kc = 0; kc < n[2]; kc++) {
          for (nr = 0; nr < nrec_; nr++) {
            fwrite(data + index, sizeof(double), 1, fp_data);
            index++;
          }
        }
      }
    }
  }
  else {
    for (ic = 0; ic < n[0]; ic++) {
      for (jc = 0; jc < n[1]; jc++) {
        for (kc = 0; kc < n[2]; kc++) {

          if (output_index_) {
            /* Add the global (i,j,k) index starting at 1 each way */
            fprintf(fp_data, "%4d %4d %4d ", 1 + ic, 1 + jc, 1 + kc);
          }

          for (nr = 0; nr < nrec_ - 1; nr++) {
            fprintf(fp_data, "%13.6e ", *(data + index));
            index++;
          }
          fprintf(fp_data, "%13.6e\n", *(data + index));
          index++;
        }
      }
    }
  }

  return;
}


/*****************************************************************************
 *
 *  write_data_cmf
 *
 *  Loop around the sites in 'reverse' order (column major format).
 *
 *****************************************************************************/

void write_data_cmf(FILE * fp_data, int n[3], double * data) {

  int ic, jc, kc, index, nr;

  index = 0;

  if (output_binary_) {
    for (kc = 1; kc <= n[2]; kc++) {
      for (jc = 1; jc <= n[1]; jc++) {
        for (ic = 1; ic <= n[0]; ic++) {
          index = site_index(ic, jc, kc, n);
          for (nr = 0; nr < nrec_; nr++) {
            fwrite(data + nrec_*index + nr, sizeof(double), 1, fp_data);
          }
        }
      }
    }
  }
  else {
    for (kc = 1; kc <= n[2]; kc++) {
      for (jc = 1; jc <= n[1]; jc++) {
        for (ic = 1; ic <= n[0]; ic++) {
          index = site_index(ic, jc, kc, n);

          if (output_index_) {
             /* Add the global (i,j,k) index starting at 1 each way */
             fprintf(fp_data, "%4d %4d %4d ", ic, jc, kc);
           }
          for (nr = 0; nr < nrec_ - 1; nr++) {
            fprintf(fp_data, "%13.6e ", *(data + nrec_*index + nr));
          }
          fprintf(fp_data, "%13.6e\n", *(data + nrec_*index + nr));
        }
      }
    }
  }

  return;
}


/****************************************************************************
 *
 *  write_data_psi
 *
 *  Write the potential, the number densities of the species 
 *  (at the moment only two) and the total electric charge density
 *  (depending on the initial flags).
 *
 ****************************************************************************/


void write_data_psi(FILE * fp_psi_out, FILE * fp_rho0_out, FILE * fp_rho1_out, 
	FILE * fp_elc_out, FILE * fp_elf_out, int n[3], double * data, double * pp) {

  int ic, jc, kc, index0, index1, nr;

  index0 = 0;
  index1 = 0;

  if (output_binary_) {

    for (ic = 0; ic < n[0]; ic++) {
      for (jc = 0; jc < n[1]; jc++) {
	for (kc = 0; kc < n[2]; kc++) {

          if (output_ek_psi_ != 0) {
	    fwrite(data + index0, sizeof(double), 1, fp_psi_out);
	  }
	  index0++;
	  if (output_ek_rho_ != 0) {
	    fwrite(data + index0, sizeof(double), 1, fp_rho0_out);
	  }
	  index0++;
	  if (output_ek_rho_ != 0) {
	    fwrite(data + index0, sizeof(double), 1, fp_rho1_out);
	  }
	  index0++;
	  if (output_ek_elc_ != 0) {
	    fwrite(data + index0, sizeof(double), 1, fp_elc_out);
	  }
	  index0++;
	  if (output_ek_elc_ != 0) {
	    fwrite(data + index0, sizeof(double), 1, fp_elc_out);
	  }
	  index0++;
	  if (output_ek_elf_ != 0) {
	    fwrite(pp + index1, sizeof(double), 1, fp_elf_out);
	  }
	  index1++;

	}
      }
    }

  }

  else {

    for (ic = 0; ic < n[0]; ic++) {
      for (jc = 0; jc < n[1]; jc++) {
	for (kc = 0; kc < n[2]; kc++) {

	  if (output_index_) {
	    /* Add the global (i,j,k) index starting at 1 each way */
            if (output_ek_psi_ != 0) fprintf(fp_psi_out, "%4d %4d %4d ", 1+ic, 1+jc, 1+kc);
            if (output_ek_rho_ != 0) {
				     fprintf(fp_rho0_out, "%4d %4d %4d ", 1+ic, 1+jc, 1+kc);
				     fprintf(fp_rho1_out, "%4d %4d %4d ", 1+ic, 1+jc, 1+kc);
	    }
            if (output_ek_elc_ != 0) fprintf(fp_elc_out, "%4d %4d %4d ", 1+ic, 1+jc, 1+kc);
            if (output_ek_elf_ != 0) fprintf(fp_elf_out, "%4d %4d %4d ", 1+ic, 1+jc, 1+kc);
	  }

          if (output_ek_psi_ != 0) {
	    fprintf(fp_psi_out, "%13.6e\n", *(data + index0));
	  }
	  index0++;
          if (output_ek_rho_ != 0) {
	    fprintf(fp_rho0_out, "%13.6e\n", *(data + index0));
	  }
	  index0++;
          if (output_ek_rho_ != 0) {
	    fprintf(fp_rho1_out, "%13.6e\n", *(data + index0));
	  }
	  index0++;
          if (output_ek_elc_ != 0) {
	    fprintf(fp_elc_out, "%13.6e\n", *(data + index0));
	  }
	  index0++;
          if (output_ek_elf_ != 0) {

	    for (nr = 0; nr < 2; nr++) {
	      fprintf(fp_elf_out, "%13.6e ", *(pp + index1));
	      index1++;
	    }
	    fprintf(fp_elf_out, "%13.6e\n", *(pp + index1));
	    index1++;
	  }

	}
      }
    }

  }
  return;
}

/*****************************************************************************
 *
 *  write_data_psi_cmf
 *
 *  Loop around the sites in 'reverse' order (column major format).
 *
 *****************************************************************************/

void write_data_psi_cmf(FILE * fp_psi_out, FILE * fp_rho0_out, FILE * fp_rho1_out, 
	FILE * fp_elc_out, FILE * fp_elf_out, int n[3], double * data, double * pp) {

  int ic, jc, kc, index0, index1, nr;

  index0 = 0;

  if (output_binary_) {

    for (kc = 1; kc <= n[2]; kc++) {
      for (jc = 1; jc <= n[1]; jc++) {
	for (ic = 1; ic <= n[0]; ic++) {

          index0 = nrec_*site_index(ic, jc, kc, n);
          index1 = npp_*site_index(ic, jc, kc, n);

          if (output_ek_psi_ != 0) {
	    fwrite(data + index0, sizeof(double), 1, fp_psi_out);
	  }
	  index0++;
          if (output_ek_rho_ != 0) {
	    fwrite(data + index0, sizeof(double), 1, fp_rho0_out);
	  }
	  index0++;
          if (output_ek_rho_ != 0) {
	    fwrite(data + index0, sizeof(double), 1, fp_rho1_out);
	  }
	  index0++;
          if (output_ek_elc_ != 0) {
	    fwrite(data + index0, sizeof(double), 1, fp_elc_out);
	  }
	  index0++;
          if (output_ek_elf_ != 0) {
	    fwrite(pp + index1, sizeof(double), 1, fp_elf_out);
	  }
	  index1++;

	}
      }
    }

  }

  else {

    for (kc = 1; kc <= n[2]; kc++) {
      for (jc = 1; jc <= n[1]; jc++) {
	for (ic = 1; ic <= n[0]; ic++) {

          index0 = nrec_*site_index(ic, jc, kc, n);
          index1 = npp_*site_index(ic, jc, kc, n);

	  if (output_index_) {
	    /* Add the global (i,j,k) index starting at 1 each way */
            if (output_ek_psi_ != 0) fprintf(fp_psi_out, "%4d %4d %4d ", ic, jc, kc);
            if (output_ek_rho_ != 0) {
				     fprintf(fp_rho0_out, "%4d %4d %4d ", ic, jc, kc);
				     fprintf(fp_rho1_out, "%4d %4d %4d ", ic, jc, kc);
	    }
            if (output_ek_elc_ != 0) fprintf(fp_elc_out, "%4d %4d %4d ", ic, jc, kc);
            if (output_ek_elf_ != 0) fprintf(fp_elf_out, "%4d %4d %4d ", ic, jc, kc);
	  }

          if (output_ek_psi_ != 0) {
	    fprintf(fp_psi_out, "%13.6e\n", *(data + index0));
	  }
	  index0++;
	  if (output_ek_rho_ != 0) {
	    fprintf(fp_rho0_out, "%13.6e\n", *(data + index0));
	  }
	  index0++;
	  if (output_ek_rho_ != 0) {
	    fprintf(fp_rho1_out, "%13.6e\n", *(data + index0));
	  }
	  index0++;
	  if (output_ek_elc_ != 0) {
	    fprintf(fp_elc_out, "%13.6e\n", *(data + index0));
	  }
	  index0++;
	  if (output_ek_elf_ != 0) {
	  for (nr = 0; nr < 2; nr++) {
	    fprintf(fp_elf_out, "%13.6e ", *(pp + index1));
	    index1++;
	  }
	    fprintf(fp_elf_out, "%13.6e\n", *(pp + index1));
	    index1++;
	  }

	}
      }
    }

  }
  return;
}

/****************************************************************************
 *
 *  write_data_q
 *
 *  Write the scalar op, the director and the biaxial op (depending on
 *  the initial flags).
 *
 ****************************************************************************/


void write_data_q(FILE * fp_out, FILE * fd_out, FILE * fb_out, 
	int n[3], double * data) {

  int ic, jc, kc, index, nr;

  index = 0;

  if (output_binary_) {

    for (ic = 0; ic < n[0]; ic++) {
      for (jc = 0; jc < n[1]; jc++) {
	for (kc = 0; kc < n[2]; kc++) {

          if (output_lc_op_ != 0) {
	    fwrite(data + index, sizeof(double), 1, fp_out);
	  }
	  index++;

	  for (nr = 0; nr < 3; nr++) {
	    if (output_lc_dir_ != 0) {
	      fwrite(data + index, sizeof(double), 1, fd_out);
	    }
	    index++;
	  }

          if (output_lc_biop_ != 0) {
	    fwrite(data + index, sizeof(double), 1, fb_out);

	  }
	  index++;

	}
      }
    }

  }

  else {

    for (ic = 0; ic < n[0]; ic++) {
      for (jc = 0; jc < n[1]; jc++) {
	for (kc = 0; kc < n[2]; kc++) {

	  if (output_index_) {
	    /* Add the global (i,j,k) index starting at 1 each way */
            if (output_lc_op_ != 0) fprintf(fp_out, "%4d %4d %4d ", 1+ic, 1+jc, 1+kc);
            if (output_lc_dir_ != 0) fprintf(fd_out, "%4d %4d %4d ", 1+ic, 1+jc, 1+kc);
            if (output_lc_biop_ != 0) fprintf(fb_out, "%4d %4d %4d ", 1+ic, 1+jc, 1+kc);
	  }

          if (output_lc_op_ != 0) {
	    fprintf(fp_out, "%13.6e\n", *(data + index));
	  }
	  index++;

	  if (*(data + index) > 0) {
	    for (nr = 0; nr < 2; nr++) {
	      if (output_lc_dir_ != 0) {
		fprintf(fd_out, "%13.6e ", *(data + index));
	      }
	      index++;
	    }
	    if (output_lc_dir_ != 0) {
	      fprintf(fd_out, "%13.6e\n", *(data + index));
	    }
	    index++;
	  }
	  else {
	    for (nr = 0; nr < 2; nr++) {
	      if (output_lc_dir_ != 0) {
		fprintf(fd_out, "%13.6e ", -*(data + index));
	      }
	      index++;
	    }
	    if (output_lc_dir_ != 0) {
	      fprintf(fd_out, "%13.6e\n", -*(data + index));
	    }
	    index++;
	  }

          if (output_lc_biop_ != 0) {
	    fprintf(fb_out, "%13.6e\n", *(data + index));
	  }
	  index++;

	}
      }
    }
  }

  return;
}

/*****************************************************************************
 *
 *  write_data_q_cmf
 *
 *  Loop around the sites in 'reverse' order (column major format).
 *
 *****************************************************************************/

void write_data_q_cmf(FILE * fp_out, FILE * fd_out, FILE * fb_out, 
	int n[3], double * data) {

  int ic, jc, kc, index, nr;

  index = 0;

  if (output_binary_) {

    for (kc = 1; kc <= n[2]; kc++) {
      for (jc = 1; jc <= n[1]; jc++) {
	for (ic = 1; ic <= n[0]; ic++) {
          index = nrec_*site_index(ic, jc, kc, n);

          if (output_lc_op_ != 0) {
	    fwrite(data + index, sizeof(double), 1, fp_out);
	  }
	  index++;

	  for (nr = 0; nr < 3; nr++) {
	    if (output_lc_dir_ != 0) {
	      fwrite(data + index, sizeof(double), 1, fd_out);
	    }
	    index++;
	  }

          if (output_lc_biop_ != 0) {
	    fwrite(data + index, sizeof(double), 1, fb_out);

	  }
	  index++;

	}
      }
    }

  }

  else {

    for (kc = 1; kc <= n[2]; kc++) {
      for (jc = 1; jc <= n[1]; jc++) {
	for (ic = 1; ic <= n[0]; ic++) {
          index = nrec_*site_index(ic, jc, kc, n);

	  if (output_index_) {
	    /* Add the global (i,j,k) index starting at 1 each way */
            if (output_lc_op_ != 0) fprintf(fp_out, "%4d %4d %4d ", ic, jc, kc);
            if (output_lc_dir_ != 0) fprintf(fd_out, "%4d %4d %4d ", ic, jc, kc);
            if (output_lc_biop_ != 0) fprintf(fb_out, "%4d %4d %4d ", ic, jc, kc);
	  }

          if (output_lc_op_ != 0) {
	    fprintf(fp_out, "%13.6e\n", *(data + index));
	  }
	  index++;

	  if (*(data + index) > 0) {
	    for (nr = 0; nr < 2; nr++) {
	      if (output_lc_dir_ != 0) {
		fprintf(fd_out, "%13.6e ", *(data + index));
	      }
	      index++;
	    }
	    if (output_lc_dir_ != 0) {
	      fprintf(fd_out, "%13.6e\n", *(data + index));
	    }
	    index++;
	  }
	  else {
	    for (nr = 0; nr < 2; nr++) {
	      if (output_lc_dir_ != 0) {
		fprintf(fd_out, "%13.6e ", -*(data + index));
	      }
	      index++;
	    }
	    if (output_lc_dir_ != 0) {
	      fprintf(fd_out, "%13.6e\n", -*(data + index));
	    }
	    index++;
	  }

          if (output_lc_biop_ != 0) {
	    fprintf(fb_out, "%13.6e\n", *(data + index));
	  }
	  index++;

	}
      }
    }
  }

  return;
}

/****************************************************************************
 *
 *  le_displacement
 *
 ****************************************************************************/

int le_displacement(int ic, int jc) {

    int dj;

    dj = jc + le_displacements_[ic-1];

    dj = dj % ntotal[1];
    if (dj > ntargets[1]) dj -= ntargets[1];
    if (dj < 1) dj += ntargets[1];

    return dj;
}

/****************************************************************************
 *
 *  le_set_displacements
 *
 ****************************************************************************/

void le_set_displacements() {

  int ic;
  int di;
  double dy;
  double duy;

  for (ic = 0; ic < ntotal[0]; ic++) {
    le_displacements_[ic] = 0.0;
    le_duy_[ic] = 0.0;
  }

  if (nplanes_ > 0) {

    di = ntotal[0] / nplanes_;
    dy = -(nplanes_/2.0)*le_displace_;
    duy = -(nplanes_/2.0)*le_speed_;

    /* Fist half block */
    for (ic = 1; ic <= di/2; ic++) {
      le_displacements_[ic-1] = dy;
      le_duy_[ic-1] = duy;
    }

    dy += le_displace_;
    duy += le_speed_;
    for (ic = di/2 + 1; ic <= ntotal[0] - di/2; ic++) {
      le_displacements_[ic-1] = dy;
      le_duy_[ic-1] = duy;
      if ( (ic - di/2) % di == 0 ) {
	dy += le_displace_;
	duy += le_speed_;
      }
    }

    /* Last half block */
    for (ic = ntotal[0] - di/2 + 1; ic <= ntotal[0]; ic++) {
      le_displacements_[ic-1] = dy;
      le_duy_[ic-1] = duy;
    }
  }

  return;
}

/****************************************************************************
 *
 *  site_index
 *
 ****************************************************************************/

int site_index(int ic, int jc, int kc, const int n[3]) {

  int index;

  index = (n[1]*n[2]*(ic-1) + n[2]*(jc-1) + kc-1);

  return index;
}


/*****************************************************************************
 *
 *  le_unroll
 *
 *  Unroll the data in the presence of planes.
 *  This is always done relative to the middle of the system (as defined
 *  in le_displacements_[]).
 *
 *  If this is the velocity field, we need to make a correction to u_y
 *  to allow for the motion of the planes.
 *
 *****************************************************************************/

void le_unroll(double * data) {

  int ic, jc, kc, n;
  int j0, j1, j2, j3, jdy;
  double * buffer;
  double dy, fr;
  double du[3];

  /* Allocate the temporary buffer */

  buffer = (double *) malloc(nrec_*ntargets[1]*ntargets[2]*sizeof(double));
  if (buffer == NULL) {
    printf("malloc(buffer) failed\n");
    exit(-1);
  }

  du[0] = 0.0;
  du[1] = 0.0;
  du[2] = 0.0;

  for (ic = 1; ic <= ntargets[0]; ic++) {
    dy = le_displacements_[ic-1];
    jdy = floor(dy);
    fr = 1.0 - (dy - jdy);
    if (is_velocity_) du[1] = le_duy_[ic-1];

    for (jc = 1; jc <= ntargets[1]; jc++) {
      j0 = 1 + (jc - jdy - 3 + 1000*ntotal[1]) % ntotal[1];
      j1 = 1 + j0 % ntotal[1];
      j2 = 1 + j1 % ntotal[1];
      j3 = 1 + j2 % ntotal[1];

      for (kc = 1; kc <= ntargets[2]; kc++) {
	for (n = 0; n < nrec_; n++) {
	  buffer[nrec_*site_index(1,jc,kc,ntargets) + n] =
	    - (1.0/6.0)*fr*(fr-1.0)*(fr-2.0)
	               *data[nrec_*site_index(ic,j0,kc,ntargets) + n]
	    + 0.5*(fr*fr-1.0)*(fr-2.0)
	         *data[nrec_*site_index(ic,j1,kc,ntargets) + n]
	    - 0.5*fr*(fr+1.0)*(fr-2.0)
	         *data[nrec_*site_index(ic,j2,kc,ntargets) + n]
	    + (1.0/6.0)*fr*(fr*fr-1.0)
	               *data[nrec_*site_index(ic,j3,kc,ntargets) + n];
	}
      }
    }
    /* Put the whole buffer plane back in place */

    for (jc = 1; jc <= ntargets[1]; jc++) {
      for (kc = 1; kc <= ntargets[2]; kc++) {
	for (n = 0; n < nrec_; n++) {
	  data[nrec_*site_index(ic,jc,kc,ntargets) + n] =
	    buffer[nrec_*site_index(1,jc,kc,ntargets) + n];
	  if (n < 3) {
	    data[nrec_*site_index(ic,jc,kc,ntargets) + n] += du[n];
	  }
	}
      }
    }

  }

  free(buffer);

  return;
}

/****************************************************************************
 *
 *  reverse_byte_order_double
 *
 *  Reverse the bytes in the char argument to make a double.
 *
 *****************************************************************************/

double reverse_byte_order_double(char * c) {

  double result;
  char * p = (char *) &result;
  int b;

  for (b = 0; b < sizeof(double); b++) {
    p[b] = c[sizeof(double) - (b + 1)];
  }

  return result;
}

/****************************************************************************
 *
 *  electric_field
 *
 *  This routine calculates the electric field strength from the
 *  electrostatic potential. Currently periodic boundary conditions
 *  are assumed.
 *
 *****************************************************************************/

void electric_field(double *datasection, double *ppsection){

  int ic, jc, kc, ntot;
  int a;
  int index, indexpp; 
  int ixup, iyup, izup, ixdwn, iydwn, izdwn;
  int xs, ys, zs; /* strides */
  double e[3];

  xs = nrec_*ntargets[1]*ntargets[2];
  ys = nrec_*ntargets[2];
  zs = nrec_;

  ntot = nrec_*ntargets[0]*ntargets[1]*ntargets[2];

  for (ic = 1; ic <= ntargets[0]; ic++) {
    for (jc = 1; jc <= ntargets[1]; jc++) {
      for (kc = 1; kc <= ntargets[2]; kc++) {

	index   = nrec_*site_index(ic, jc, kc, ntargets);
	indexpp = npp_*site_index(ic, jc, kc, ntargets);

	ixup = index + xs;
	iyup = index + ys;
	izup = index + zs;

	ixdwn = index - xs;
	iydwn = index - ys;
	izdwn = index - zs;

	for (a = 0; a < 3; a++) e[a] = 0.0;
 
	if (ic == 1) {
	  ixdwn = nrec_*site_index(ntargets[0], jc, kc, ntargets);
	  e[0] = 0.5*(e0[0]*ntargets[0]);
	}
	if (jc == 1) {
	  iydwn = nrec_*site_index(ic, ntargets[1], kc, ntargets);
	  e[1] = 0.5*(e0[1]*ntargets[1]);
	}
	if (kc == 1) {
	  izdwn = nrec_*site_index(ic, jc, ntargets[2], ntargets);
	  e[2] = 0.5*(e0[2]*ntargets[2]);
	}

	if (ic == ntargets[0]) {
	  ixup = nrec_*site_index(1, jc, kc, ntargets);
	  e[0] = 0.5*(e0[0]*ntargets[0]);
	}
	if (jc == ntargets[1]) {
	  iyup = nrec_*site_index(ic, 1, kc, ntargets);
	  e[1] = 0.5*(e0[1]*ntargets[1]);
	}
	if (kc == ntargets[2]) {
	  izup = nrec_*site_index(ic, jc, 1, ntargets);
	  e[2] = 0.5*(e0[2]*ntargets[2]);
	}

	e[0] -= 0.5*( *(datasection + ixup) - *(datasection + ixdwn));
	e[1] -= 0.5*( *(datasection + iyup) - *(datasection + iydwn));
	e[2] -= 0.5*( *(datasection + izup) - *(datasection + izdwn));

	for (a = 0; a < 3; a++) {
	  *(ppsection + indexpp + a) = e[a];
	}
      }
    }
  }

  return;
}

/****************************************************************************
 *
 *  Order parameters
 *
 *  This routine diagonalises the Q-tensor and returns scalar, biaxial 
 *  order parameters and director field.
 *
 *  Note: The data from the diagonalisation routine is copied in place 
 *        into datasection as both the Q-tensor and scalar, biaxial order 
 *        parameter and director field have each five relevant components.  
 *
 *****************************************************************************/

void order_parameters(double *datasection){

int ic, jc, kc, nr, index;
double q[5], qs[5];

    for (ic = 1; ic <= ntargets[0]; ic++) {
      for (jc = 1; jc <= ntargets[1]; jc++) {
        for (kc = 1; kc <= ntargets[2]; kc++) {
          index = nrec_*site_index(ic, jc, kc, ntargets);
          for (nr = 0; nr < nrec_; nr++) {
            q[nr] = *(datasection + index + nr);
           }
	  calculate_scalar_biaxial_order_parameter_director(q, qs);
          for (nr = 0; nr < nrec_; nr++) {
            *(datasection + index + nr) = qs[nr];

      }
        }
      }
    }

return;
}


/*****************************************************************************
 *
 *  calculate_scalar_biaxial_order_parameter_director
 *
 *  Return the value of the scalar and biaxial order parameter and director for
 *  given Q tensor.
 *
 *  The biaxial OP is defined by B = sqrt(1 - 6*(tr(Q^3))^2 / (tr(Q^2))^3).
 *  As Q is traceless and symmetric we get the following dependencies:
 *
 *              Q = ((s,0,0),(0,t,0)(0,0,-s-t))
 *    (tr(Q^3))^2 = 9*s^2*t^2*(s^2 + 2*s*t + t^2)
 *    (tr(Q^2))^3 = 8*(s^6 + 6*s^4t^2 + 6*s^2t^4 + t^6 + 3*s^5t + 3*st^5 + 7*s^3t^3)
 *
 *****************************************************************************/

void calculate_scalar_biaxial_order_parameter_director(double q[5], double qs[5]) {

  int ifail;
  double eigenvalue[3];
  double eigenvector[3][3];
  double matrix_q[3][3];
  double s, t, Q3_2, Q2_3;

  matrix_q[0][0] = q[0];
  matrix_q[0][1] = q[1];
  matrix_q[0][2] = q[2];
  matrix_q[1][1] = q[3];
  matrix_q[1][2] = q[4];

  matrix_q[1][0] = q[1];
  matrix_q[2][0] = q[2];
  matrix_q[2][1] = q[4];
  matrix_q[2][2] = -q[0]-q[3];

  ifail = util_jacobi_sort(matrix_q, eigenvalue, eigenvector);

  if (ifail != 0) {
    qs[0] = 0.0;
    qs[1] = 0.0;
    qs[2] = 0.0;
    qs[3] = 0.0;
    qs[4] = 0.0;
  }
  else {
    qs[0] = eigenvalue[0];
    qs[1] = eigenvector[0][0];
    qs[2] = eigenvector[1][0];
    qs[3] = eigenvector[2][0];

    s = eigenvalue[0];
    t = eigenvalue[1];
    Q3_2 = 9.0*s*s*t*t*(s*s + 2.0*s*t + t*t);
    Q2_3 = 8.0*(s*s*s*s*s*s + 6.0*s*s*s*s*t*t
            + 6.0*s*s*t*t*t*t + t*t*t*t*t*t
            + 3.0*s*s*s*s*s*t + 3.0*s*t*t*t*t*t
            + 7.0*s*s*s*t*t*t);

    qs[4] = sqrt(fabs(1.0-6.0*Q3_2/Q2_3));
  }

  return;
}



/*****************************************************************************
 *
 *  util_jacobi_sort
 *
 *  Returns sorted eigenvalues and eigenvectors, highest eigenvalue first.
 *
 *  Returns zero on success.
 *
 *****************************************************************************/

int util_jacobi_sort(double a[3][3], double vals[3], double vecs[3][3]) {

  int ifail;

  ifail = util_jacobi(a, vals, vecs);

  /* And sort */

  if (vals[0] < vals[1]) util_swap(0, 1, vals, vecs);
  if (vals[0] < vals[2]) util_swap(0, 2, vals, vecs);
  if (vals[1] < vals[2]) util_swap(1, 2, vals, vecs);

  return ifail;
}


/*****************************************************************************
 *
 *  util_jacobi
 *
 *  Find the eigenvalues and eigenvectors of a 3x3 symmetric matrix a.
 *  This routine from Press et al. (page 467). The eigenvectors are
 *  returned as the columns of vecs[nrow][ncol].
 *
 *  Returns 0 on success. Garbage out usually means garbage in!
 *
 *****************************************************************************/

int util_jacobi(double a[3][3], double vals[3], double vecs[3][3]) {

  int iterate, ia, ib, ic;
  double tresh, theta, tau, t, sum, s, h, g, c;
  double b[3], z[3];

  const double d_[3][3]    = {{1.0, 0.0, 0.0}, {0.0, 1.0, 0.0}, {0.0, 0.0, 1.0}};
  const int maxjacobi = 200;    /* Maximum number of iterations */

  for (ia = 0; ia < 3; ia++) {
    for (ib = 0; ib < 3; ib++) {
      vecs[ia][ib] = d_[ia][ib];
    }
    vals[ia] = a[ia][ia];
    b[ia] = a[ia][ia];
    z[ia] = 0.0;
  }

  for (iterate = 1; iterate <= maxjacobi; iterate++) {
    sum = 0.0;

    for (ia = 0; ia < 2; ia++) {
      for (ib = ia + 1; ib < 3; ib++) {
        sum += fabs(a[ia][ib]);
      }
    }

    if (sum < DBL_MIN) return 0;

    if (iterate < 4)
      tresh = 0.2*sum/(3*3);
    else
      tresh = 0.0;

    for (ia = 0; ia < 2; ia++) {
      for (ib = ia + 1; ib < 3; ib++) {

        g = 100.0*fabs(a[ia][ib]);

        if (iterate > 4 && (fabs(vals[ia]) + g) == fabs(vals[ia]) &&
            (fabs(vals[ib]) + g) == fabs(vals[ib])) {
          a[ia][ib] = 0.0;
        }
        else if (fabs(a[ia][ib]) > tresh) {
          h = vals[ib] - vals[ia];
          if ((fabs(h) + g) == fabs(h)) {
            t = (a[ia][ib])/h;
          }
          else {
            theta = 0.5*h/a[ia][ib];
            t = 1.0/(fabs(theta) + sqrt(1.0 + theta*theta));
            if (theta < 0.0) t = -t;
          }

          c = 1.0/sqrt(1 + t*t);
          s = t*c;
          tau = s/(1.0 + c);
          h = t*a[ia][ib];
          z[ia] -= h;
          z[ib] += h;
          vals[ia] -= h;
          vals[ib] += h;
          a[ia][ib] = 0.0;

          for (ic = 0; ic <= ia - 1; ic++) {
            assert(ic < 3);
            g = a[ic][ia];
            h = a[ic][ib];
            a[ic][ia] = g - s*(h + g*tau);
            a[ic][ib] = h + s*(g - h*tau);
          }
          for (ic = ia + 1; ic <= ib - 1; ic++) {
            assert(ic < 3);
            g = a[ia][ic];
            h = a[ic][ib];
            a[ia][ic] = g - s*(h + g*tau);
            a[ic][ib] = h + s*(g - h*tau);
          }
          for (ic = ib + 1; ic < 3; ic++) {
            g = a[ia][ic];
            h = a[ib][ic];
            a[ia][ic] = g - s*(h + g*tau);
            a[ib][ic] = h + s*(g - h*tau);
          }
          for (ic = 0; ic < 3; ic++) {
            g = vecs[ic][ia];
            h = vecs[ic][ib];
            vecs[ic][ia] = g - s*(h + g*tau);
            vecs[ic][ib] = h + s*(g - h*tau);
          }
        }
      }
    }

    for (ia = 0; ia < 3; ia++) {
      b[ia] += z[ia];
      vals[ia] = b[ia];
      z[ia] = 0.0;
    }
  }

  return -1;
}



/*****************************************************************************
 *
 *  util_swap
 *
 *  Intended for a[3] eigenvalues and b[nrow][ncol] column eigenvectors.
 *
 *****************************************************************************/

void util_swap(int ia, int ib, double a[3], double b[3][3]) {

  int ic;
  double tmp;

  tmp = a[ia];
  a[ia] = a[ib];
  a[ib] = tmp;

  for (ic = 0; ic < 3; ic++) {
    tmp = b[ic][ia];
    b[ic][ia] = b[ic][ib];
    b[ic][ib] = tmp;
  }

  return;
}

/***************************************************************************
 *
 *  is_bigendian
 *
 *  Byte order for this 4-byte int is 00 00 00 01 for big endian (most
 *  significant byte stored first).
 *
 ***************************************************************************/

int is_bigendian() {

  const int i = 1;

  return (*(char *) &i == 0);
}


