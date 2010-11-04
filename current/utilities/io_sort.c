/****************************************************************************
 *
 *  io_sort.c
 *
 *  Invoke with three arguments, e.g.,
 *
 *  ./a.out type_id nio_groups file_stub
 *
 *  To sort parallel I/O into a single ordered file for analysis.
 *
 *  Type 1 Disclination locations
 *         3*int (being ic jc kc)
 *
 *  Type 2 Order parameter etc on lattice
 *         3*int + 10*double
 *
 *  Type 3 Stress on lattice
 *         3*int + 9*double (being ic jc kc Sxx Sxy ... Szz)

 *  Type 4 Director on lattice
 *         3*int + 3*double (being ic jc kc dx dy dz)
 *
 *  Expected file names e.g.,: file_stub-n-nio
 *         n = current io group
 *         nio = total number of io groups (or, number of files per step)
 *
 *  The output will appear in a new file 'file_stub'
 *
 ****************************************************************************/

#include <stdio.h>
#include <stdlib.h>

/* System size (could be read at run time) */

int lx_ = 128;
int ly_ = 128;
int lz_ = 128;

float * data_out_;
int nio_;


int get_global_index(int, int, int);
void read_disclination(const char *);
void write_disclination(const char *);
void read_order_velo(const char *);
void write_order_velo(const char *);
void read_stress(const char *);
void write_stress(const char *);
void read_director(const char *);
void write_director(const char *);

int main (int argc, char ** argv) {

  int n;
  int nbuffer;
  int id_type;
  char file_name[FILENAME_MAX];
  char file_stub[FILENAME_MAX];

  void (* read_function)(const char *);
  void (* write_function)(const char *);

  /* Read command line */

  if (argc != 4) {
    printf("check command line arguments\n");
    exit(-1);
  }

  id_type = atoi(argv[1]);
  nio_     = atoi(argv[2]);
  sprintf(file_stub, argv[3]);

  printf("Input type is %d\n", id_type);
  printf("Input number I/O groups is %d\n", nio_);
  printf("Input file stub is %s\n", file_stub);

  /* nio must divide lx_ here, i.e., each parallel file contains
   * 1/nio of the total data (disclinations excluded) */
  if (lx_ % nio_) {
      printf("lx_ % nio is not zero\n");
      exit(-1);
  }

  if (id_type == 1) {
    printf("disclination file\n");
    nbuffer = 1;
    read_function = read_disclination;
    write_function = write_disclination;
  }

  if (id_type == 2) {
    printf("order_velo file\n");
    nbuffer = 10;
    read_function = read_order_velo;
    write_function = write_order_velo;
  }

  if (id_type == 3) {
    printf("stress file\n");
    nbuffer = 9;
    read_function = read_stress;
    write_function = write_stress;
  }

  if (id_type == 4) {
    printf("director file\n");
    nbuffer = 9;
    read_function = read_director;
    write_function = write_director;
  }

  if (id_type < 1 || id_type > 4) {
    printf("Unknown type\n");
    exit(-1);
  }

  /* Allocate memory */

  data_out_ = (float *) calloc(nbuffer*lx_*ly_*lz_, sizeof(float));
  if (data_out_ == NULL) {
    printf("calloc(data_out_) failed\n");
    exit(-1);
  }


  /* Read parallel output files in turn */

  for (n = 0; n < nio_; n++) {
    sprintf(file_name, "%s-%d-%d", file_stub, n, nio_);
    printf("Reading from %s\n", file_name);
    read_function(file_name);
  }

  printf("Writing combined data to %s\n", file_stub);
  write_function(file_stub);

  free(data_out_);

  return 0;
}

/****************************************************************************
 *
 *  read_disclination
 *
 *  We read the discliniation locations until EOF is reached.
 *  Disclinations are stored in data_out_ as
 *     0 = no disclination
 *     1 = disclination
 *
 ****************************************************************************/

void read_disclination(const char * file_name) {

  int ibuf[3];
  int index, iread;

  FILE * fp;

  fp = fopen(file_name, "r");
  if (fp == NULL) printf("fopen(%s) failed\n");

  while (!feof(fp)) {
    /* Check the number of data items read in case the file is empty... */
    iread = fread(ibuf, sizeof(int), 3, fp);
    if (iread == 3) {
      index = get_global_index(ibuf[0], ibuf[1], ibuf[2]);
      data_out_[index] = 1.0;
    }
  }

  fclose(fp);

  return;
}

/****************************************************************************
 *
 *  write_disclination
 *
 *  Write the entire dislocation set in order.
 *
 ****************************************************************************/

void write_disclination(const char * file_name) {

  int ic, jc, kc, index;
  FILE * fp;

  fp = fopen(file_name, "w");
  if (fp == NULL) printf("fopen(disclination file) failed\n");

  for (ic = 0; ic < lx_; ic++) {
    for (jc = 0; jc < ly_; jc++) {
      for (kc = 0; kc < lz_; kc++) {

	index = get_global_index(ic, jc, kc);

	if (data_out_[index] > 0.5) {
	  fprintf(fp, "%d %d %d\n", ic, jc, kc);
	}

      }
    }
  }

  fclose(fp);

  return;
}

/****************************************************************************
 *
 *  read_order_velo
 *
 *  Read
 *  ic jc kc Qxx Qxy Qxz Qyy Qyz u_x u_y u_z eigenvalue molfieldxx
 *
 *  as 3*int 10*double
 *  and store the doubles in the data_out_ array as float.
 *
 ***************************************************************************/

void read_order_velo(const char * file_name) {

  int ic, jc, kc, p;
  int ibuf[3];
  int index;
  double rbuf[10];

  FILE * fp;

  fp = fopen(file_name, "r");
  if (fp == NULL) printf("fopen(%s) failed\n", file_name);


  for (ic = 0; ic < lx_/nio_; ic++) {
    for (jc = 0; jc < ly_; jc++) {
      for (kc = 0; kc < lz_; kc++) {

	fread(ibuf, sizeof(int), 3, fp);
	fread(rbuf, sizeof(double), 10, fp);

	index = 10*get_global_index(ibuf[0], ibuf[1], ibuf[2]);

	for (p = 0; p < 10; p++) {
	  data_out_[index+p] = (float) rbuf[p];
	}

      }
    }
  }

  fclose(fp);


  return;
}

/****************************************************************************
 *
 *  write_order_velo
 *
 *  Write the global output.
 *
 ****************************************************************************/

void write_order_velo(const char * file_name) {

  int ic, jc, kc, index;
  FILE * fp;

  fp = fopen(file_name, "w");
  if (fp == NULL) printf("fopen(order_velo file) failed\n");

  for (ic = 0; ic < lx_; ic++) {
    for (jc = 0; jc < ly_; jc++) {
      for (kc = 0; kc < lz_; kc++) {

	index = 10*get_global_index(ic, jc, kc);

	fprintf(fp, "%d %d %d %g %g %g %g %g %g %g %g %g %g\n",
		ic, jc, kc, data_out_[index], data_out_[index+1],
		data_out_[index+2], data_out_[index+3], data_out_[index+4],
		data_out_[index+5], data_out_[index+6], data_out_[index+7],
		data_out_[index+8], data_out_[index+9]);
      }
    }
  }

  fclose(fp);  

  return;
}


/****************************************************************************
 *
 *  read_stress
 *
 *  Read
 *  ic jc kc Sxx Sxy ... Szz
 *
 *  as 3*int 9*double
 *  and store the doubles in the data_out_ array as float.
 *
 ***************************************************************************/

void read_stress(const char * file_name) {

  int ic, jc, kc, p;
  int ibuf[3];
  int index;
  double rbuf[9];

  FILE * fp;

  fp = fopen(file_name, "r");
  if (fp == NULL) printf("fopen(%s) failed\n");

  for (ic = 0; ic < lx_/nio_; ic++) {
    for (jc = 0; jc < ly_; jc++) {
      for (kc = 0; kc < lz_; kc++) {

	fread(ibuf, sizeof(int), 3, fp);
	fread(rbuf, sizeof(double), 9, fp);
	index = 9*get_global_index(ibuf[0], ibuf[1], ibuf[2]);

	for (p = 0; p < 9; p++) {
	  data_out_[index+p] = (float) rbuf[p];
	}
      }
    }
  }

  fclose(fp);

  return;
}

/****************************************************************************
 *
 *  write_stress
 *
 *  Write the global output.
 *
 ****************************************************************************/

void write_stress(const char * file_name) {

  int ic, jc, kc, index;
  FILE * fp;

  fp = fopen(file_name, "w");
  if (fp == NULL) printf("fopen(stress file) failed\n");

  for (ic = 0; ic < lx_; ic++) {
    for (jc = 0; jc < ly_; jc++) {
      for (kc = 0; kc < lz_; kc++) {

	index = 9*get_global_index(ic, jc, kc);

	fprintf(fp, "%d %d %d %g %g %g %g %g %g %g %g %g %g\n",
		ic, jc, kc, data_out_[index], data_out_[index+1],
		data_out_[index+2], data_out_[index+3], data_out_[index+4],
		data_out_[index+5], data_out_[index+6], data_out_[index+7],
		data_out_[index+8]);
      }
    }
  }

  fclose(fp);  

  return;
}


/****************************************************************************
 *
 *  read_director
 *
 *  Read
 *  ic jc kc dx dy dz
 *
 *  as 3*int 3*double
 *  and store the doubles in the data_out_ array as float.
 *
 ***************************************************************************/

void read_director(const char * file_name) {

  int ic, jc, kc, p;
  int ibuf[3];
  int index;
  double rbuf[3];

  FILE * fp;

  fp = fopen(file_name, "r");
  if (fp == NULL) printf("fopen(%s) failed\n");

  for (ic = 0; ic < lx_/nio_; ic++) {
    for (jc = 0; jc < ly_; jc++) {
      for (kc = 0; kc < lz_; kc++) {

	fread(ibuf, sizeof(int), 3, fp);
	fread(rbuf, sizeof(double), 3, fp);
	index = 3*get_global_index(ibuf[0], ibuf[1], ibuf[2]);

	for (p = 0; p < 3; p++) {
	  data_out_[index+p] = (float) rbuf[p];
	}
      }
    }
  }

  fclose(fp);

  return;
}

/****************************************************************************
 *
 *  write_director
 *
 *  Write the global output.
 *
 ****************************************************************************/

void write_director(const char * file_name) {

  int ic, jc, kc, index;
  FILE * fp;

  fp = fopen(file_name, "w");
  if (fp == NULL) printf("fopen(stress file) failed\n");

  for (ic = 0; ic < lx_; ic++) {
    for (jc = 0; jc < ly_; jc++) {
      for (kc = 0; kc < lz_; kc++) {

	index = 3*get_global_index(ic, jc, kc);

	fprintf(fp, "%d %d %d %g %g %g\n", ic, jc, kc, data_out_[index],
		 data_out_[index+1], data_out_[index+2]);
      }
    }
  }

  fclose(fp);  

  return;
}
/****************************************************************************
 *
 *  get_global_index
 *
 ****************************************************************************/

int get_global_index(int i, int j, int k) {
  int index;

  index = ly_*lz_*i + lz_*j + k;

  return index;
}
