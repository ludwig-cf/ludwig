/* Program to read a config file from the hydrid code */
/* We reverse the byte order and write everything out again. */
/* Run as ./a.out <old file name> <new file name> */ 

#include <stdio.h>
#include <stdlib.h>

/* Here is the total system size */

#define NX 256
#define NY 256
#define NZ 256

/* The total number of MPI tasks is nproc = NXPROC*NYPROC*NZPROC */

#define NXPROC 8
#define NYPROC 16
#define NZPROC 16

/* The number of IO groups is a simple decomposition of rank/nio_zize
 * with rank taken from the cartesian communicator and
 * nio_size = nproc / nio */ 

#define NIO 8


void read_write_per_processor(FILE * fpold, FILE * fpnew, int n);
double reverse_byte_order_double(char * c);

int main (int argc, char ** argv) {

    int nproc;
    int n, nio_size;

    char filename[FILENAME_MAX];

    FILE * fpold;
    FILE * fpnew;

    nproc = NXPROC*NYPROC*NZPROC;
    nio_size = nproc / NIO;

    printf("Hello\n");
    printf("We have system: %4d %4d %4d\n", NX, NY, NZ);
    printf("        decomp: %4d %4d %4d\n", NXPROC, NYPROC, NZPROC);
    printf("Total processor %4d\n", nproc);

    if (argc != 3) {
	printf("Usage %s <old file> <new file>\n", argv[0]);
	exit(0);
    }

    strcpy(filename, argv[1]);
    fpold = fopen(filename, "r");
    if (fpold == NULL) exit(0);

    strcpy(filename, argv[2]);
    fpnew = fopen(filename, "w");
    if (fpnew == NULL) exit(0);

    for (n = 0; n < nio_size; n++) {
	printf("Read/write for rank %4d\n", n);
	read_write_per_processor(fpold, fpnew, n);
    }

    fclose(fpnew);
    fclose(fpold);

    return 0;
}


/****************************************************************************
 *
 *  read_write_per_processor
 *
 ****************************************************************************/

void read_write_per_processor(FILE * fpold, FILE * fpnew, int rank) {

    int nlocal[3];
    int ic, jc, kc, n;

    char buffer[8];
    double datum;

    nlocal[0] = NX/NXPROC;
    nlocal[1] = NY/NYPROC;
    nlocal[2] = NZ/NZPROC;

    /* State nlocal[X]*nlocal[Y]*nlocal[Z]*20 doubles */

    for (ic = 1; ic <= nlocal[0]; ic++) {
	for (jc = 1; jc <= nlocal[1]; jc++) {
	    for (kc = 1; kc <= nlocal[2]; kc++) {
		for (n = 1; n <= 20; n++) {
		    fread(buffer, sizeof(char), sizeof(double), fpold);
		    datum = reverse_byte_order_double(buffer);
		    fwrite(&datum, sizeof(double), 1, fpnew);
		}
	    }
	}
    }

    /* Red shifts (two doubles) */

    fread(buffer, sizeof(char), sizeof(double), fpold);
    datum = reverse_byte_order_double(buffer);
    fwrite(&datum, sizeof(double), 1, fpnew);

    fread(buffer, sizeof(char), sizeof(double), fpold);
    datum = reverse_byte_order_double(buffer);
    fwrite(&datum, sizeof(double), 1, fpnew);

    /* Shear stuff  nlocal[X]*ntotal[Y]*nlocal[Z]*5 doubles */

    for (ic = 1; ic <= nlocal[0]; ic++) {
	for (jc = 1; jc <= NY; jc++) {
	    for (kc = 1; kc <= nlocal[2]; kc++) {
		for (n = 1; n <= 5; n++) {
		    fread(buffer, sizeof(char), sizeof(double), fpold);
		    datum = reverse_byte_order_double(buffer);
		    fwrite(&datum, sizeof(double), 1, fpnew);
		}
	    }
	}
    }


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

