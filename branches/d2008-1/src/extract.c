/****************************************************************************
 *
 *  i/o extraction
 *
 ****************************************************************************/


#include <stdio.h>
#include <stdlib.h>

int nlocal[3];
int rank, cx, cy, cz, nlx, nly, nlz, nox, noy, noz;

int ntotal[3]  = {512, 1024, 512};
int ntargets[3] = {512, 1024, 512}; /* Output target */

/* Lees Edwards */
int      nplanes_ = 32;
double   le_speed_ = 0.008;
double   le_displace_ = 0;
double * le_displacements_;

int site_index(int,int,int, const int *);
void read_data(FILE *, int *, double *);
void write_data(FILE *, int *, double *);
int copy_data(double *, double *);
void set_displacements(void);
void le_unroll(double *);

int main(int argc, char ** argv) {

    int pe[3]      = {16, 16, 16};
    int nio        = 512;
    int ntime;

    int i, n, p, pe_per_io;


    double * datalocal;
    double * datasection;
    char io_metadata[FILENAME_MAX];
    char io_data[FILENAME_MAX];
    char line[FILENAME_MAX];

    FILE * fp_metadata;
    FILE * fp_data;

    /* Get time from the command line */

    if (argc != 2) {
	printf("Use %s time step\n", argv[0]);
	exit(-1);
    }

    ntime = atoi(argv[1]);
    le_displace_ = ntime*le_speed_;
    printf("Time: %d displacement: %f\n", ntime, le_displace_);

    for (i = 0; i < 3; i++) {
	nlocal[i] = ntotal[i]/pe[i];
    }
    /* No processors each io group */
    pe_per_io = pe[0]*pe[1]*pe[2]/nio;

    /* No sites per processor */
    n = nlocal[0]*nlocal[1]*nlocal[2];
    datalocal = (double *) malloc(n*sizeof(double));
    if (datalocal == NULL) printf("Not allocated datalocal\n");
    
    /* No. sites in the section */ 
    n = ntargets[0]*ntargets[1]*ntargets[2];
    datasection = (double *) calloc(n, sizeof(double));
    if (datasection == NULL) printf("malloc(datasection) failed\n");

    /* LE displacements */
    le_displacements_ = (double *) malloc(ntotal[0]*sizeof(double));
    if (le_displacements_ == NULL) printf("malloc(le_displacements_)\n");
    set_displacements();

    /* Control */
    for (n = 1; n <= nio; n++) {

	/* open metadata file */

        sprintf(io_metadata, "%s.%d-%d", "io-metadata", nio, n);
        printf("Reading metadata file ... %s ", io_metadata);

	fp_metadata = fopen(io_metadata, "r");
	if (fp_metadata == NULL) printf("fopen(%s) failed\n", io_metadata);

#ifdef _NEW_
	sprintf(io_metadata, "phi.%3.3d-%3.3d.meta", nio, n);
	printf("Reading metadata file ... %s ", io_metadata);

	fp_metadata = fopen(io_metadata, "r");
	if (fp_metadata == NULL) printf("fopen(%s) failed\n", io_metadata);

	for (p = 0; p < 9; p++) {
	  fgets(line, FILENAME_MAX, fp_metadata);
	  printf("%s", line);
	}
#endif
	/* open data file */

        sprintf(io_data, "phi-%6.6d.%d-%d", ntime, nio, n);
        printf("-> %s\n", io_data);

#ifdef _NEW_
	sprintf(io_data, "phi-%6.6d.%3.3d-%3.3d", ntime, nio, n);
	printf("-> %s\n", io_data);
#endif
	fp_data = fopen(io_data, "r+b");
	if (fp_data == NULL) printf("fopen(%s) failed\n", io_data);


	for (p = 0; p < pe_per_io; p++) {

	  /* read metadata for this pe */

	  fscanf(fp_metadata, "%d %d %d %d %d %d %d %d %d %d",
		 &rank, &cx, &cy, &cz, &nlx, &nly, &nlz, &nox, &noy, &noz);

	  printf("%d %d %d %d %d %d %d %d %d %d\n",
		 rank, cx, cy, cz, nlx, nly, nlz, nox, noy, noz);

	  /* read data for this pe */
	  read_data(fp_data, nlocal, datalocal);

	  /* keep or discard */
	  copy_data(datalocal, datasection);
	}

	fclose(fp_data);
	fclose(fp_metadata);
    }

    le_unroll(datasection);

    /* Write a single file with the final section */

    sprintf(io_data, "phi-%6.6d", ntime);
    fp_data = fopen(io_data, "w+b");
    if (fp_data == NULL) printf("fopen(%s) failed\n", io_data);

    printf("\nWriting result to %s\n", io_data);

    write_data(fp_data, ntargets, datasection);

    fclose(fp_data);
    free(datalocal);
    free(le_displacements_);

    return 0;
}

/****************************************************************************
 *
 *  copy_data
 *
 *  Fill the global section with data from the current pe.
 *
 ****************************************************************************/

int copy_data(double * datalocal, double * datasection) {

    int ic, jc, kc, ic_g, jc_g, kc_g;
    int index_l, index_s;

    /* Sweep through the local sites for this pe */

    for (ic = 1; ic <= nlocal[0]; ic++) {
	ic_g = nox + ic;
	for (jc = 1; jc <= nlocal[1]; jc++) {
	    jc_g = noy + jc;
	    for (kc = 1; kc <= nlocal[2]; kc++) {
		kc_g = noz + kc;
		index_s = site_index(ic_g, jc_g, kc_g, ntotal);

		index_l = site_index(ic, jc, kc, nlocal);
		*(datasection + index_s) = *(datalocal + index_l);
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

    int ic, jc, kc, local, index;
    float phi_f;
    double phi;

    for (ic = 1; ic <= n[0]; ic++) {
	for (jc = 1; jc <= n[1]; jc++) {
	    for (kc = 1; kc <= n[2]; kc++) {
		index = site_index(ic, jc, kc, nlocal);

		/* index, double (old format)
		fread(&local, sizeof(int), 1, fp_data);
		fread(data + index, sizeof(double), 1, fp_data); */

		fread(&phi_f, sizeof(float), 1, fp_data);
		*(data + index) = (double) phi_f;
	    }
	}
    }

    return;
}

/****************************************************************************
 *
 *  write_data
 *
 *  Write contiguous block of data.
 *
 ****************************************************************************/

void write_data(FILE * fp_data, int n[3], double * data) {

    int ic, jc, kc, index = 0;
    float phi_f;

    for (ic = 0; ic < n[0]; ic++) {
	for (jc = 0; jc < n[1]; jc++) {
	    for (kc = 0; kc < n[2]; kc++) {
		phi_f = (float) data[index];
		fwrite(&phi_f, sizeof(float), 1, fp_data);
		index++;
	    }
	}
    }

    return;
}

/****************************************************************************
 *
 *  set_displacements
 *
 ****************************************************************************/

void set_displacements() {

    int ic;
    int di;
    double dy;

    di = ntotal[0] / nplanes_;
    dy = -(nplanes_/2.0)*le_displace_;

    /* Fist half block */
    for (ic = 1; ic <= di/2; ic++) {
	le_displacements_[ic-1] = dy;
    }

    dy += le_displace_;
    for (ic = di/2 + 1; ic <= ntotal[0] - di/2; ic++) {
	le_displacements_[ic-1] = dy;
	if ( (ic - di/2) % di == 0 ) dy += le_displace_;
    }

    /* Last half block */
    for (ic = ntotal[0] - di/2 + 1; ic <= ntotal[0]; ic++) {
	le_displacements_[ic-1] = dy;
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
 *****************************************************************************/

void le_unroll(double * data) {

    int ic, jc, kc;
    int j1, j2, jdy;
    double * buffer;
    double dy, fr;
    
    buffer = (double *) malloc(ntotal[1]*ntotal[2]*sizeof(double));
    if (buffer == NULL) {
	printf("malloc(buffer) failed\n");
	exit(-1);
    }

    /* Allocate the temporary buffer */

    for (ic = 1; ic <= ntotal[0]; ic++) {
	dy = le_displacements_[ic-1];
	jdy = floor(dy);
	fr = dy - jdy;

	for (jc = 1; jc <= ntotal[1]; jc++) {
	    j1 = 1 + (jc - jdy - 2 + 100*ntotal[1]) % ntotal[1];
	    j2 = 1 + j1 % ntotal[1];

	    for (kc = 1; kc <= ntotal[2]; kc++) {
		buffer[site_index(1,jc,kc,ntotal)] = 
		    fr*data[site_index(ic,j1,kc,ntotal)] +
		    (1.0 - fr)*data[site_index(ic,j2,kc,ntotal)];
	    }
	}
	/* Put the whole buffer plane back in place */

	for (jc = 1; jc <= ntotal[1]; jc++) {
	    for (kc = 1; kc <= ntotal[2]; kc++) {
		data[site_index(ic,jc,kc,ntotal)] =
		    buffer[site_index(1,jc,kc,ntotal)];
	    }
	}

    }

    free(buffer);

    return;
}
