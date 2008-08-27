/****************************************************************************
 *
 *  i/o extraction
 *
 ****************************************************************************/


#include <stdio.h>
#include <stdlib.h>

int nlocal[3];
int rank, cx, cy, cz, nlx, nly, nlz, nox, noy, noz;

int ntotal[3]  = {512, 1024, 1};
int ntargets[3] = {512, 1024, 1}; /* Output target */
int nplanes_ = 32;
double le_speed_ = 0.008;
int le_displace_ = 0;
int * le_displacements_;

int site_index_local(int,int,int);
int index_section(int, int, int);
void read_data(FILE *, int *, double *);
void write_data(FILE *, int *, double *);
int copy_data(double *, double *);
int le_displacement(int, int);
void set_displacements(void);

int main(int argc, char ** argv) {

    int pe[3]      = {2, 4, 1};
    int nio        = 1;
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
    printf("Time: %d displacement: %d\n", ntime, le_displace_);


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
    le_displacements_ = (int *) malloc(ntotal[0]*sizeof(int));
    if (le_displacements_ == NULL) printf("malloc(le_displacements_)\n");
    set_displacements();

    /* Control */
    for (n = 1; n <= nio; n++) {

	/* open metadata file */
	sprintf(io_metadata, "phi.%3.3d-%3.3d.meta", nio, n);
	printf("Reading metadata file ... %s ", io_metadata);

	fp_metadata = fopen(io_metadata, "r");
	if (fp_metadata == NULL) printf("fopen(%s) failed\n", io_metadata);

	for (p = 0; p < 9; p++) {
	  fgets(line, FILENAME_MAX, fp_metadata);
	  printf("%s", line);
	}

	/* open data file */
	sprintf(io_data, "phi-%6.6d.%3.3d-%3.3d", ntime, nio, n);
	printf("-> %s\n", io_data);

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
 *  Fill the required section with data from the current pe, if
 *  they intersect.
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
		index_s = index_section(ic_g, jc_g, kc_g);

		if (index_s >= 0) {
		    index_l = site_index_local(ic, jc, kc);
		    *(datasection + index_s) = *(datalocal + index_l);
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

    int ic, jc, kc, local, index;
    double phi;

    for (ic = 1; ic <= n[0]; ic++) {
	for (jc = 1; jc <= n[1]; jc++) {
	    for (kc = 1; kc <= n[2]; kc++) {
		index = site_index_local(ic, jc, kc);

		/* index, double (old format)
		fread(&local, sizeof(int), 1, fp_data);
		fread(data + index, sizeof(double), 1, fp_data); */

		fread(&phi, sizeof(double), 1, fp_data);
		*(data + index) = phi;
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

    int ic, jc, kc, index = 0;

    for (ic = 0; ic < n[0]; ic++) {
	for (jc = 0; jc < n[1]; jc++) {
	    for (kc = 0; kc < n[2]; kc++) {
		fwrite(data + index, sizeof(double), 1, fp_data);
		index++;
	    }
	}
    }

    return;
}

/****************************************************************************
 *
 *  index_section
 *
 *  For given global lattice index (ic, jc, kc), return the index
 *  in the final section. -1 is returned if it's not in the section.
 *
 *  When LE planes are present, global index jc is adjusted as a
 *  function of ic to take account of the planes. We assume an
 *  integer plane displacement.
 *
 ****************************************************************************/

int index_section(int ic, int jc, int kc) {

    int index = -1;
    int djc;

    /* assume section origin is (1,1,1) */
    /* (ic,jc,kc) must be in doamin, so */

    if (nplanes_ > 0) {
	djc = le_displacement(ic, jc);
    }

    if (ic <= ntargets[0] && djc <= ntargets[1] && kc <= ntargets[2]) {
	/* work out index */
	index = ntargets[1]*ntargets[2]*(ic-1) + ntargets[2]*(djc-1) + kc-1;
    }

    return index;
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
 *  set_displacements
 *
 ****************************************************************************/

void set_displacements() {

    int ic, dy;
    int di;

    di = ntotal[0] / nplanes_;
    dy = -(nplanes_/2)*le_displace_;

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

int site_index_local(int ic, int jc, int kc) {

    int index;

    index = (nlocal[1]*nlocal[2]*(ic-1) + nlocal[2]*(jc-1) + kc-1);

    return index;
}
