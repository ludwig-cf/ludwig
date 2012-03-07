/****************************************************************************
 *
 *  initialise.c
 *
 *  Toolkit to create parallel input files of the 
 *  tensor order parameter phi and LB-distributions in 
 *  row-major order.
 *
 *  The details of the I/0-topology have to be given below,
 *  whereas the individual configuration of order parameter 
 *  and the distribution are set are hard-coded in separtate 
 *  routines.
 * 
 *  ./initialise
 *
 *  WARNING: Backup of the input file is required as it
 *  might be replaced when the command is issued. 
 *
 *  Compile with $(CC) -o initialise initialise.c -lm
 *
 *  (c) 2012 Edinburgh Parallel Computing Centre &
 *           University College London
 *
 ****************************************************************************/
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#define Pi 3.141592653589793

/********************************************************
 All parameters below have to be given at compile time 
********************************************************/
int ntime_ = 50000; 			/* timestep */

int ntotal_in_[3] = {1, 32, 32};	/* total system size input file */
int pe_in_[3] = {1, 2, 2}; 		/* Cartesian decomposition input file */
int phi_in_io_[3] = {1, 1, 1}; 		/* I/O decomposition input file */

int ntotal_[3] = {128, 128, 32}; 		/* total system size */
int pe_[3] = {2, 2, 1}; 		/* Cartesian processor decomposition */
int phi_io_[3] = {1, 1, 1}; 		/* I/O decomposition for order parameter */

int dist_io_[3] = {1, 1, 1}; 		/* I/O decomposition for distributions */

double q0=2.0*Pi/32.0;			/* pitch wave vector */
double amp=0.03333333;			/* initial amplitude */
int NVEL_ = 19;				/* No. of lattice velocities */
int N_torus_ = 2;			/* No. of tori in initial configuration */
int file_input_ = 1; 			/* switch for file input */
int input_binary_ = 1;			/* switch for format of input */
int output_binary_ = 1;			/* switch for format of final output */
/*******************************************************/

int input_isbigendian_ = 0;		/* If we have to deal with endianness - currently not supported */
int nlocal[3];				/* size of local domain */
int nlocal_in[3];			/* size of local domain */

struct io_info {
  const char *stub;
  int nrec;
  int io_grid[3];
  int nio;
  int pe[3];
  int pe_per_io;
  int ntotal[3];
  int nlocal[3];
};

void set_phi_nematic(double, double, double, double ****);
void set_torus_radius_centre(double *, double **);
void set_phi_torus(double ****, double *, double **, double ****);
void set_phi_cf1(double ****, double ****);
void set_phi_cf2(double ****, double ****);
void set_dist(double ****);
void copy_global2local(struct io_info *, double ****, int *, double *);
void copy_local2global(struct io_info *, double *, int *, double ****);
void read_files(struct io_info *, double ****, double *);
void read_pe(struct io_info *, FILE *, double *);
void write_pe(struct io_info *, FILE *, double *);
void write_files(struct io_info *, double ****, double *);
int site_index(int, int, int, const int *);
void Mz(double ** , double **, double); 


int main(int argc, char ** argv) {

  double * datalocal;
  double **** phi;
  double **** phi_in;
  double **** dist;
  double * R, ** Z; /* radii and centres of tori */
  int i, j, k, n; 
  struct io_info  phi_info, phi_in_info, dist_info;
  int pe_per_io;

  R = (double *) calloc(N_torus_, sizeof(double));
  if (R == NULL) printf("calloc(R) failed\n");
  Z = (double **) calloc(N_torus_, sizeof(double));
  if (Z == NULL) printf("calloc(Z) failed\n");
  for (i = 0; i < N_torus_; i++){
    Z[i] = (double *) calloc(3, sizeof(double));
    if (Z[i] == NULL) printf("calloc(Z) failed\n");
  }

  ////////////////////////// 
  /* Create phi data file */
 
 /* Work out parallel decompsition that is used and the number of
     processors per I/O group and store in structure. */

  for (i = 0; i < 3; i++) {
    nlocal_in[i] = ntotal_in_[i]/pe_in_[i];
  }

  phi_in_info.stub = "phi"; 
  phi_in_info.nrec = 5; 

  for (i = 0; i < 3; i++) {
    phi_in_info.io_grid[i] = phi_in_io_[i];
    phi_in_info.pe[i] = pe_in_[i];
    phi_in_info.ntotal[i] = ntotal_in_[i];
    phi_in_info.nlocal[i] = nlocal_in[i];
  }

  phi_in_info.nio = phi_in_io_[0]*phi_in_io_[1]*phi_in_io_[2];
  phi_in_info.pe_per_io = pe_in_[0]*pe_in_[1]*pe_in_[2]/phi_in_info.nio;

  /* Allocate storage for input array */
  n = phi_in_info.nrec*nlocal_in[0]*nlocal_in[1]*nlocal_in[2];
  datalocal = (double *) calloc(n, sizeof(double));
  if (datalocal == NULL) printf("calloc(datalocal) failed\n");
    
  phi_in = (double ****) calloc(ntotal_in_[0]+2, sizeof(double));
  if (phi_in == NULL) printf("calloc(phi_in) failed\n");
  for (i = 0; i <= ntotal_in_[0]+1; i++){
    phi_in[i] = (double ***) calloc(ntotal_in_[1]+2, sizeof(double));
    if (phi_in[i] == NULL) printf("calloc(phi_in) failed\n");
    for (j = 0; j <= ntotal_in_[1]+1; j++){
      phi_in[i][j] = (double **) calloc(ntotal_in_[2]+2, sizeof(double));
      if (phi_in[i][j] == NULL) printf("calloc(phi_in) failed\n");
      for (k = 0; k <= ntotal_in_[2]+1; k++){
	phi_in[i][j][k] = (double *) calloc(phi_in_info.nrec, sizeof(double));
	if (phi_in[i][j][k] == NULL) printf("calloc(phi_in) failed\n");
      }
    }
  }

  /* Read order parameter input files */
  if (file_input_) read_files(&phi_in_info,phi_in,datalocal);

  free(datalocal);

  /* Work out parallel decompsition that is used and the number of
     processors per I/O group and store in structure. */

  for (i = 0; i < 3; i++) {
    nlocal[i] = ntotal_[i]/pe_[i];
  }

  phi_info.stub = "phi"; 
  phi_info.nrec = 5; 

  for (i = 0; i < 3; i++) {
    phi_info.io_grid[i] = phi_io_[i];
    phi_info.pe[i] = pe_[i];
    phi_info.ntotal[i] = ntotal_[i];
    phi_info.nlocal[i] = nlocal[i];
  }
  phi_info.nio = phi_io_[0]*phi_io_[1]*phi_io_[2];
  phi_info.pe_per_io = pe_[0]*pe_[1]*pe_[2]/phi_info.nio;

  /* Allocate storage for output array */
  n = phi_info.nrec*nlocal[0]*nlocal[1]*nlocal[2];
  datalocal = (double *) calloc(n, sizeof(double));
  if (datalocal == NULL) printf("calloc(datalocal) failed\n");
    
  phi = (double ****) calloc(ntotal_[0]+2, sizeof(double));
  if (phi == NULL) printf("calloc(phi) failed\n");
  for (i = 0; i <= ntotal_[0]+1; i++){
    phi[i] = (double ***) calloc(ntotal_[1]+2, sizeof(double));
    if (phi[i] == NULL) printf("calloc(phi) failed\n");
    for (j = 0; j <= ntotal_[1]+1; j++){
      phi[i][j] = (double **) calloc(ntotal_[2]+2, sizeof(double));
      if (phi[i][j] == NULL) printf("calloc(phi) failed\n");
      for (k = 0; k <= ntotal_[2]+1; k++){
	phi[i][j][k] = (double *) calloc(phi_info.nrec, sizeof(double));
	if (phi[i][j][k] == NULL) printf("calloc(phi) failed\n");
      }
    }
  }

  /*************************************/
  /* Include subroutines here that set */
  /*   and modify the order parameter  */ 
  /*************************************/

//  set_phi_cf1(phi_in,phi);
//  set_phi_cf2(phi_in,phi);

  set_phi_nematic(0,0,1,phi);
  set_torus_radius_centre(R,Z);
  set_phi_torus(phi_in,R,Z,phi);

  /* write order parameter output */
  write_files(&phi_info,phi,datalocal);

  free(datalocal);

  free(phi);
  free(phi_in);

  //////////////////////////////
  /* Create distribution file */

  dist_info.stub = "dist"; 
  dist_info.nrec = NVEL_; 

  for (i = 0; i < 3; i++) {
    dist_info.io_grid[i] = dist_io_[i];
    dist_info.pe[i] = pe_[i];
    dist_info.ntotal[i] = ntotal_[i];
    dist_info.nlocal[i] = nlocal[i];
  }
  dist_info.nio = dist_io_[0]*dist_io_[1]*dist_io_[2];
  dist_info.pe_per_io = pe_[0]*pe_[1]*pe_[2]/dist_info.nio;

  /* Allocate storage */
  n = dist_info.nrec*nlocal[0]*nlocal[1]*nlocal[2];
  datalocal = (double *) calloc(n, sizeof(double));
  if (datalocal == NULL) printf("calloc(datalocal) failed\n");

  dist = (double ****) calloc(ntotal_[0]+2, sizeof(double));
  if (dist == NULL) printf("calloc(dist) failed\n");

  for (i = 0; i <= ntotal_[0]+1; i++){
    dist[i] = (double ***) calloc(ntotal_[1]+2, sizeof(double));
    if (dist[i] == NULL) printf("calloc(dist) failed\n");

    for (j = 0; j <= ntotal_[1]+1; j++){
      dist[i][j] = (double **) calloc(ntotal_[2]+2, sizeof(double));
      if (dist[i][j] == NULL) printf("calloc(dist) failed\n");

      for (k = 0; k <= ntotal_[2]+1; k++){
	dist[i][j][k] = (double *) calloc(dist_info.nrec, sizeof(double));
	if (dist[i][j][k] == NULL) printf("calloc(dist) failed\n");
      }
    }
  }

  /* Set initial values for distribution data (currently u=0)*/ 
  set_dist(dist);
  write_files(&dist_info,dist,datalocal);

  free(datalocal);
  free(dist);
}

/* end main routine */

/****************************************************************************
 *
 *  read_files
 *
 *  Driver routine for reading input data files
 *
 ****************************************************************************/
void read_files(struct io_info * file_info, double **** data, double * datalocal){

  int i, j, k, ip, jp, kp, n, p;
  char io_data[FILENAME_MAX];
  char stub[FILENAME_MAX];
  FILE * fp_data;
  int noffset[3]; /* offset of local domain */

  /* Sweeping through i/o groups */
  n = 1;
  for (k = 0; k < file_info->io_grid[2]; k++) {
    for (j = 0; j < file_info->io_grid[1]; j++) {
      for (i = 0; i < file_info->io_grid[0]; i++) {

	sprintf(stub, "%s",file_info->stub);
	sprintf(io_data, "%s-%8.8d.%3.3d-%3.3d", file_info->stub, ntime_, file_info->nio, n);
	printf("Reading from <- %s\n", io_data);

	fp_data = fopen(io_data, "r");
	if (fp_data == NULL) {
	  printf("fopen(%s) failed - continue with writing files\n", io_data);
	  return;
	}

	/* i/o group x-offset */
	noffset[0] = i*file_info->pe[0]/file_info->io_grid[0]*file_info->nlocal[0];

	/* Sweeping through # of pe per i/o group */
	for (ip = 0; ip < file_info->pe[0]/file_info->io_grid[0]; ip++) {

	  /* i/o group y-offset */
	  noffset[1] = j*file_info->pe[1]/file_info->io_grid[1]*file_info->nlocal[1];

	  for (jp = 0; jp < file_info->pe[1]/file_info->io_grid[1]; jp++) {

	    /* i/o group z-offset */
	    noffset[2] = k*file_info->pe[2]/file_info->io_grid[2]*file_info->nlocal[2];

	    for (kp = 0; kp < file_info->pe[2]/file_info->io_grid[2]; kp++) {

	      /* Reading and relevant part for one pe and copying into global array */

	      read_pe(file_info, fp_data, datalocal);
	      copy_local2global(file_info, datalocal, noffset, data);

	      /* pe z-increment */
	      noffset[2] += file_info->nlocal[2];
	    }
	    /* pe y-increment */
	    noffset[1] += file_info->nlocal[1];
	  }
	  /* pe x-increment */
	  noffset[0] += file_info->nlocal[0];
	}

	fclose(fp_data);
	n++;

      }
    }
  }

  return;
}

/****************************************************************************
 *
 *  read_pe
 *
 *  Read data block relevant for one pe
 *
 ****************************************************************************/

void read_pe(struct io_info * file_info,  FILE * fp_data, double * data){

  int i, ic, jc, kc, index, nr;
  double phi;
  int nlocal[3];

  for(i = 0; i < 3; i++){
    nlocal[i] = file_info->nlocal[i];
  }

  if (input_binary_) {
    for (ic = 1; ic <= nlocal[0]; ic++) {
      for (jc = 1; jc <= nlocal[1]; jc++) {
        for (kc = 1; kc <= nlocal[2]; kc++) {
          index = site_index(ic, jc, kc, nlocal);

          for (nr = 0; nr < file_info->nrec; nr++) {
            fread(&phi, sizeof(double), 1, fp_data);
            *(data + file_info->nrec*index + nr) = phi;
          }
        }
      }
    }
  }
  else {
    for (ic = 1; ic <= nlocal[0]; ic++) {
      for (jc = 1; jc <= nlocal[1]; jc++) {
        for (kc = 1; kc <= nlocal[2]; kc++) {
          index = site_index(ic, jc, kc, nlocal);

          for (nr = 0; nr < file_info->nrec; nr++) {
            fscanf(fp_data, "%le", &phi);
	    *(data + file_info->nrec*index + nr) = phi;
          }
        }
      }
    }
  }

  return;
}

/****************************************************************************
 *
 *  write_files
 *
 *  Driver routine for writing output files
 *
 ****************************************************************************/

void write_files(struct io_info * file_info, double **** data, double * datalocal){

  int i, j, k, ip, jp, kp, n, p;
  char io_data[FILENAME_MAX];
  char stub[FILENAME_MAX];
  FILE * fp_data;
  int noffset[3]; /* offset of local domain */

  /* Sweeping through i/o groups */
  n = 1;
  for (k = 0; k < file_info->io_grid[2]; k++) {
    for (j = 0; j < file_info->io_grid[1]; j++) {
      for (i = 0; i < file_info->io_grid[0]; i++) {

	sprintf(stub, "%s",file_info->stub);
	sprintf(io_data, "%s-%8.8d.%3.3d-%3.3d", file_info->stub, ntime_, file_info->nio, n);
	printf("Writing to -> %s\n", io_data);

	fp_data = fopen(io_data, "w+b");
	if (fp_data == NULL) printf("fopen(%s) failed\n", io_data);

	/* i/o group x-offset */
	noffset[0] = i*file_info->pe[0]/file_info->io_grid[0]*file_info->nlocal[0];

	/* Sweeping through # of pe per i/o group */
	for (ip = 0; ip < file_info->pe[0]/file_info->io_grid[0]; ip++) {

	  /* i/o group y-offset */
	  noffset[1] = j*file_info->pe[1]/file_info->io_grid[1]*file_info->nlocal[1];

	  for (jp = 0; jp < file_info->pe[1]/file_info->io_grid[1]; jp++) {

	    /* i/o group z-offset */
	    noffset[2] = k*file_info->pe[2]/file_info->io_grid[2]*file_info->nlocal[2];

	    for (kp = 0; kp < file_info->pe[2]/file_info->io_grid[2]; kp++) {

	      /* Copying and writing relevant part for one pe */
	      copy_global2local(file_info, data, noffset, datalocal);
	      write_pe(file_info, fp_data, datalocal);

	      /* pe z-increment */
	      noffset[2] += file_info->nlocal[2];
	    }
	    /* pe y-increment */
	    noffset[1] += file_info->nlocal[1];
	  }
	  /* pe x-increment */
	  noffset[0] += file_info->nlocal[0];
	}

	fclose(fp_data);
	n++;

      }
    }
  }

  return;
}

/****************************************************************************
 *
 *  write_pe
 *
 *  Write data block relevant for one pe
 *
 ****************************************************************************/

void write_pe(struct io_info * file_info, FILE * fp_data, double * data) {

  int i, ic, jc, kc, index, nr;
  int nlocal[3];

  for(i = 0; i < 3; i++){
    nlocal[i] = file_info->nlocal[i];
  }

  if (output_binary_) {
    for (ic = 1; ic <= nlocal[0]; ic++) {
      for (jc = 1; jc <= nlocal[1]; jc++) {
	for (kc = 1; kc <= nlocal[2]; kc++) {
          index = file_info->nrec*site_index(ic, jc, kc, nlocal);
	  for (nr = 0; nr < file_info->nrec; nr++) {
	    fwrite((data + index + nr), sizeof(double), 1, fp_data);
	  }
	}
      }
    }
  }
  else {
    for (ic = 1; ic <= nlocal[0]; ic++) {
      for (jc = 1; jc <= nlocal[1]; jc++) {
	for (kc = 1; kc <= nlocal[2]; kc++) {
          index = file_info->nrec*site_index(ic, jc, kc, nlocal);
	  for (nr = 0; nr < file_info->nrec; nr++) {
	    fprintf(fp_data, "%le ", *(data + index + nr));
	  }
	  fprintf(fp_data, "\n");
	}
      }
    }
  }

  return;
}

/****************************************************************************
 *
 *  copy_local2global
 *
 *  Copy local data into a global array.
 *
 ****************************************************************************/

void copy_local2global(struct io_info * file_info, double * datalocal, int * noffset, double **** data) {

  int i, ic, jc, kc, ic_g, jc_g, kc_g;
  int index_l, index_g, nr;
  int nlocal[3];

  for(i = 0; i < 3; i++){
    nlocal[i] = file_info->nlocal[i];
  }

  /* Sweep through initial data for this pe */
  for (ic = 1; ic <= nlocal[0]; ic++) {
    ic_g = noffset[0] + ic;
    for (jc = 1; jc <= nlocal[1]; jc++) {
      jc_g = noffset[1] + jc;
      for (kc = 1; kc <= nlocal[2]; kc++) {
	kc_g = noffset[2] + kc;

	for (nr = 0; nr < file_info->nrec; nr++) {
	  index_l = file_info->nrec*site_index(ic, jc, kc, nlocal);
	  data[ic_g][jc_g][kc_g][nr] = *(datalocal + index_l + nr);
	}
      }
    }
  }

  return;
}
/****************************************************************************
 *
 *  copy_global2local
 *
 *  Copy global data into the local array.
 *
 ****************************************************************************/

void copy_global2local(struct io_info * file_info, double **** data, int * noffset, double * datalocal) {

  int ic, jc, kc, ic_g, jc_g, kc_g;
  int index_l, index_g, nr;
  int i,j,k;
  int nlocal[3];


  for(i = 0; i < 3; i++){
    nlocal[i] = file_info->nlocal[i];
  }

  /* Sweep through initial data for this pe */
  for (ic = 1; ic <= nlocal[0]; ic++) {
    ic_g = noffset[0] + ic;
    for (jc = 1; jc <= nlocal[1]; jc++) {
      jc_g = noffset[1] + jc;
      for (kc = 1; kc <= nlocal[2]; kc++) {
	kc_g = noffset[2] + kc;

	for (nr = 0; nr < file_info->nrec; nr++) {
	  index_l = file_info->nrec*site_index(ic, jc, kc, nlocal);
	  *(datalocal + index_l + nr) = data[ic_g][jc_g][kc_g][nr];

	}
      }
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

/****************************************************************************
 *
 *  set_dist
 *
 * Sets initial configuration of distributions to zero velocity
 *
 ****************************************************************************/
void set_dist(double **** dist){

  int i, j, k, l;
  double rho=1.0;
  double wv[NVEL_], w0, w1, w2;

  if(NVEL_==9){
    w0=16.0/36.0; 
    w1=4.0/36.0; 
    w2=1.0/36.0;
    wv[0] = w0;
    wv[1] = w2;
    wv[2] = w1;
    wv[3] = w2;
    wv[4] = w1;
    wv[5] = w1;
    wv[6] = w2;
    wv[7] = w1;
    wv[8] = w2;
  }

  if(NVEL_==19){
    w0=12.0/36.0; 
    w1=2.0/36.0; 
    w2=1.0/36.0;
    wv[0] = w0;
    wv[1] = w2;
    wv[2] = w2;
    wv[3] = w1;
    wv[4] = w2;
    wv[5] = w2;
    wv[6] = w2;
    wv[7] = w1;
    wv[8] = w2;
    wv[9] = w1;
    wv[10] = w1;
    wv[11] = w2;
    wv[12] = w1;
    wv[13] = w2;
    wv[14] = w2;
    wv[15] = w2;
    wv[16] = w1;
    wv[17] = w2;
    wv[18] = w2;
  }

  for (i=1; i<=ntotal_[0]; i++){
    for (j=1; j<=ntotal_[1]; j++){
      for (k=1; k<=ntotal_[2]; k++){
	for (l=0; l<NVEL_; l++){
	  dist[i][j][k][l] = rho*wv[l];
	}
      }
    }
  }

  return;
}

/****************************************************************************
 *
 *  set_phi_cf1
 *
 * Setting configuration for cholesteric finger of the 1st kind (CF-1) 
 *
 ****************************************************************************/
void set_phi_cf1(double **** phi_in, double **** phi){

  int i,j,k;
  double alpha0, beta0, gamma0, nx,ny,nz;


  for (i=1; i<=ntotal_[0]; i++){
    for (j=1; j<=ntotal_[1]; j++){
      for (k=1; k<=ntotal_[2]; k++){


	alpha0 = 0.5*Pi*sin(Pi*k/ntotal_[2]);
	gamma0 = 0.5*Pi*sin(Pi*k/ntotal_[2]);
	beta0 = -2.0*(Pi*k/ntotal_[2]-0.5*Pi);

	nx = cos(beta0)*sin(gamma0)*sin(q0*j)-cos(alpha0)*sin(beta0)*sin(gamma0)*cos(q0*j)+sin(alpha0)*sin(beta0)*cos(gamma0);
	ny = -sin(beta0)*sin(gamma0)*sin(q0*j)-cos(alpha0)*cos(beta0)*sin(gamma0)*cos(q0*j)+sin(alpha0)*cos(beta0)*cos(gamma0);
	nz = sin(alpha0)*sin(gamma0)*cos(q0*j)+cos(alpha0)*cos(gamma0);

	phi[i][j][k][0] = amp*(3.0/2.0*nx*nx-1.0/2.0);
	phi[i][j][k][1] = amp*(3.0/2.0*nx*ny);
	phi[i][j][k][2] = amp*(3.0/2.0*nx*nz);
	phi[i][j][k][3] = amp*(3.0/2.0*ny*ny-1.0/2.0);
	phi[i][j][k][4] = amp*(3.0/2.0*ny*nz);

      }
    }
  }

  return;
}

/****************************************************************************
 *
 *  set_phi_cf2
 *
 * Setting configuration for cholesteric finger of the 2nd kind (CF-2) 
 * 
 * by rotating CF-1 in the lower half around z-axis
 *
 ****************************************************************************/
void set_phi_cf2(double **** phi_in, double **** phi){

  int i,j,k;

  for (i=1; i<=ntotal_[0]; i++){
    for (j=1; j<=ntotal_[1]; j++){
      for (k=1; k<=ntotal_[2]; k++){

	phi[i][j][k][0] = phi_in[i][j][k][0];
	phi[i][j][k][1] = phi_in[i][j][k][1];
	phi[i][j][k][2] = phi_in[i][j][k][2];
	phi[i][j][k][3] = phi_in[i][j][k][3];
	phi[i][j][k][4] = phi_in[i][j][k][4];

	if(k<ntotal_[2]/2.){

	phi[i][j][k][1] = -phi_in[i][j][k][1];
	phi[i][j][k][4] = -phi_in[i][j][k][4];

	}

      }
    }
  }

  return;
}

/****************************************************************************
 *
 *  set_phi_nematic
 *
 * Set nematic configuration  
 *
 ****************************************************************************/
void set_phi_nematic(double nx, double ny, double nz, double **** phi){

  int i,j,k;

  for (i=1; i<=ntotal_[0]; i++){
    for (j=1; j<=ntotal_[1]; j++){
      for (k=1; k<=ntotal_[2]; k++){

	phi[i][j][k][0] = amp*(3.0/2.0*nx*nx-1.0/2.0);
	phi[i][j][k][1] = amp*(3.0/2.0*nx*ny);
	phi[i][j][k][2] = amp*(3.0/2.0*nx*nz);
	phi[i][j][k][3] = amp*(3.0/2.0*ny*ny-1.0/2.0);
	phi[i][j][k][4] = amp*(3.0/2.0*ny*nz);

      }
    }
  }

  return;
}

/****************************************************************************
 *
 *  set_phi_torus
 *
 *  Creates toroidal order parameter configuration 
 *  from 2D cross section (y-z-slice)
 *
 ****************************************************************************/
void set_phi_torus(double **** phi_in, double * R, double ** Z, double **** phi){


  double x[3], xr[3]; /* 3D coordinate vector on 2D-slice and rotated vector */
  double y[3]; /* output coordinate */
  double Q[3][3], Qr[3][3]; /* OP-tensor on 2D-slice and rotated tensor */

  int i,j,k,l,n;
  int ii, ij, ik;
  double **M, **Mt; /* rotation matrices */ 
  double alpha, dalpha=0.0001*Pi; /* rotation angle and increment */
  double thres = 1e-15;

  M = (double **) calloc(3, sizeof(double));
  Mt = (double **) calloc(3, sizeof(double));
  for (i = 0; i < 3; i++){
    M[i] = (double *) calloc(3, sizeof(double));
    Mt[i] = (double *) calloc(3, sizeof(double));
  }


  for (n=0; n< N_torus_; n++){
    /* rotate 2D-slice to form a torus */
    for (alpha=0; alpha<=2.0*Pi; alpha+=dalpha){
    printf("Creating torus %d; alpha = %g\r",n, alpha);

      Mz(M,Mt,alpha);
   
      /* sweep through input data and set OP on torus */
      for (ii=1; ii<=ntotal_in_[0]; ii++){
	if (ii>1) {printf("Torus creation failed: input is not 2D\n"); exit(1);}
	for (ij=1; ij<=ntotal_in_[1]; ij++){
	  for (ik=1; ik<=ntotal_in_[2]; ik++){

	    /* local coordinate on 2D cross section */
	    x[0] = ii-1;
	    if (R[n]<0)  x[1] = R[n] + ij-1;
	    else  x[1] = R[n] - ntotal_in_[1] + ij;
	    x[2] = ik-1;

	    /* OP on 2D cross section */
	    Q[0][0] = phi_in[ii][ij][ik][0];
	    Q[0][1] = phi_in[ii][ij][ik][1]; 
	    Q[0][2] = phi_in[ii][ij][ik][2]; 
	    Q[1][1] = phi_in[ii][ij][ik][3]; 
	    Q[1][2] = phi_in[ii][ij][ik][4]; 
	    Q[1][0] = Q[0][1];
	    Q[2][0] = Q[0][2];
	    Q[2][1] = Q[1][2];
	    Q[2][2] = -Q[0][0]-Q[1][1]; 

	    /* apply rotation to coordinate vector and tensor */
	    for (i=0; i<3; i++){
	      xr[i] = 0.0;
	      for (j=0; j<3; j++){
		xr[i] += M[i][j] * x[j]; 
	      }
	    }

	    for (i=0; i<3; i++){
	      for (j=0; j<3; j++){
		Qr[i][j] = 0.0;
		for (k=0; k<3; k++){
		  for (l=0; l<3; l++){
		    Qr[i][j] += Mt[i][k] * Q[k][l] * M[l][j];
		  }
		}
	      }
	    }

	    /* determine output coordinate vector */
	    y[0] = (xr[0] + Z[n][0]);
	    if (y[0]<1) y[0] = 1;
	    if (y[0]>ntotal_[0]) y[0] = ntotal_[0];
	    y[1] = (xr[1] + Z[n][1]);
	    if (y[1]<1) y[0] = 1;
	    if (y[1]>ntotal_[1]) y[1] = ntotal_[1];
	    y[2] = (xr[2] + Z[n][2]);
	    if (y[2]<1) y[0] = 1;
	    if (y[2]>ntotal_[2]) y[2] = ntotal_[2];
 
	    phi[(int)(y[0])][(int)(y[1])][(int)(y[2])][0] = Qr[0][0];
	    phi[(int)(y[0])][(int)(y[1])][(int)(y[2])][1] = Qr[0][1];
	    phi[(int)(y[0])][(int)(y[1])][(int)(y[2])][2] = Qr[0][2];
	    phi[(int)(y[0])][(int)(y[1])][(int)(y[2])][3] = Qr[1][1];
	    phi[(int)(y[0])][(int)(y[1])][(int)(y[2])][4] = Qr[1][2];

	    }
	  }
	}
      }

    printf("\n");
  }
  return;
}

/****************************************************************************
 *
 *  Mz
 *
 *  Rotation matrix and its inverse 
 *
 ****************************************************************************/
void Mz(double ** M, double ** Mt, double alpha){

  M[0][0] = cos(alpha);
  M[0][1] = -sin(alpha);
  M[0][2] = 0;
  M[1][0] = sin(alpha);
  M[1][1] = cos(alpha);
  M[1][2] = 0;
  M[2][0] = 0;
  M[2][1] = 0;
  M[2][2] = 1;

  Mt[0][0] = M[0][0];
  Mt[0][1] = M[1][0];
  Mt[0][2] = M[2][0];
  Mt[1][0] = M[0][1];
  Mt[1][1] = M[1][1];
  Mt[1][2] = M[2][1];
  Mt[2][0] = M[0][2];
  Mt[2][1] = M[1][2];
  Mt[2][2] = M[2][2];

  return;
}

/****************************************************************************
 *
 *  set_torus_radius_centre
 *
 *  Sets the radii and centres of the tori that are created 
 *  from 2D cross section (y-z-slice) in set_phi_torus
 *
 *  Note: radius R<0 means the slice of the cross
 *        section is on the lhs at alpha=0 wrt Z.
 *        This allows to create 'inside-out' tori.
 *
 ****************************************************************************/
void set_torus_radius_centre(double * R, double ** Z){

  R[0] = ntotal_in_[1];
  Z[0][0] = ntotal_[0]/4.0; //2.0 
  Z[0][1] = ntotal_[1]/4.0; //2.0
  Z[0][2] = 0; 

  R[1] = ntotal_in_[1];
  Z[1][0] = 3.0*ntotal_[0]/4.0; 
  Z[1][1] = 3.0*ntotal_[1]/4.0; 
  Z[1][2] = 0; 

return;
}
