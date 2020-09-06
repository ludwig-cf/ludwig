/*****************************************************************************
 *
 *  capillary.c
 *
 *  Compile and link with -lm for the maths library. 
 *
 *  This utility produces an output file suitable for initialising
 *  a capillary structure in Ludwig. 
 *
 *  It is assumed the periodic dimension is z, and that the system
 *  is square in the x-y directions. No 'leakage' in the x-y
 *  directions is ensured by making the last site in each direction
 *  solid.
 *
 *  The various system parameters should be set at comiple time,
 *  and are described below. The output file is always in BINARY
 *  format.
 *
 *  1. Output capaillary structure
 *  Set the required parameters and invoke with no argument
 *      ./a.out
 *   
 *  2. Profiles
 *  If the program is invoked with a single phi output file
 *  argument, e.g.,
 *      ./a.out phi-001000.001-001
 *  a scatter plot of the profile of the interface in the
 *  wetting half of the capillary will be produced. That
 *  is height vs r, the radial distance from the centre.
 *  The output file should match the capillary structure!
 *
 *  Edinburgh Soft Matter and Statistcal Physics Group and
 *  Edinburgh Parallel Computing Centre
 *  (c) 2008-2019 The University of Edinburgh
 *
 *  Contributing authors:
 *  Kevin Stratford (kevin@epcc.ed.ac.uk)
 *
 *****************************************************************************/

#include <assert.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>

/* This is a copy from ../src/map.h; it would be better to include
 * directly, but that incurs additional dependencies on targetDP.h */

enum map_status {MAP_FLUID, MAP_BOUNDARY, MAP_COLLOID, MAP_STATUS_MAX};

/* SYSTEM SIZE */
/* Set the system size as desired. Clearly, this must match the system
 * set in the main input file for Ludwig. */

const int xmax = 20;
const int ymax = 20;
const int zmax = 20;

const int crystalline_cell_size = 10; //added for implementation of crystalline capillaries

/* CROSS SECTION */
/* You can choose a square or circular cross section */

enum {CIRCLE, SQUARE, XWALL, YWALL, ZWALL, XWALL_OBSTACLES, XWALL_BOTTOM,
      SPECIAL_CROSS, SCC, BCC, FCC}; //changed for implementation of crystalline capillaries
const int xsection = YWALL;

/*Modify the local geometry of the wall*/

int obstacle_number = 1; /* number of obstacles per wall */
int obstacle_length = 6; /* along the wall direction */
int obstacle_height = 10; /* perpendicular from wall */
int obstacle_depth  = 6; /* perpendicular to length and height */
			 /* NOTE: obstacle_depth == xmax/ymax/zmax 
				  means obstacles don't have a z-boundary */

/* SURFACE CHARGE */

const double sigma = 0.125;

/* FREE ENERGY PARAMETERS */
/* Set the fluid and solid free energy parameters. The fluid parameters
 * must match those used in the main calculation. See Desplat et al.
 * Comp. Phys. Comm. (2001) for details. */

const double kappa = 0.053;
const double B = 0.0625;
const double H = 0.00;
const double C = 0.000;	// Following Desplat et al.

/* WETTING */
/* A section of capillary between z1 and z2 (inclusive) will have
 * wetting property H = H, the remainder H = 0 */

const int z1 = 1;
const int z2 = 36;

/* OUTPUT */
/* You can generate a file with solid/fluid status information only,
 * or one which includes the wetting parameter H or charge Q. */

/* Wetting: Please don't use STATUS_WITH_H; use STATUS_WITH_C_H
   and set C = 0, if you just want H */

enum {STATUS_ONLY, STATUS_WITH_H, STATUS_WITH_C_H, STATUS_WITH_SIGMA};
const int output_type = STATUS_ONLY;

/* OUTPUT FILENAME */

const char * outputfilename = "capillary.001-001";

static void profile(const char *);

int map_special_cross(char * map, int * nsolid);

/*****************************************************************************
 *
 *  main program
 *
 *****************************************************************************/

int main(int argc, char ** argv) {


  char * map_in;
  FILE * fp_orig;
  int i, j, k, n;
  int nsolid = 0;
  int k_pic;

  double * map_h; // for wetting coefficient H
  double * map_c; // for additional wetting coefficient C

  double * map_sig; // for (surface) charge

  double rc = 0.5*(xmax-2);
  double x0 = 0.5*xmax + 0.5;
  double y0 = 0.5*ymax + 0.5;
  double x, y, r;
  double h, h1, theta;

  //added for implementation of crystalline capillaries
  double crystalline_cell_radius;
  double diff_x_edges, diff_y_edges, diff_z_edges, r_edges;

  int iobst;
  int obst_start[2*obstacle_number][3];
  int obst_stop[2*obstacle_number][3];
  int gap_length;

  FILE  * WriteFile;	
  char  file[800];

  if (argc == 2) profile(argv[1]);

  if (output_type == STATUS_WITH_H || output_type == STATUS_WITH_C_H) {

    printf("Free energy parameters:\n");
    printf("free energy parameter kappa = %f\n", kappa);
    printf("free energy parameter B     = %f\n", B);
    printf("surface free energy   H     = %f\n", H);
    h = H*sqrt(1.0/(kappa*B));
    printf("dimensionless parameter h   = %f\n", h);
    h1 = 0.5*(-pow(1.0 - h, 1.5) + pow(1.0 + h, 1.5));
    printf("dimensionless parameter h1=cos(theta)   = %f\n", h1);
    theta = acos(h1);
    printf("contact angle theta         = %f radians\n", theta);
    theta = theta*180.0/(4.0*atan(1.0));
    printf("                            = %f degrees\n", theta);

  }

  if (output_type == STATUS_WITH_SIGMA) {
    printf("Surface charge sigma = %f\n", sigma);
  }

  map_in = (char *) malloc(xmax*ymax*zmax*sizeof(char));
  if (map_in == NULL) exit(-1);

  map_h = (double *) malloc(xmax*ymax*zmax*sizeof(double));
  if (map_h == NULL) exit(-1);

  map_c = (double *) malloc(xmax*ymax*zmax*sizeof(double));
  if (map_c == NULL) exit(-1);

  map_sig = (double *) malloc(xmax*ymax*zmax*sizeof(double));
  if (map_sig == NULL) exit(-1);

  k_pic = 0; /* Picture */

  /* Begin switch */
  switch (xsection) {

  case SPECIAL_CROSS:
    map_special_cross(map_in, &nsolid);
    k_pic = 1;
    break;

  case CIRCLE:
    for (i = 0; i < xmax; i++) {
      x = 1.0 + i - x0;
      for (j = 0; j < ymax; j++) {
	y = 1.0 + j - y0;
	for (k = 0; k < zmax; k++) {
	  n = ymax*zmax*i + zmax*j + k;

	  map_in[n] = MAP_BOUNDARY;
	  if (output_type == STATUS_WITH_H) { map_h[n] = 0.0; }
	  if (output_type == STATUS_WITH_C_H) { map_h[n] = 0.0; map_c[n] = 0.0; }
	  map_sig[n] = 0.0;

	  /* Fluid if r(x,y) <= capillary width (L/2) */
	  r = sqrt(x*x + y*y);
	  if (r <= rc) {
	    map_in[n] = MAP_FLUID;
	  }

	  if (map_in[n] == MAP_BOUNDARY) {
	    nsolid++;
	    if (k >= z1 && k <= z2) {
	      if (output_type == STATUS_WITH_H) { map_h[n] = H; }
	      if (output_type == STATUS_WITH_C_H) { map_h[n] = H; map_c[n] = C; }
	      map_sig[n] = sigma;
	    }
	  }
	}
      }
    }

    break;

  case SCC: ;
    /* simple cubic crystal */

    if(xmax % crystalline_cell_size != 0 || ymax % crystalline_cell_size != 0 ||
    zmax % crystalline_cell_size !=0){
        printf("ERROR: wrong ratio of the capillary dimension with respect "
        "to the crystalline cell size. Please check the parameters "
        "xmax, ymax, zmax, and crystalline_cell_size!\n");
        exit(-1);
    }

    k_pic = 0; //picture
    printf("k_pic: %d\n", k_pic);

    //radius of crystalline particle
    crystalline_cell_radius = 0.5 * crystalline_cell_size;
    double center_i, center_j, center_k;
    double diff_x, diff_y, diff_z;
    printf("crystalline_cell_radius: %f \n", crystalline_cell_radius);

    for (i = 0; i < xmax; i++) {
      for (j = 0; j < ymax; j++) {
	    for (k = 0; k < zmax; k++) {
            //distance between the node (i,j,k) and the centre of the nearest crystalline particle,
            //located at the edges of the crystalline cell
	        diff_x = i - round((double)i/crystalline_cell_size) * crystalline_cell_size;
	        diff_y = j - round((double)j/crystalline_cell_size) * crystalline_cell_size;
	        diff_z = k - round((double)k/crystalline_cell_size) * crystalline_cell_size;

	        r = sqrt(diff_x*diff_x + diff_y*diff_y + diff_z*diff_z);

	        n = ymax*zmax*i + zmax*j + k;

	        map_in[n] = MAP_FLUID;
	        if (output_type == STATUS_WITH_H) { map_h[n] = 0.0; }
	        if (output_type == STATUS_WITH_C_H) { map_h[n] = 0.0; map_c[n] = 0.0; }
	        map_sig[n] = 0.0;

	        if(r <= crystalline_cell_radius){
	            nsolid++;
	            map_in[n] = MAP_BOUNDARY;
	            if (output_type == STATUS_WITH_H) { map_h[n] = H; }
	            if (output_type == STATUS_WITH_C_H) { map_h[n] = H; map_c[n] = C; }
	            map_sig[n] = sigma;
	        }
	    }
	  }
	}
	break;

  case BCC: ;
    /* body centered cubic crystal */

    if(xmax % crystalline_cell_size != 0 || ymax % crystalline_cell_size != 0 ||
    zmax % crystalline_cell_size !=0){
        printf("ERROR: wrong ratio of the capillary dimension with respect "
        "to the crystalline cell size. Please check the parameters "
        "xmax, ymax, zmax, and crystalline_cell_size!\n");
        exit(-1);
    }

    k_pic = 5; //picture
    printf("k_pic: %d\n", k_pic);

    //radius of crystalline particle
    crystalline_cell_radius = 0.25 * sqrt(3) * crystalline_cell_size;
    double diff_x_center, diff_y_center, diff_z_center, r_center;
    printf("crystalline_cell_radius: %f \n", crystalline_cell_radius);

    for (i = 0; i < xmax; i++) {
      for (j = 0; j < ymax; j++) {
	    for (k = 0; k < zmax; k++) {
	        //distance between the node (i,j,k) and the centre of the nearest crystalline particle,
            //located at the edges of the crystalline cell
	        diff_x_edges = i - round((double)i/crystalline_cell_size) * crystalline_cell_size;
	        diff_y_edges = j - round((double)j/crystalline_cell_size) * crystalline_cell_size;
	        diff_z_edges = k - round((double)k/crystalline_cell_size) * crystalline_cell_size;

	        r_edges = sqrt(diff_x_edges*diff_x_edges + diff_y_edges*diff_y_edges + diff_z_edges*diff_z_edges);

	        //distance between the node (i,j,k) and the centre of the crystalline particle,
            //located at the centre of the crystalline cell
	        diff_x_center = i - (floor((double)i/crystalline_cell_size) + 0.5) * crystalline_cell_size;
	        diff_y_center = j - (floor((double)j/crystalline_cell_size) + 0.5) * crystalline_cell_size;
	        diff_z_center = k - (floor((double)k/crystalline_cell_size) + 0.5) * crystalline_cell_size;

	        r_center = sqrt(diff_x_center*diff_x_center + diff_y_center*diff_y_center + diff_z_center*diff_z_center);

	        n = ymax*zmax*i + zmax*j + k;

	        map_in[n] = MAP_FLUID;
	        if (output_type == STATUS_WITH_H) { map_h[n] = 0.0; }
	        if (output_type == STATUS_WITH_C_H) { map_h[n] = 0.0; map_c[n] = 0.0; }
	        map_sig[n] = 0.0;

	        if(r_edges <= crystalline_cell_radius || r_center <= crystalline_cell_radius){
	            nsolid++;
	            map_in[n] = MAP_BOUNDARY;
	            if (output_type == STATUS_WITH_H) { map_h[n] = H; }
	            if (output_type == STATUS_WITH_C_H) { map_h[n] = H; map_c[n] = C; }
	            map_sig[n] = sigma;
	        }
	    }
	  }
	}
	break;

  case FCC: ;
    /* face centered cubic crystal */

    if(xmax % crystalline_cell_size != 0 || ymax % crystalline_cell_size != 0 ||
    zmax % crystalline_cell_size !=0){
        printf("ERROR: wrong ratio of the capillary dimension with respect "
        "to the crystalline cell size. Please check the parameters "
        "xmax, ymax, zmax, and crystalline_cell_size!\n");
        exit(-1);
    }

    k_pic = 0; //picture
    printf("k_pic: %d\n", k_pic);

    //radius of crystalline particle
    crystalline_cell_radius = 0.25 * sqrt(2) * crystalline_cell_size;
    double diff_x_center_xy, diff_y_center_xy, diff_z_center_xy, r_center_xy;
    double diff_x_center_xz, diff_y_center_xz, diff_z_center_xz, r_center_xz;
    double diff_x_center_yz, diff_y_center_yz, diff_z_center_yz, r_center_yz;
    printf("crystalline_cell_radius: %f \n", crystalline_cell_radius);

    for (i = 0; i < xmax; i++) {
      for (j = 0; j < ymax; j++) {
	    for (k = 0; k < zmax; k++) {
	        //distance between the node (i,j,k) and the centre of the nearest crystalline particle,
            //located at the edges of the crystalline cell
	        diff_x_edges = i - round((double)i/crystalline_cell_size) * crystalline_cell_size;
	        diff_y_edges = j - round((double)j/crystalline_cell_size) * crystalline_cell_size;
	        diff_z_edges = k - round((double)k/crystalline_cell_size) * crystalline_cell_size;

	        r_edges = sqrt(diff_x_edges*diff_x_edges + diff_y_edges*diff_y_edges + diff_z_edges*diff_z_edges);

	        //distance between the node (i,j,k) and the centre of the crystalline particle,
            //located at the centre of the xy-surface in the crystalline cell
	        diff_x_center_xy = i - (floor((double)i/crystalline_cell_size) + 0.5) * crystalline_cell_size;
	        diff_y_center_xy = j - (floor((double)j/crystalline_cell_size) + 0.5) * crystalline_cell_size;
	        diff_z_center_xy = k - round((double)k/crystalline_cell_size) * crystalline_cell_size;

	        r_center_xy = sqrt(diff_x_center_xy*diff_x_center_xy + diff_y_center_xy*diff_y_center_xy + diff_z_center_xy*diff_z_center_xy);

	        //distance between the node (i,j,k) and the centre of the crystalline particle,
            //located at the centre of the xz-surface in the crystalline cell
	        diff_x_center_xz = i - (floor((double)i/crystalline_cell_size) + 0.5) * crystalline_cell_size;
	        diff_y_center_xz = j - round((double)j/crystalline_cell_size) * crystalline_cell_size;
	        diff_z_center_xz = k - (floor((double)k/crystalline_cell_size) + 0.5) * crystalline_cell_size;

	        r_center_xz = sqrt(diff_x_center_xz*diff_x_center_xz + diff_y_center_xz*diff_y_center_xz + diff_z_center_xz*diff_z_center_xz);

	        //distance between the node (i,j,k) and the centre of the crystalline particle,
            //located at the centre of the yz-surface in the crystalline cell
	        diff_x_center_yz = i - round((double)i/crystalline_cell_size) * crystalline_cell_size;
	        diff_y_center_yz = j - (floor((double)j/crystalline_cell_size) + 0.5) * crystalline_cell_size;
	        diff_z_center_yz = k - (floor((double)k/crystalline_cell_size) + 0.5) * crystalline_cell_size;

	        r_center_yz = sqrt(diff_x_center_yz*diff_x_center_yz + diff_y_center_yz*diff_y_center_yz + diff_z_center_yz*diff_z_center_yz);

	        n = ymax*zmax*i + zmax*j + k;

	        map_in[n] = MAP_FLUID;
	        if (output_type == STATUS_WITH_H) { map_h[n] = 0.0; }
	        if (output_type == STATUS_WITH_C_H) { map_h[n] = 0.0; map_c[n] = 0.0; }
	        map_sig[n] = 0.0;

	        if(r_edges <= crystalline_cell_radius || r_center_xy <= crystalline_cell_radius ||
	            r_center_xz <= crystalline_cell_radius || r_center_yz <= crystalline_cell_radius){
	            nsolid++;
	            map_in[n] = MAP_BOUNDARY;
	            if (output_type == STATUS_WITH_H) { map_h[n] = H; }
	            if (output_type == STATUS_WITH_C_H) { map_h[n] = H; map_c[n] = C; }
	            map_sig[n] = sigma;
	        }
	    }
	  }
	}
	break;

  case SQUARE:

    /* Square */

    for (i = 0; i < xmax; i++) {
      for (j = 0; j < ymax; j++) {
	for (k = 0; k < zmax; k++) {

	  n = ymax*zmax*i + zmax*j + k;

	  map_in[n] = MAP_FLUID;
	  if (output_type == STATUS_WITH_H) { map_h[n] = 0.0; }
	  if (output_type == STATUS_WITH_C_H) { map_h[n] = 0.0; map_c[n] = 0.0; }
	  map_sig[n] = 0.0;
	  
	  if (i == 0 || j == 0 || i == xmax - 1 || j == ymax - 1) {
	    map_in[n] = MAP_BOUNDARY;
	  }

	  if (map_in[n] == MAP_BOUNDARY) {
	    nsolid++;
	    if (k >= z1 && k <= z2) {
	      if (output_type == STATUS_WITH_H) { map_h[n] = H; }
       	      if (output_type == STATUS_WITH_C_H) { map_h[n] = H; map_c[n] = C; }
	      map_sig[n] = sigma;
	    }
	  }
	}
      }
    }

    break;


  case XWALL:

    for (i = 0; i < xmax; i++) {
      for (j = 0; j < ymax; j++) {
	for (k = 0; k < zmax; k++) {
	  n = ymax*zmax*i + zmax*j + k;
	  map_in[n] = MAP_FLUID;
	  if (output_type == STATUS_WITH_H) { map_h[n] = 0.0; } 
	  if (output_type == STATUS_WITH_C_H) { map_h[n] = 0.0; map_c[n] = 0.0; }
	  map_sig[n] = 0.0;
	  if (i == 0 || i == xmax - 1) {
	    map_in[n] = MAP_BOUNDARY;
	    if (output_type == STATUS_WITH_SIGMA) {
            map_sig[n] = sigma;
	    }
	    if (output_type == STATUS_WITH_H) { map_h[n] = H; }
	    if (output_type == STATUS_WITH_C_H) { map_h[n] = H; map_c[n] = C; }
	    ++nsolid;
	  }
	}
      }
    }

    break;


  case XWALL_BOTTOM:

    for (i = 0; i < xmax; i++) {
      for (j = 0; j < ymax; j++) {
	for (k = 0; k < zmax; k++) {
	  n = ymax*zmax*i + zmax*j + k;
	  map_in[n] = MAP_FLUID;
	  if (output_type == STATUS_WITH_H) { map_h[n] = 0.0; } 
	  if (output_type == STATUS_WITH_C_H) { map_h[n] = 0.0; map_c[n] = 0.0; }
	  map_sig[n] = 0.0;
	  if (i == 0) {
	    map_in[n] = MAP_BOUNDARY;
	    if (output_type == STATUS_WITH_SIGMA) {
            map_sig[n] = sigma;
	    }
	    if (output_type == STATUS_WITH_H) { map_h[n] = H; }
	    if (output_type == STATUS_WITH_C_H) { map_h[n] = H; map_c[n] = C; }
	    ++nsolid;
	  }
	  if (i == xmax - 1) {
	    map_in[n] = MAP_BOUNDARY;
	    if (output_type == STATUS_WITH_SIGMA) {
            map_sig[n] = sigma;
	    }
	    if (output_type == STATUS_WITH_H) { map_h[n] = 0.0; }
	    if (output_type == STATUS_WITH_C_H) { map_h[n] = 0.0; map_c[n] = 0.0; }
	    ++nsolid;
	  }


	}
      }
    }

    break;


  case XWALL_OBSTACLES:

    printf("\n%d obstacles per x-wall of length %d, height %d and depth %d\n\n", 
	      obstacle_number, obstacle_length, obstacle_height, obstacle_depth);

    /* define obstacles on bottom wall */
    for (iobst = 0; iobst < obstacle_number; iobst++) {

      /* height of obstacle */
      obst_start[iobst][0] = 1;
      obst_stop[iobst][0]  = obst_start[iobst][0] + obstacle_height - 1;

      /* y-gap between two obstacles along x-wall */
      gap_length = (ymax-obstacle_number*obstacle_length)/(obstacle_number);
      obst_start[iobst][1] = gap_length/2 + iobst*(gap_length+obstacle_length);
      obst_stop[iobst][1]  = obst_start[iobst][1] + obstacle_length - 1;

      /* gap between boundary and obstacle if centrally positioned along z */
      gap_length = (zmax-obstacle_depth)/2;
      obst_start[iobst][2] = gap_length;
      obst_stop[iobst][2]  = obst_start[iobst][2] + obstacle_depth - 1;

      printf("Obstacle %d x-position %d to %d, y-position %d to %d, z-position %d to %d\n", iobst, 
	obst_start[iobst][0],obst_stop[iobst][0],obst_start[iobst][1],obst_stop[iobst][1],obst_start[iobst][2],obst_stop[iobst][2]);
    }

    /* define obstacles on top wall */
    for (iobst = obstacle_number; iobst < 2*obstacle_number; iobst++) {

      /* height of obstacle */
      obst_start[iobst][0] = xmax - 1 - obstacle_height;
      obst_stop[iobst][0]  = xmax - 1 - 1;

      /* y-gap between two obstacles along x-wall */
      gap_length = (ymax-obstacle_number*obstacle_length)/(obstacle_number);
      obst_start[iobst][1] = gap_length/2 + (iobst-obstacle_number)*(gap_length+obstacle_length);
      obst_stop[iobst][1]  = obst_start[iobst][1] + obstacle_length - 1;

      /* gap between boundary and obstacle if centrally positioned along z */
      gap_length = (zmax-obstacle_depth)/2;
      obst_start[iobst][2] = gap_length;
      obst_stop[iobst][2]  = obst_start[iobst][2] + obstacle_depth - 1;

      printf("Obstacle %d x-position %d to %d, y-position %d to %d, z-position %d to %d\n", iobst, 
	obst_start[iobst][0],obst_stop[iobst][0],obst_start[iobst][1],obst_stop[iobst][1],obst_start[iobst][2],obst_stop[iobst][2]);
    }

    sprintf(file,"Configuration_capillary.dat");
    WriteFile=fopen(file,"w");
    fprintf(WriteFile,"#i j k map sigma\n");

    for (i = 0; i < xmax; i++) {
      for (j = 0; j < ymax; j++) {
	for (k = 0; k < zmax; k++) {

	  n = ymax*zmax*i + zmax*j + k;
	  map_in[n]  = MAP_FLUID;
	  map_sig[n] = 0.0;

	  /* x-walls */

	  if (i == 0 || i == xmax-1) {

	    map_in[n] = MAP_BOUNDARY;
	    ++nsolid;

	    if (output_type == STATUS_WITH_SIGMA) {

	      /* set wall charge */
	      map_sig[n] = sigma; 

	      /* remove charge at contact lines with obstacles */
	      for (iobst = 0; iobst < 2*obstacle_number ; iobst++) {
		if (j >= obst_start[iobst][1] && j <= obst_stop[iobst][1] 
		    && k >= obst_start[iobst][2] && k <= obst_stop[iobst][2]){
		  map_sig[n] = 0.0;
		}
	      }

	    }

	  }

	  /* obstacles on bottom wall */
	  for (iobst = 0; iobst < obstacle_number ; iobst++) {

	    if (i >= obst_start[iobst][0] && i <= obst_stop[iobst][0]
	     && j >= obst_start[iobst][1] && j <= obst_stop[iobst][1]
	     && k >= obst_start[iobst][2] && k <= obst_stop[iobst][2]) {

	      map_in[n] = MAP_BOUNDARY;
	      ++nsolid;

	      if (output_type == STATUS_WITH_SIGMA) { 

		// add charge to x-boundary of obstacle
		if (i == obst_stop[iobst][0]) map_sig[n] = sigma; 
		// add charge to y-boundary of obstacle
		if (j == obst_start[iobst][1] || j == obst_stop[iobst][1]) map_sig[n] = sigma;		
		// add charge to z-boundary of obstacle, no boundary for obstacle_depth == zmax
		if ((k == obst_start[iobst][2] || k == obst_stop[iobst][2]) && obstacle_depth != zmax) map_sig[n] = sigma;		

	      }
	    }
	  } 

	  /* obstacles on top wall */
	  for (iobst = obstacle_number; iobst < 2*obstacle_number ; iobst++) {

	    if (i >= obst_start[iobst][0] && i <= obst_stop[iobst][0]
	     && j >= obst_start[iobst][1] && j <= obst_stop[iobst][1]
	     && k >= obst_start[iobst][2] && k <= obst_stop[iobst][2]) {

	      map_in[n] = MAP_BOUNDARY;
	      ++nsolid;

	      if (output_type == STATUS_WITH_SIGMA) { 

		// add charge to x-boundary of obstacle
		if (i == obst_start[iobst][0]) map_sig[n] = sigma; 
		// add charge to y-boundary of obstacle
		if (j == obst_start[iobst][1] || j == obst_stop[iobst][1]) map_sig[n] = sigma;		
		// add charge to z-boundary of obstacl, no boundary for obstacle_depth == zmax
		if ((k == obst_start[iobst][2] || k == obst_stop[iobst][2]) && obstacle_depth != zmax) map_sig[n] = sigma;		

	      }
	    }
	  }

	  fprintf(WriteFile,"%d %d %d %d %f\n", i, j, k, map_in[n], map_sig[n]);


	}
      }
    }

    fclose(WriteFile);

    break;


  case YWALL:

    for (i = 0; i < xmax; i++) {
      for (j = 0; j < ymax; j++) {
	for (k = 0; k < zmax; k++) {
	  n = ymax*zmax*i + zmax*j + k;
	  map_in[n] = MAP_FLUID;
	  map_sig[n] = 0.0;
	  if (output_type == STATUS_WITH_H) { map_h[n] = 0.0; }
          if (output_type == STATUS_WITH_C_H) { map_h[n] = 0.0; map_c[n] = 0.0; }
	  if (j == 0 || j == ymax - 1) {
	    map_in[n] = MAP_BOUNDARY;
	    if (output_type == STATUS_WITH_SIGMA) {
	      map_sig[n] = sigma;
	    }
	    if (output_type == STATUS_WITH_H) { map_h[n] = H; }
            if (output_type == STATUS_WITH_C_H) { map_h[n] = H; map_c[n] = C; }
	    ++nsolid;
	  }
	}
      }
    }
    break;


  case ZWALL:

    for (i = 0; i < xmax; i++) {
      for (j = 0; j < ymax; j++) {
	for (k = 0; k < zmax; k++) {
	  n = ymax*zmax*i + zmax*j + k;
	  map_in[n] = MAP_FLUID;
	  map_sig[n] = 0.0;
	  if (output_type == STATUS_WITH_H) { map_h[n] = 0.0; }
          if (output_type == STATUS_WITH_C_H) { map_h[n] = 0.0; map_c[n] = 0.0; }
	  if (k == 0 || k == zmax - 1) {
	    map_in[n] = MAP_BOUNDARY;
	    if (output_type == STATUS_WITH_SIGMA) {
	      map_sig[n] = sigma;
	    }
	    if (output_type == STATUS_WITH_H) { map_h[n] = H; }
            if (output_type == STATUS_WITH_C_H) { map_h[n] = H; map_c[n] = C; }
	    ++nsolid;
	  }
	}
      }
    }

    break;
  default:
    printf("No cross-section!\n");
    /* End switch */
  }

  /* picture */

  printf("\nCross section (%d = fluid, %d = solid)\n", MAP_FLUID, MAP_BOUNDARY);

  for (i = 0; i < xmax; i++) {
    for (j = 0; j < ymax; j++) {
	n = ymax*zmax*i + zmax*j + k_pic;
      
	if (map_in[n] == MAP_BOUNDARY) printf(" %d", MAP_BOUNDARY);
	if (map_in[n] == MAP_FLUID)    printf(" %d", MAP_FLUID);
    }
    printf("\n");
  }


  if (output_type == STATUS_WITH_H)  {
    sprintf(file,"Configuration_capillary.dat");
    WriteFile=fopen(file,"w");
    fprintf(WriteFile,"#x y z n map H\n");

    for (i = 0; i < xmax; i++) {
      for (j = 0; j < ymax; j++) {
	for (k = 0; k < zmax; k++) {

	n = ymax*zmax*i + zmax*j + k;

	if (map_in[n] == MAP_BOUNDARY) { fprintf(WriteFile,"%i %i %i %i %d %f\n", i, j, k, n, MAP_BOUNDARY, map_h[n]); }
	if (map_in[n] == MAP_FLUID)    { fprintf(WriteFile,"%i %i %i %i %d %f\n", i, j, k, n, MAP_FLUID, map_h[n]); }

	}  
      }  
    }
    fclose(WriteFile);
  }


  if (output_type == STATUS_WITH_C_H)  {
    sprintf(file,"Configuration_capillary.dat");
    WriteFile=fopen(file,"w");
    fprintf(WriteFile,"#x y z n map H C\n");

    for (i = 0; i < xmax; i++) {
      for (j = 0; j < ymax; j++) {
	for (k = 0; k < zmax; k++) {

	n = ymax*zmax*i + zmax*j + k;
      
	if (map_in[n] == MAP_BOUNDARY) { fprintf(WriteFile,"%i %i %i %i %d %f %f\n", i, j, k, n, MAP_BOUNDARY, map_h[n], map_c[n]); }
	if (map_in[n] == MAP_FLUID)    { fprintf(WriteFile,"%i %i %i %i %d %f %f\n", i, j, k, n, MAP_FLUID, map_h[n], map_c[n]); }

	}  
      }  
    }
    fclose(WriteFile);
  }


  //printf("n = %d nsolid = %d nfluid = %d\n", xmax*ymax*zmax, nsolid,
	// xmax*ymax*zmax - nsolid);
  //changed for implementation of crystalline capillaries in order to see the volume fraction of the crystals
  printf("n = %d nsolid = %d nfluid = %d nsolid fraction: %f \n", xmax*ymax*zmax, nsolid,
	 xmax*ymax*zmax - nsolid, (double)nsolid/(xmax*ymax*zmax));

  /* Write new data as char */

  fp_orig = fopen(outputfilename, "w");
  if (fp_orig == NULL) {
    printf("Cant open output\n");
    exit(-1);
  }

  for (i = 0; i < xmax; i++) {
    for (j = 0; j < ymax; j++) {
      for (k = 0; k < zmax; k++) {
	n = ymax*zmax*i + zmax*j + k;

	fputc(map_in[n], fp_orig);
	if (output_type == STATUS_WITH_H) {
	  fwrite(map_h + n, sizeof(double), 1, fp_orig);
	}
	if (output_type == STATUS_WITH_C_H) {
	  fwrite(map_c + n, sizeof(double), 1, fp_orig);
	  fwrite(map_h + n, sizeof(double), 1, fp_orig);
	}

	if (output_type == STATUS_WITH_SIGMA) {
	  fwrite(map_sig + n, sizeof(double), 1, fp_orig);
	}
      }
    }
  }

  fclose(fp_orig);

  free(map_in);
  free(map_c);
  free(map_h);
  free(map_sig);

  return 0;
}

/*****************************************************************************
 *
 *  profile
 *
 *  This attempts to give a profile of the interface as a function
 *  of the radial distance from the centre of the capillary defined
 *  above.
 *
 *  For each position (i, j) we examine the 1-d profile phi(z) and
 *  locate the position of zero by linear interpolation. The results
 *  go to standard output.
 *
 *  Note that the test for the interface assumes the phi = -1 block
 *  is in the middle of the system (see block initialisation in
 *  src/phi_stats.c).
 *
 *****************************************************************************/

static void profile(const char * filename) {

  int ic, jc, kc, index;
  int inside;
  int nread;
  double rc = 0.5*(xmax-2);
  double x0 = 0.5*xmax + 0.5;
  double y0 = 0.5*ymax + 0.5;
  double r, x, y;
  double * phi;
  FILE * fp;

  phi = (double *) malloc(xmax*ymax*zmax*sizeof(double));
  if (phi == NULL) {
    printf("malloc(phi) failed\n");
    exit(-1);
  }

  /* Read the data */

  printf("Reading phi data from %s...\n", filename);

  fp = fopen(filename, "r");
  if (fp == NULL) {
    printf("Failed to open %s\n", filename);
    exit(-1);
  }

  for (ic = 0; ic < xmax; ic++) {
    for (jc = 0; jc < ymax; jc++) {
      for (kc = 0; kc < zmax; kc++) {
	index = ic*zmax*ymax + jc*zmax + kc;
	nread = fread(phi + index, sizeof(double), 1, fp);
	assert(nread == 1);
      }
    }
  }

  fclose(fp);

  /* Find the interface for each solid location */

  for (ic = 0; ic < xmax; ic++) {
    x = 1.0 + ic - x0;
    for (jc = 0; jc < ymax; jc++) {
      y = 1.0 + jc - y0;

      r = sqrt(x*x + y*y);

      /* Work out whether solid or fluid */
      inside = 0;
      if (xsection == SQUARE) {
	if (ic > 0 && ic < xmax-1 && jc > 0 && jc < ymax-1) inside = 1;
      }
      if (xsection == CIRCLE) {
	if (r <= rc) inside = 1;
      }

      if (inside) {
	/* Examine the profile */
	double h, dh;
	h = -1.0;

	for (kc = z1; kc <= z2; kc++) {
	  index = ic*zmax*ymax + jc*zmax + kc;
	  if (phi[index] > 0.0 && phi[index+1] < 0.0) {
	    /* Linear interpolation to get surface position */
	    dh = phi[index] / (phi[index] - phi[index+1]);
	    h = 1.0 + kc + dh;
	  }
	}
	printf("%f %f\n", r, h);
      }
    }
  }

  free(phi);

  /* Do not return! */
  exit(0);

  return;
}

/******************************************************************************
 *
 *  map_special_cross
 *
 *  Make a cross shape which looks like a '+' in the x-y plane.
 *
 *  We use the system size from above so
 *   (Lx, Ly, Lz) = (xmax, ymax, zmax).
 *
 *  Along the x-direction we have a periodic channel of width
 *  W, with the centre of the channel at Ly/2. The channel is
 *  'open', ie., periodic at x = 1 and x = Lx.
 *
 *  In the y-direction we have two dead-end arms of length
 *  W_arm in the x-direction. The extent of each arm in the
 *  y-direction is (Ly - W - 2)/2 (one point is solid at
 *  y = 1 and y = Ly to form the dead-end.
 *
 *  The height is the same everywhere in the z-direction
 *  (height = Lz - 2, ie., it is 'closed' at top and bottom).
 *
 *  Example: xmax = 19, ymax = 13, zmax = 9; w = 5, w_arm = 4.
 *  To minimise computational modes, it's a good idea to have
 *  xmax, ymax, zmax, odd numbers.
 *
 *****************************************************************************/

int map_special_cross(char * map_in, int * nsolid) {

  const int w = 5;
  const int w_arm = 4;
  int index, ic, jc, kc;
  int x0, x1, j0, j1;
  int ns;

  assert(map_in);
  assert(nsolid);

  /* The plan.
   * 1. Set all the points to solid
   * 2. Drive a period channel along the x-direction
   * 3. Drive a non-periodic channel across the y-direction */

  printf("Special cross routine\n");
  printf("Lx Ly Lz: %4d %4d %4d\n", xmax, ymax, zmax);
  printf("Channel width: %3d\n", w);
  printf("Arm length:    %3d\n", w_arm);

  assert(ymax % 2);  /* Odd number of points */
  assert(w % 2);     /* Odd number of points */

  /* Set all points to solid */

  for (ic = 1; ic <= xmax; ic++) {
    for (jc = 1; jc <= ymax; jc++) {
      for (kc = 1; kc <= zmax; kc++) {

	/* Memory index */
	index = ymax*zmax*(ic-1) + zmax*(jc-1) + (kc-1);
	map_in[index] = MAP_BOUNDARY;
      }
    }
  }

  /* Centred peridoic channel in x-direction */

  j0 = (ymax+1)/2 - (w-1)/2;
  j1 = (ymax+1)/2 + (w-1)/2;

  for (ic = 1; ic <= xmax; ic++) {
    for (jc = j0; jc <= j1; jc++) {
      for (kc = 2; kc <= zmax-1; kc++) {

	/* Memory index */
	index = ymax*zmax*(ic-1) + zmax*(jc-1) + (kc-1);
	map_in[index] = MAP_FLUID;
      }
    }
  }

  /* The 'arms' of the cross */

  x0 = (xmax - w_arm + 1)/2 + (xmax % 2);
  x1 = x0 + w_arm - 1;

  for (ic = x0; ic <= x1; ic++) {
    for (jc = 2; jc <= ymax-1; jc++) {
      for (kc = 2; kc <= zmax-1; kc++) {

	/* Memory index */
	index = ymax*zmax*(ic-1) + zmax*(jc-1) + (kc-1);
	map_in[index] = MAP_FLUID;
      }
    }
  }

  /* Count solid points for return */

  ns = 0;

  for (ic = 1; ic <= xmax; ic++) {
    for (jc = 1; jc <= ymax; jc++) {
      for (kc = 1; kc <= zmax; kc++) {

	/* Memory index */
	index = ymax*zmax*(ic-1) + zmax*(jc-1) + (kc-1);
	if (map_in[index] == MAP_BOUNDARY) ++ns;
      }
    }
  }

  *nsolid = ns;

  return 0;
}
