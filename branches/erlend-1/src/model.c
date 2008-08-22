/*****************************************************************************
 *
 *  model.c
 *
 *  This encapsulates data/operations related to distributions.
 *  However, the implementation of "Site" is exposed for performance
 *  reasons. For non-performance critical operations, prefer the
 *  access functions.
 *
 *  The LB model is either _D3Q15_ or _D3Q19_, as included in model.h.
 *
 *  $Id: model.c,v 1.9.6.17 2008-08-22 00:43:20 erlend Exp $
 *
 *  Kevin Stratford (kevin@epcc.ed.ac.uk)
 *
 *****************************************************************************/

#include <assert.h>
#include <stdio.h>
#include <stdlib.h>

#include "pe.h"
#include "runtime.h"
#include "coords.h"
#include "model.h"
#include "timer.h"

const double cs2  = (1.0/3.0);
const double rcs2 = 3.0;
const double d_[3][3] = {{1.0, 0.0, 0.0}, {0.0, 1.0, 0.0}, {0.0, 0.0, 1.0}};
const int nhalo_site_ = 1;

Site  * site;

static int nsites_ = 0;
static int xfac_;
static int yfac_;

#ifdef _MPI_
static MPI_Datatype DT_plane_XY;
static MPI_Datatype DT_plane_YZ;
static MPI_Datatype DT_plane_XZ;
static MPI_Datatype DT_plane_XY_right;
static MPI_Datatype DT_plane_XY_left;
static MPI_Datatype DT_plane_XZ_right;
static MPI_Datatype DT_plane_XZ_left;
static MPI_Datatype DT_plane_YZ_right;
static MPI_Datatype DT_plane_YZ_left;
MPI_Datatype DT_Site;
MPI_Datatype DT_Site_xright;
MPI_Datatype DT_Site_xleft;
MPI_Datatype DT_Site_yright;
MPI_Datatype DT_Site_yleft;
MPI_Datatype DT_Site_zright;
MPI_Datatype DT_Site_zleft;
enum mpi_tags {TAG_FWD = 900, TAG_BWD}; 


/* Takes the indices of blocks in cv (given by e.g. xfwd_disp_cv[])
 * and extends this to the whole of Site[i] (i.e. for all distributions).
 *
 * (in)  count:       the number of blocks to send in this dimension,
 *                    and the size of dispArray[] (countcv*ndist+2);
 * (in)  indexDisp[]: the index in cv of a block of data;
 * (out) dispArray[]: the displacement (in bytes) in Site[i] 
 *                    of a block of data;
 */
void getAintDisp(int indexDisp[], MPI_Aint dispArray[], int count) {
  int i;
  int countcv = (count-2)/ndist;
  MPI_Aint site_addresses[count-1];
  /* site_addresses[0] is the base, from which offsets are calculated. */
  MPI_Address(&site[0].f[0], &site_addresses[0]);
  /* Get the addresses of start blocks into site_addresses. */
  for (i = 0; i < countcv; i++) {
    MPI_Address(&site[0].f[indexDisp[i]], &site_addresses[i+1]);
    MPI_Address(&site[0].g[indexDisp[i]], &site_addresses[countcv+1+i]);
  }
  /* Tell MPI_Type_struct about the start and finish of the whole array */
  dispArray[0] = 0;
  dispArray[count-1] = sizeof(site[0]);

  for (i = 1; i < count - 1; i++) {
    dispArray[i] = site_addresses[i] - site_addresses[0];
  }
}

/* Takes the block lengths in cv (given by e.g. xblocklens_cv[])
 * and extends this to the whole of Site[i] (i.e. for all distributions).
 *
 * (in)  count:          the number of blocks to send in this dimension,
 *                       and the size of blocklens[] (countcv*ndist+2);
 * (in)  blocklens_cv[]: the length of each block of data to send in 
 *                       this dimension in cv;
 * (out) blocklens[]:    the length of each block of data to send in 
 *                       this dimension, in Site[i];
 */
void getblocklens(int blocklens_cv[], int blocklens[], int count) {
  int countcv = (count-2)/ndist;
  blocklens[0] = 1; /* MPI_LB */
  blocklens[count-1] = 1; /* MPI_UB */
  int i,j;
  for (i = 0; i < ndist; i++) {
    for (j = 1; j < countcv + 1; j++) {
      blocklens[i*countcv + j] = blocklens_cv[j-1];
    }
  }
}

/* Generates an array of MPI_DOUBLEs for use with MPI_Type_struct.
 * The array is of length count, where count = countcv*ndist + 2,
 * i.e. it extends the count for cv to a count for Site[i]
 *      so that it includes all distributions.  The +2 is for the
 *      lower and upper bounds.
 *
 * (in)  count:  the number of blocks to send in this dimension,
 *               and the size of types[] (countcv*ndist+2);
 * (out) types:  an array of MPI_DOUBLE, but with the first and last
 *               elements always MPI_LB (lower bound) and MPI_UB
 *               (upper bound) respectively.
 */
void gettypes(MPI_Datatype types[], int count) {
  types[0] = MPI_LB;
  types[count-1] = MPI_UB;
  int i;
  for (i = 1; i < count - 1; i++) {
    types[i] = MPI_DOUBLE;
  }
}

/*
 * A convenient method to call all of the above for a particular dimension.
 */
void getDerivedDTParms(int count, MPI_Datatype types[], \
		       int indexDisp_fwd[], int indexDisp_bwd[],	\
		       MPI_Aint dispArray_fwd[], MPI_Aint dispArray_bwd[],\
		       int blocklens_cv[], int blocklens[]) {
  getAintDisp(indexDisp_fwd, dispArray_fwd, count);
  getAintDisp(indexDisp_bwd, dispArray_bwd, count);
  getblocklens(blocklens_cv, blocklens, count);
  gettypes(types, count);
}

#endif

/***************************************************************************
 *
 *  init_site
 *
 *  Allocate memory for the distributions. If MPI2 is used, then
 *  this must use the appropriate utility to accomodate LE planes.
 *
 ***************************************************************************/
 
void init_site() {

  int N[3];
  int nx, ny, nz;

  get_N_local(N);

  nx = N[X] + 2*nhalo_site_;
  ny = N[Y] + 2*nhalo_site_;
  nz = N[Z] + 2*nhalo_site_;
  nsites_ = nx*ny*nz;
  yfac_   = nz;
  xfac_   = nz*ny;

  info("Requesting %d bytes for site data\n", nsites_*sizeof(Site));
  info(" sizeof(site)=%d\n", sizeof(Site));

#ifdef _MPI_2_
 {
   int ifail;

   ifail = MPI_Alloc_mem(nsites_*sizeof(Site), MPI_INFO_NULL, &site);
   if (ifail == MPI_ERR_NO_MEM) fatal("MPI_Alloc_mem(site) failed\n");
 }
#else

  /* Use calloc. */

  site = (Site  *) calloc(nsites_, sizeof(Site));
  if (site == NULL) fatal("calloc(site) failed\n");

#endif

#ifdef _MPI_
  /* Set up the MPI Datatypes used for site, and its corresponding
   * halo messages:
   *
   * in XY plane nx*ny blocks of 1 site with stride nz;
   * in XZ plane nx blocks of nz sites with stride ny*nz;
   * in YZ plane one contiguous block of ny*nz sites.
   *
   * This is only confirmed for nhalo_site_ = 1. */

#ifdef _D3Q15_
  info("Using model: d3q15 \n");
#endif  
#ifdef _D3Q19_
  info("Using model: d3q19 \n");
#endif

  if (use_reduced_halos()) {
    info("Using reduced halos. \n");

    /* Set up the reduced datatypes.
     * First get the parameters for MPI_Type_struct
     */
    MPI_Aint xdisp_fwd[xcount];
    MPI_Aint xdisp_bwd[xcount];
    int xblocklens[xcount];
    MPI_Datatype xtypes[xcount];
    getDerivedDTParms(xcount, xtypes,		  \
		      xdisp_fwd_cv, xdisp_bwd_cv, \
		      xdisp_fwd, xdisp_bwd,	  \
		      xblocklens_cv, xblocklens);

    MPI_Aint ydisp_fwd[ycount];
    MPI_Aint ydisp_bwd[ycount];
    int yblocklens[ycount];
    MPI_Datatype ytypes[ycount];
    getDerivedDTParms(ycount, ytypes,		  \
		      ydisp_fwd_cv, ydisp_bwd_cv, \
		      ydisp_fwd, ydisp_bwd,	  \
		      yblocklens_cv, yblocklens);

    MPI_Aint zdisp_fwd[zcount];
    MPI_Aint zdisp_bwd[zcount];
    int zblocklens[zcount];
    MPI_Datatype ztypes[zcount];
    getDerivedDTParms(zcount, ztypes,		  \
		      zdisp_fwd_cv, zdisp_bwd_cv, \
		      zdisp_fwd, zdisp_bwd,	  \
		      zblocklens_cv, zblocklens);

    /* Create a reduced representation of Site in each direction */
    MPI_Type_struct(xcount, xblocklens, xdisp_fwd, xtypes, &DT_Site_xright);
    MPI_Type_struct(xcount, xblocklens, xdisp_bwd, xtypes, &DT_Site_xleft);
    MPI_Type_struct(ycount, yblocklens, ydisp_fwd, ytypes, &DT_Site_yright);
    MPI_Type_struct(ycount, yblocklens, ydisp_bwd, ytypes, &DT_Site_yleft);
    MPI_Type_struct(zcount, zblocklens, zdisp_fwd, ztypes, &DT_Site_zright);
    MPI_Type_struct(zcount, zblocklens, zdisp_bwd, ztypes, &DT_Site_zleft);
    MPI_Type_commit(&DT_Site_xright);
    MPI_Type_commit(&DT_Site_xleft);
    MPI_Type_commit(&DT_Site_yright);
    MPI_Type_commit(&DT_Site_yleft);
    MPI_Type_commit(&DT_Site_zright);
    MPI_Type_commit(&DT_Site_zleft);

    /* Create the planes of Sites that will be sent to the halos. */
    MPI_Type_vector(nx*ny, 1, nz, DT_Site_zright, &DT_plane_XY_right);
    MPI_Type_commit(&DT_plane_XY_right);
    MPI_Type_vector(nx*ny, 1, nz, DT_Site_zleft, &DT_plane_XY_left);
    MPI_Type_commit(&DT_plane_XY_left);
    MPI_Type_vector(nx, nz, ny*nz, DT_Site_yright, &DT_plane_XZ_right);
    MPI_Type_commit(&DT_plane_XZ_right);
    MPI_Type_vector(nx, nz, ny*nz, DT_Site_yleft, &DT_plane_XZ_left);
    MPI_Type_commit(&DT_plane_XZ_left);

    MPI_Type_contiguous(ny*nz, DT_Site_xright, &DT_plane_YZ_right);
    MPI_Type_commit(&DT_plane_YZ_right);
    MPI_Type_contiguous(ny*nz, DT_Site_xleft, &DT_plane_YZ_left);
    MPI_Type_commit(&DT_plane_YZ_left);

  } else {

    info("Using full halos. \n");
    MPI_Type_contiguous(sizeof(Site), MPI_BYTE, &DT_Site);
    MPI_Type_commit(&DT_Site);

    MPI_Type_contiguous(ny*nz, DT_Site, &DT_plane_YZ);
    MPI_Type_commit(&DT_plane_YZ);
    MPI_Type_hvector(nx, nz, ny*nz*sizeof(Site), DT_Site, &DT_plane_XZ);
    MPI_Type_commit(&DT_plane_XZ);
    MPI_Type_vector(nx*ny, 1, nz, DT_Site, &DT_plane_XY);
    MPI_Type_commit(&DT_plane_XY);
    /* use the reduced datatypes names... */
    DT_plane_YZ_right = DT_plane_YZ;
    DT_plane_YZ_left = DT_plane_YZ;
    DT_plane_XZ_right = DT_plane_XZ;
    DT_plane_XZ_left = DT_plane_XZ;
    DT_plane_XY_right = DT_plane_XY;
    DT_plane_XY_left = DT_plane_XY;

  }

 #endif

  return;
}

/*****************************************************************************
 *
 *  finish_site
 *
 *  Clean up.
 *
 *****************************************************************************/

void finish_site() {

#ifdef _MPI_2_
  MPI_Free_mem(site);
#else
  free(site);
#endif

#ifdef _MPI_
/*
  MPI_Type_free(&DT_Site);
  MPI_Type_free(&DT_plane_XY);
  MPI_Type_free(&DT_plane_XZ);
  MPI_Type_free(&DT_plane_YZ);
*/
  MPI_Type_free(&DT_Site_xright);
#endif

  return;
}

/*****************************************************************************
 *
 *  index_site
 *
 *  Map the Cartesian coordinates (i, j, k) to index in site.
 *
 *****************************************************************************/

int index_site(const int i, const int j, const int k) {

  return (i*xfac_ + j*yfac_ + k);
}

/*****************************************************************************
 *
 *  set_rho
 *
 *  Project rho onto the distribution at position index, assuming zero
 *  velocity and zero stress.
 *
 *****************************************************************************/

void set_rho(const double rho, const int index) {

  int   p;

  assert(index >= 0 || index < nsites_);

  for (p = 0; p < NVEL; p++) {
    site[index].f[p] = wv[p]*rho;
  }

  return;
}

/*****************************************************************************
 *
 *  set_rho_u_at_site
 *
 *  Project rho, u onto distribution at position index, assuming
 *  zero stress.
 *
 *****************************************************************************/

void set_rho_u_at_site(const double rho, const double u[], const int index) {

  int p;
  double udotc;

  assert(index >= 0 || index < nsites_);

  for (p = 0; p < NVEL; p++) {
    udotc = u[X]*cv[p][X] + u[Y]*cv[p][Y] + u[Z]*cv[p][Z];
    site[index].f[p] = wv[p]*(rho + rcs2*udotc);
  }

  return;
}

/*****************************************************************************
 *
 *  set_phi
 *
 *  Sets the order parameter distribution at index address, assuming
 *  zero order parameter flux and zero stress.
 *
 *  Note that this is currently incompatible with the reprojection
 *  at the collision stage where all the phi would go into the rest
 *  distribution.
 *
 ****************************************************************************/

void set_phi(const double phi, const int index) {

  int   p;

  assert(index >= 0 || index < nsites_);

  for (p = 0; p < NVEL; p++) {
    site[index].g[p] = wv[p]*phi;
  }

  return;
}

/*****************************************************************************
 *
 *  set_f_at_site
 *
 *****************************************************************************/

void set_f_at_site(const int index, const int p, const double fp) {

  assert(index >= 0 || index < nsites_);
  site[index].f[p] = fp;

  return;
}

/*****************************************************************************
 *
 *  get_f_at_site
 *
 *****************************************************************************/

double get_f_at_site(const int index, const int p) {

  assert(index >= 0 || index < nsites_);
  return site[index].f[p];
}

/*****************************************************************************
 *
 *  set_g_at_site
 *
 *****************************************************************************/

void set_g_at_site(const int index, const int p, const double gp) {

  assert(index >= 0 || index < nsites_);
  site[index].g[p] = gp;

  return;
}

/*****************************************************************************
 *
 *  get_g_at_site
 *
 *****************************************************************************/

double get_g_at_site(const int index, const int p) {

  assert(index >= 0 || index < nsites_);
  return site[index].g[p];
}

/*****************************************************************************
 *
 *  get_rho_at_site
 *
 *  Return the density at lattice node index.
 *
 *****************************************************************************/

double get_rho_at_site(const int index) {

  double rho;
  double * f;
  int   p;

  assert(index >= 0 || index < nsites_);

  rho = 0.0;
  f = site[index].f;

  for (p = 0; p < NVEL; p++)
    rho += f[p];

  return rho;
}

/****************************************************************************
 *
 *  get_phi_at_site
 *
 *  Return the order parameter at lattice node index.
 *
 ****************************************************************************/

double get_phi_at_site(const int index) {

  double   phi;
  double * g;
  int     p;

  assert(index >= 0 || index < nsites_);

  phi = 0.0;
  g = site[index].g;

  for (p = 0; p < NVEL; p++) {
    phi += g[p];
  }

  return phi;
}

/*****************************************************************************
 *
 *  get_momentum_at_site
 *
 *  Return momentum density at lattice node index.
 *
 *****************************************************************************/

void get_momentum_at_site(const int index, double rhou[ND]) {

  double  * f;
  int       i, p;

  assert(index >= 0 || index < nsites_);

  for (i = 0; i < ND; i++) {
    rhou[i] = 0.0;
  }

  f  = site[index].f;

  for (p = 0; p < NVEL; p++) {
    for (i = 0; i < ND; i++) {
      rhou[i] += f[p]*cv[p][i];
    }
  }

  return;
}

/*****************************************************************************
 *
 *  halo_site
 *
 *  Swap the distributions at the periodic/processor boundaries
 *  in each direction.
 *
 *****************************************************************************/

void halo_site() {

  int i, j, k;
  int xfac, yfac;
  int N[3];

#ifdef _MPI_
  MPI_Request request[4];
  MPI_Status status[4];
#endif

  TIMER_start(TIMER_HALO_LATTICE);

  get_N_local(N);
  yfac =  N[Z] + 2*nhalo_site_;
  xfac = (N[Y] + 2*nhalo_site_)*yfac;

  /* The x-direction (YZ plane) */
  
  if (cart_size(X) == 1) {
    for (j = 1; j <= N[Y]; j++) {
      for (k = 1; k <= N[Z]; k++) {
	site[                j*yfac + k] = site[N[X]*xfac + j*yfac + k];
	site[(N[X]+1)*xfac + j*yfac + k] = site[     xfac + j*yfac + k];
      }
    }
  }
  else {
#ifdef _MPI_   
    MPI_Issend(&site[xfac].f[0], 1, DT_plane_YZ_left, cart_neighb(BACKWARD,X),
	       TAG_BWD, cart_comm(), &request[0]);
    MPI_Irecv(&site[(N[X]+1)*xfac].f[0], 1, DT_plane_YZ_left, cart_neighb(FORWARD,X), 
	       TAG_BWD, cart_comm(), &request[1]);
    MPI_Issend(&site[N[X]*xfac].f[0], 1, DT_plane_YZ_right, cart_neighb(FORWARD,X),
               TAG_FWD, cart_comm(), &request[2]);
    MPI_Irecv(&site[0].f[0], 1, DT_plane_YZ_right, cart_neighb(BACKWARD,X),
               TAG_FWD, cart_comm(), &request[3]);

    MPI_Waitall(4, request, status);
#endif
  }
  
  /* The y-direction (XZ plane) */
  
  if (cart_size(Y) == 1) {
    for (i = 0; i <= N[X]+1; i++) {
      for (k = 1; k <= N[Z]; k++) {
	site[i*xfac                 + k] = site[i*xfac + N[Y]*yfac + k];
	site[i*xfac + (N[Y]+1)*yfac + k] = site[i*xfac +      yfac + k];
      }
    }
  }
  else {
#ifdef _MPI_
    MPI_Issend(&site[yfac].f[0], 1, DT_plane_XZ_left, cart_neighb(BACKWARD,Y),
	       TAG_BWD, cart_comm(), &request[0]);
    MPI_Irecv(&site[(N[Y]+1)*yfac].f[0], 1, DT_plane_XZ_left, cart_neighb(FORWARD,Y),
	      TAG_BWD, cart_comm(), &request[1]);
    MPI_Issend(&site[N[Y]*yfac].f[0], 1, DT_plane_XZ_right, cart_neighb(FORWARD,Y),
	       TAG_FWD, cart_comm(), &request[2]);
    MPI_Irecv(&site[0].f[0], 1, DT_plane_XZ_right, cart_neighb(BACKWARD,Y),
	      TAG_FWD, cart_comm(), &request[3]);
    MPI_Waitall(4, request, status);
#endif
  }
  
  /* Finally, z-direction (XY plane) */
  
  if (cart_size(Z) == 1) {
    for (i = 0; i<= N[X]+1; i++) {
      for (j = 0; j <= N[Y]+1; j++) {
	site[i*xfac + j*yfac           ] = site[i*xfac + j*yfac + N[Z]];
	site[i*xfac + j*yfac + N[Z] + 1] = site[i*xfac + j*yfac + 1   ];
      }
    }
  }
  else {
#ifdef _MPI_
    MPI_Issend(&site[1].f[0], 1, DT_plane_XY_left, cart_neighb(BACKWARD,Z),
	       TAG_BWD, cart_comm(), &request[0]);
    MPI_Irecv(&site[N[Z]+1].f[0], 1, DT_plane_XY_left, cart_neighb(FORWARD,Z),
	      TAG_BWD, cart_comm(), &request[1]);
    MPI_Issend(&site[N[Z]].f[0], 1, DT_plane_XY_right, cart_neighb(FORWARD,Z),
	       TAG_FWD, cart_comm(), &request[2]);  
    MPI_Irecv(&site[0].f[0], 1, DT_plane_XY_right, cart_neighb(BACKWARD,Z),
	      TAG_FWD, cart_comm(), &request[3]);
    MPI_Waitall(4, request, status);
#endif
    }
 
  TIMER_stop(TIMER_HALO_LATTICE);

  return;
}
