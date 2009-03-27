/*****************************************************************************
 *
 *  Lees-Edwards transformations for distributions.
 *
 *  Note that the distributions have displacement u*t
 *  not u*(t-1) returned by le_get_displacement().
 *  This is for reasons of backwards compatability.
 *
 *****************************************************************************/

#include <stdlib.h>
#include <math.h>

#include "pe.h"
#include "timer.h"
#include "coords.h"
#include "control.h"
#include "runtime.h"
#include "model.h"
#include "physics.h"
#include "leesedwards.h"

extern Site * site;
extern MPI_Datatype DT_Site;

static double * LeesEdw_site;
static void LE_update_buffers(void);
static void le_update_parallel(void);

/*****************************************************************************
 *
 * Applies Lees-Edwards: transformation of site contents across LE walls: 
 * and the components of the distribution functions which cross LE walls
 *
 *****************************************************************************/

void LE_apply_LEBC(void) {

  int     plane,xfac,yfac,xfac2,yfac2,zfac2,jj,kk,ind,ind0,ind1;
  int     LE_loc;
  int     i,j;
  double  LE_vel;
  int     N[3];

  double *f, *g;
  double rho, phi, ds[3][3], dsphi[3][3], udotc, jdotc, sdotq, sphidotq;
  double u[3], jphi[3], du[3], djphi[3];
  double t;
  int ia, ib;

  const double r2rcs4 = 4.5;         /* The constant 1 / 2 c_s^4 */

  int p,side;
  int nplane;

  nplane = le_get_nplane_local();
  if (nplane == 0) return;

  TIMER_start(TIMER_LE);

  get_N_local(N);
  yfac  =  N[Z]+2*nhalo_;
  xfac  = (N[Y]+2*nhalo_) * (N[Z]+2*nhalo_);
  zfac2 = LE_N_VEL_XING * 2;  /* final x2 because 2 dist functions (f and g) */
  yfac2 = yfac * zfac2;
  xfac2 = xfac * zfac2; 
  
  /* 
   * Allocate memory for Lees-Edwards buffers ("translated" values) and 
   * "unrolled" phis: see LE_update_buffers() and LE_unroll_phi().
   * Storage required for LeesEdw_site[] is:
   *   2*N_LE_planes   one plane on each side of LE wall
   *   * 2             because there are two distribution functions f and g
   *   * LE_N_VEL_XING simply save components crossing the LE wall 
   *   * sizeof(double) because site components are doubles
   */

  LeesEdw_site = (double *)
    malloc(2*nplane*xfac*2*LE_N_VEL_XING*sizeof(double));
  
  if(LeesEdw_site==NULL) {
    fatal("LE_Init(): failed to allocate %d bytes for LE buffers\n",
	  2*(2*LE_N_VEL_XING+1)*nplane*xfac*sizeof(double));
  }

  t = 1.0*get_step();

  /* Stage 2: use Ronojoy's scheme to update fs and gs */

  for (plane = 0; plane < nplane; plane++) {
    for (side = 0; side < 2; side++) {

      /* Start with plane below Lees-Edwards BC */

      if (side == 0) {
	LE_vel =-le_plane_uy(t);
	LE_loc = le_plane_location(plane);
      }
      else {
	/* Finally, deal with plane above LEBC */
	LE_vel =+le_plane_uy(t);
	LE_loc = le_plane_location(plane) + 1;
      }

      /* First, for plane `below' LE plane, ie, crossing LE plane going up */

      for (jj = 1; jj <= N[Y]; jj++) {
	for (kk = 1; kk <= N[Z]; kk++) {
	  
	  /* ind = LE_loc*xfac + jj*yfac + kk;*/
	  ind = get_site_index(LE_loc, jj, kk);
	    	  
	  /* For fi: M0 = rho; M1 = rho.u[]; M2 = s[][] */
	  /* Corrected expressions for Lees-Edwards are as follows: */
	  /* M0   -> M0 */
	  /* M1i  -> M1i + u_LE_i*M0 */
	  /* M2ij -> M2ij + u_LE_i*M1j + u_LE_j*M1i + u_LE_i.u_LE_j*M0 */
	  /* (where u_LE[X]=u_LE[Z]=0; u_LE[Y] = plane speed uy) */
	  
	  f = site[ind].f;
	  g = site[ind].g;

	  /* Compute 0th and 1st moments */

	  rho = f[0];
	  phi = g[0];

	  for (ia = 0; ia < 3; ia++) {
	    u[ia] = 0.0;
	    jphi[ia] = 0.0;
	    du[ia] = 0.0;
	    djphi[ia] = 0.0;
	  }

	  for (p = 1; p < NVEL; p++) {
	    rho    += f[p];
	    phi    += g[p];
	    for (ia = 0; ia < 3; ia++) {
	      u[ia] += f[p]*cv[p][ia];
	      jphi[ia] += g[p]*cv[p][ia];
	    }
	  }

	  /* ... then for the momentum (note that whilst jphi[] represents */
	  /* the moment, u only represents the momentum) */

	  du[Y] = LE_vel; 
	  djphi[Y] = phi*LE_vel;
 
	  /* Include correction for Lees-Edwards BC: first for the stress */
	  /* NOTE: use the original u.y and jphi[1], i.e., befoer LE fix */

	  for (ia = 0; ia < 3; ia++) {
	    for (ib = 0; ib < 3; ib++) {
	      ds[ia][ib] = rho*(u[ia]*du[ib] + du[ia]*u[ib] + du[ia]*du[ib]);
	      dsphi[ia][ib] =
		du[ia]*jphi[ib] + jphi[ia]*du[ib] + phi*du[ia]*du[ib];
	    }
	  }

	  /* Now update the distribution */
	  for (p = 0; p < NVEL; p++) {
	      
	    udotc =    du[Y] * cv[p][1];
	    jdotc = djphi[Y] * cv[p][1];
	      
	    sdotq    = 0.0;
	    sphidotq = 0.0;
	      
	    for (i = 0; i < 3; i++) {
	      for (j = 0; j < 3; j++) {
		sdotq += ds[i][j]*q_[p][i][j];
		sphidotq += dsphi[i][j]*q_[p][i][j];
	      }
	    }
	      
	    /* Project all this back to the distributions. */
	    /* First case: the plane below Lees-Edwards BC */

	    if (side == 0) {

	      /* For each velocity intersecting the LE plane (up, ie X+1) */
	      if (cv[p][X] == 1) {
		f[p] += wv[p]*(rho*udotc*rcs2 + sdotq*r2rcs4);
		g[p] += wv[p]*(    jdotc*rcs2 + sphidotq*r2rcs4);
	      }
	    }
	    /* Now consider the case when above the LE plane */
	    else {

	      /* For each velocity intersecting the LE plane (down, ie X-1) */
	      if (cv[p][X] == -1) {
		f[p] += wv[p]*(rho*udotc*rcs2 + sdotq*r2rcs4);
		g[p] += wv[p]*(    jdotc*rcs2 + sphidotq*r2rcs4);
	      }
	    }
	  }
	}
      }
    }
  }
  
  /* Stage 3: update buffers (pre-requisite to translation) */

  halo_site();
  LE_update_buffers();

  /* Stage 4: apply translation on fs and gs crossing LE planes */

  for (plane = 0; plane < nplane; plane++) {
    LE_loc  = le_plane_location(plane);
      
    /* First, for plane 'below' LE plane: displacement = +displ */
    for (jj = 1; jj <= N[Y]; jj++) {
      for (kk = 1; kk <= N[Z]; kk++) {
	/* ind  =  LE_loc*xfac  + jj*yfac  + kk;*/
	ind = get_site_index(LE_loc, jj, kk);
	ind0 = 2*plane*xfac2 + jj*yfac2 + kk*zfac2;
	    
	/* For each velocity intersecting the LE plane (up, ie X+1) */
	ind1 = 0; 
	for (p = 0; p < NVEL; p++) {	      
	  if (cv[p][X] == 1) {
	    site[ind].f[p] = LeesEdw_site[ind0+2*ind1];
	    site[ind].g[p] = LeesEdw_site[ind0+2*ind1+1];
	    ind1++;
	  }
	}
      }
    }
      
    /* Then, for plane 'above' LE plane, ie, crossing LE plane going down */

    for (jj = 1; jj <= N[Y]; jj++) {
      for (kk = 1; kk <= N[Z]; kk++) {
	/* ind  =  (LE_loc+1)*xfac  + jj*yfac  + kk;*/
	ind = get_site_index(LE_loc+1, jj, kk);
	ind0 = (2*plane+1)*xfac2 + jj*yfac2 + kk*zfac2;
	
	/* For each velocity intersecting the LE plane (down, ie X-1) */

	ind1 = 0;

	for (p = 0; p < NVEL; p++) {
	  if(cv[p][X] == -1) {
	    site[ind].f[p] = LeesEdw_site[ind0+2*ind1];
	    site[ind].g[p] = LeesEdw_site[ind0+2*ind1+1];
	    ind1++;
	  }
	}
	
      }
    }
    /* Next side of plane */
  }

  free(LeesEdw_site);

  TIMER_stop(TIMER_LE);

  return;
}

/*****************************************************************************
 *
 *  LE_update_buffers
 *
 *****************************************************************************/

void LE_update_buffers() {

  int    nlocal[3];
  int    nplane, n;
  int    jj, kk, ind, ind0, ind1, ind2, xfac, yfac, xfac2, yfac2, zfac2;
  int    integ, LE_loc, p;
  int    disp_j1,disp_j2;
  double displ, LE_frac;
  double t;

  nplane = le_get_nplane_local();
  if (nplane == 0) return;

  if (cart_size(Y) > 1) {
    le_update_parallel();
  }
  else {

    get_N_local(nlocal);
    nplane = le_get_nplane_local();

    t = 1.0*get_step();

    yfac  =  nlocal[Z]+2*nhalo_;
    xfac  = (nlocal[Y]+2*nhalo_) * (nlocal[Z]+2*nhalo_);
    zfac2 = LE_N_VEL_XING * 2;  /* final x2 because fs and gs as well! */
    yfac2 = yfac * zfac2;
    xfac2 = xfac * zfac2;

    for (n = 0; n < nplane; n++) {
 
      LE_loc  = le_plane_location(n);

      displ = fmod(le_buffer_displacement(nhalo_, t), L(Y));
      integ = floor(displ);
      LE_frac = 1.0 - (displ - integ);

      /* Plane below (going down): +ve displacement */
      /* site_buff[i] = frac*site[i+integ] + (1-frac)*site[i+(integ+1)] */

      for (jj = 1; jj <= nlocal[Y]; jj++) {

	disp_j1 = ((jj+integ+2*nlocal[Y]-1) % nlocal[Y]) + 1;
	disp_j2 = (disp_j1 % nlocal[Y]) + 1;

	for (kk = 1; kk <= nlocal[Z]; kk++) {
	  ind  = 2*n*xfac2 +      jj*yfac2 + kk*zfac2;
	  ind0 = LE_loc *xfac  + disp_j1*yfac  + kk;
	  ind1 = LE_loc *xfac  + disp_j2*yfac  + kk;

	  ind0 = ADDR(LE_loc, disp_j1, kk);
	  ind1 = ADDR(LE_loc, disp_j2, kk);
		  
	  /* For each velocity intersecting the LE plane (up, ie X+1) */
	  ind2 = 0; 

	  for (p = 0; p < NVEL; p++) {
	    if (cv[p][X] == 1) {
	      LeesEdw_site[ind+2*ind2] = 
		LE_frac*site[ind0].f[p] + (1.0-LE_frac)*site[ind1].f[p];
	      LeesEdw_site[ind+2*ind2+1] = 
		LE_frac*site[ind0].g[p] + (1.0-LE_frac)*site[ind1].g[p];
	      ind2++;
	    }
	  }
	}
      }
	  
      /* Plane above: -ve displacement */
      /* site[i] = frac*site[i-integ] + (1-frac)*site[i-(integ+1)] */
      /* buff[i] = site[i-(integ+1)] */

      for (jj = 1; jj <= nlocal[Y]; jj++) {

	disp_j1 = ((jj-integ+2*nlocal[Y]-2) % nlocal[Y]) + 1;
	disp_j2 = ((disp_j1+nlocal[Y]) % nlocal[Y]) + 1;

	for (kk = 1; kk <= nlocal[Z]; kk++) {
	  ind = (2*n+1)*xfac2 +      jj*yfac2 + kk*zfac2;
	  ind0 = (LE_loc+1)*xfac  + disp_j1*yfac  + kk;
	  ind1 = (LE_loc+1)*xfac  + disp_j2*yfac  + kk;

	  ind0 = ADDR(LE_loc+1, disp_j1, kk);
	  ind1 = ADDR(LE_loc+1, disp_j2, kk);
		  
	  /* For each velocity intersecting the LE plane (up, ie X-1) */
	  ind2 = 0;
	  for (p = 0; p < NVEL; p++) {
	    if (cv[p][X] == -1) {
	      LeesEdw_site[ind+2*ind2] = 
		LE_frac*site[ind1].f[p] + (1.0-LE_frac)*site[ind0].f[p];
	      LeesEdw_site[ind+2*ind2+1] = 
		LE_frac*site[ind1].g[p] + (1.0-LE_frac)*site[ind0].g[p];
	      ind2++;
	    }
	  }
	}
      }
      /* Next plane */
    }
  }

  return;
}

/*****************************************************************************
 *
 *  le_update_parallel
 *
 *****************************************************************************/

static void le_update_parallel() {

  int nsites, nsites1, nsites2;
  int n, jj, kk, start_y;
  int p;
  int nlocal[3];
  int offset[3];
  int nplane;
  int ind, ind0, ind1, ind2;
  int xfac, yfac, xfac2, yfac2, zfac2;
  int LE_loc, integ;
  double LE_frac, displ;
  double t;

  int  send[2], recv[2];

  const int tag1 = 3102;
  const int tag2 = 3103;
  MPI_Request req[4];
  MPI_Status status[4];
  MPI_Comm comm = le_communicator();

  Site * buff_site;
  
  get_N_local(nlocal);
  get_N_offset(offset);
  nplane = le_get_nplane_local();

  t = 1.0*get_step();

  yfac = nlocal[Z]+2*nhalo_;
  xfac = (nlocal[Y]+2*nhalo_)*(nlocal[Z]+2*nhalo_);
  nsites = (nlocal[Y]+2*nhalo_+1)*(nlocal[Z]+2*nhalo_);

  /* 
   * Set up buffer of translated sites (by linear interpolation):
   * 1. Copy (nlocal[Y]+3)*(nlocal[Z]+2) translated sites in buff_site[]
   * 2. Perform linear interpolation and copy to LeesEdw_site[]
   */

  /* Strides for LeesEdw_site[] */

  zfac2 = LE_N_VEL_XING * 2;  /* x2 because 2 distrib funcs (f and g) */
  yfac2 = yfac * zfac2;
  xfac2 = xfac * zfac2; 

  /* Allocate memory for buffering sites */

  buff_site = (Site *) malloc(2*nsites*sizeof(Site));
  if (buff_site == NULL) fatal("mallox(buff_site) failed\n");

  for (n = 0; n < nplane; n++) {

    LE_loc  = le_plane_location(n);

    displ = fmod(le_buffer_displacement(nhalo_, t), L(Y));
    integ = floor(displ);
    LE_frac = 1.0 - (displ - integ);

    /*
     * Plane below (going down): +ve displacement:
     * LeesEdw_site[i] =
     *            frac*buff_site[i+integ] + (1-frac)*buff_site[i+integ+1]
     */

    /* Starting y coordinate (global address): range 1->N_total.y */
    start_y = ((offset[Y]+(1-nhalo_)+integ+2*N_total(Y)-1) % N_total(Y)) + 1;
      
    /* Starting y coordinate (now local address on PE target_rank1) */
    /* Valid values for start_y are in the range 0->nlocal[Y]-1 */
    /* Obviously remainder starts at y=1 on PE target_rank2 */

    start_y = start_y % nlocal[Y];
      
    /* Number of sites to fetch from target_rank1 and target_rank2 */
    /* Note that nsites = nsites1+nsites2 = (nlocal[Y]+3)*(nlocal[Z]+2) */
    nsites1 = (nlocal[Y]-start_y+1)*(nlocal[Z]+2*nhalo_);
    nsites2 =     (start_y+1+1)*(nlocal[Z]+2*nhalo_);
      
    /* Use point-to-point communication */

    le_displacement_ranks(-displ, recv, send);

    MPI_Irecv(&buff_site[0].f[0], nsites1, DT_Site, recv[0],
	      tag1, comm, &req[0]);
    MPI_Irecv(&buff_site[nsites1].f[0], nsites2, DT_Site, recv[1],
	      tag2, comm, &req[1]);

    MPI_Issend(&site[ADDR(LE_loc,start_y,1-nhalo_)].f[0], nsites1, DT_Site,
	       send[0], tag1, comm, &req[2]);
    MPI_Issend(&site[ADDR(LE_loc,1,1-nhalo_)].f[0], nsites2, DT_Site,
	       send[1], tag2, comm, &req[3]);
    MPI_Waitall(4,req,status);

    /* Plane above (going up): -ve displacement */
    /* buff[i] = (1-frac)*phi[i-(integ+1)] + frac*phi[i-integ] */
      
    /* Starting y coordinate (global address): range 1->N_total.y */
    start_y = ((offset[Y]+(1-nhalo_)-integ+2*N_total(Y)-2) % N_total(Y)) + 1;
  
    /* Starting y coordinate (now local address on PE target_rank1) */
    /* Valid values for start_y are in the range 0->nlocal[Y]-1 */
    /* Obviously remainder starts at y=1 on PE target_rank2 */

    start_y = start_y % nlocal[Y];
	  
    /* Number of sites to fetch from target_rank1 and target_rank2 */
    /* Note that nsites = nsites1+nsites2 = (nlocal[Y]+3)*(nlocal[Z]+2) */

    nsites1 = (nlocal[Y]-start_y+1)*(nlocal[Z]+2*nhalo_);
    nsites2 =     (start_y+1+1)*(nlocal[Z]+2*nhalo_);
	  
    /* Use point-to-point communication */

    le_displacement_ranks(+displ, recv, send);

    MPI_Irecv(&buff_site[nsites].f[0], nsites1, DT_Site, recv[0],
	      tag1, comm, &req[0]);
    MPI_Irecv(&buff_site[nsites+nsites1].f[0], nsites2, DT_Site,
	      recv[1], tag2, comm, &req[1]);
    MPI_Issend(&site[ADDR(LE_loc+1,start_y,1-nhalo_)].f[0], nsites1,
	       DT_Site, send[0], tag1, comm, &req[2]);
    MPI_Issend(&site[ADDR(LE_loc+1,1,1-nhalo_)].f[0], nsites2, DT_Site,
	       send[1], tag2, comm, &req[3]);
    MPI_Waitall(4,req,status);

    /* 
     * Perform linear interpolation on buffer of sites:
     * Note that although all the the 30 components of each site has been
     * copied in site_buff[], LeesEdw_buff[] will only store the
     * components crossing the LE walls: favour low latency from 
     * transferring a single large block rather than a complicated data
     * structure vs. increased required bandwidth as only a third of the
     * data transferred will be used (to be investigated further!)
     */

    /* Plane below */

    for (jj = 1; jj <= nlocal[Y]; jj++) {
      for (kk = 1; kk <= nlocal[Z]; kk++) {

	ind = 2*n*xfac2 + jj*yfac2 + kk*zfac2;
	ind0 =            (jj+nhalo_-1)*yfac  + kk + nhalo_-1;
	ind1 =        (jj+nhalo_-1+1)*yfac  + kk + nhalo_-1;
	ind2 = 0; 

	for (p = 0; p < NVEL; p++) {
	  if (cv[p][X] == 1) {
	    LeesEdw_site[ind+2*ind2] =
	      LE_frac       * buff_site[ind0].f[p] + 
	      (1.0-LE_frac) * buff_site[ind1].f[p];
	    LeesEdw_site[ind+2*ind2+1] =
	      LE_frac       * buff_site[ind0].g[p] + 
	      (1.0-LE_frac) * buff_site[ind1].g[p];
	    ind2++;
	  }
	}
      }
    }

    /* Plane above */
    for (jj = 1; jj <= nlocal[Y]; jj++) {
      for (kk = 1; kk <= nlocal[Z]; kk++) {
	ind = (2*n+1)*xfac2 +  jj   *yfac2 + kk*zfac2;
	ind0 =       nsites +  (jj+nhalo_-1)   *yfac  + kk+nhalo_-1;
	ind1 =       nsites + (jj+nhalo_-1+1)*yfac  + kk+nhalo_-1;
	ind2 = 0;

	for (p = 0; p < NVEL; p++) {
	  if (cv[p][X] == -1) {
	    LeesEdw_site[ind+2*ind2] =
	      LE_frac       * buff_site[ind1].f[p] + 
	      (1.0-LE_frac) * buff_site[ind0].f[p];
	    LeesEdw_site[ind+2*ind2+1] =
	      LE_frac       * buff_site[ind1].g[p] + 
	      (1.0-LE_frac) * buff_site[ind0].g[p];
	    ind2++;
	  }
	}
      }
    }
  }

  free(buff_site);

  return;
}


/*****************************************************************************
 *
 *  le_init_shear_profile
 *
 *  Initialise the distributions to be consistent with a steady-state
 *  linear shear profile, consistent with plane velocity.
 *
 *****************************************************************************/

void le_init_shear_profile() {

  int ic, jc, kc, index;
  int i, j, n, p;
  int N[3];
  double rho, u[ND], gradu[ND][ND];
  double eta;

  /* Only allow initialisation if the flag is set */

  n = 0;
  RUN_get_int_parameter("LE_init_profile", &n);

  if (n != 1) {
    /* do nothing */
  }
  else {
    info("Initialising shear profile\n");

    /* Initialise the density, velocity, gradu; ghost modes are zero */

    rho = get_rho0();
    eta = get_eta_shear();
    get_N_local(N);

    for (i = 0; i< ND; i++) {
      u[i] = 0.0;
      for (j = 0; j < ND; j++) {
	gradu[i][j] = 0.0;
      }
    }

    gradu[X][Y] = le_shear_rate();

    /* Loop trough the sites */

    for (ic = 1; ic <= N[X]; ic++) {
      
      u[Y] = le_get_steady_uy(ic);

      /* We can now project the physical quantities to the distribution */

      for (jc = 1; jc <= N[Y]; jc++) {
	for (kc = 1; kc <= N[Z]; kc++) {

	  index = get_site_index(ic, jc, kc);

	  for (p = 0; p < NVEL; p++) {
	    double f = 0.0;
	    double cdotu = 0.0;
	    double sdotq = 0.0;

	    for (i = 0; i < ND; i++) {
	      cdotu += cv[p][i]*u[i];
	      for (j = 0; j < ND; j++) {
		sdotq += (rho*u[i]*u[j] - eta*gradu[i][j])*q_[p][i][j];
	      }
	    }
	    f = wv[p]*(rho + rcs2*rho*cdotu + 0.5*rcs2*rcs2*sdotq);
	    set_f_at_site(index, p, f);
	  }
	  /* Next site */
	}
      }
    }
  }

  return;
}
