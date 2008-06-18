
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <assert.h>

#include "pe.h"
#include "runtime.h"
#include "timer.h"
#include "coords.h"
#include "control.h"
#include "model.h"
#include "physics.h"

#include "utilities.h"
#include "lattice.h"
#include "collision.h"
#include "leesedwards.h"

extern Site * site;

static MPI_Comm     LeesEdw_Comm;/* Communicator for Lees-Edwards (LE) RMA */
static int        * LE_ranks;
extern MPI_Datatype DT_Site;

static double      *LeesEdw_site;
static LE_Plane   *LeesEdw_plane;

static int        N_LE_plane;
static int        N_LE_total = 0;
static LE_Plane * LeesEdw = NULL;

static void   LE_init_original(void);
static void   le_init_tables(void);
static void   le_init_shear_profile(void);
static double le_get_steady_uy(const int); 


/* KS TODO: replace global LE_plane structure with... */
/* Global Lees Edwards plane parameters */

static struct le_global_parameters {
  int    n_plane;     /* Total number of planes */
  double uy_plane;    /* u[Y] for all planes */
  double shear_rate;  /* Overall shear rate */
  double dx_min;      /* Position first plane */
  double dx_sep;      /* Plane separation */
  /* Here for the time being, actually local. */
  int    nxbuffer;
  int *  index_buffer_to_real;
  int    index_real_nbuffer;
  int *  index_real_to_buffer;
  double * buffer_duy;
} le_params_;

static int initialised_ = 0;

/*****************************************************************************
 *
 *  LE_init
 *
 *  We assume there are a given number of equally-spaced planes
 *  all with the same speed.
 *
 *  Pending (KS):
 *   - do really need two LE_Plane lists?
 *   - look at displacement issue
 *   - add runtime checks
 *
 *****************************************************************************/

void LE_init() {

  int n;

  n = RUN_get_int_parameter("N_LE_plane", &N_LE_total);
  n = RUN_get_double_parameter("LE_plane_vel", &le_params_.uy_plane);

  /* Initialise the global Lees-Edwards structure. */

  if (N_LE_total != 0) {


    LeesEdw = (LE_Plane *) malloc(N_LE_total*sizeof(LE_Plane));
    if (LeesEdw == NULL) fatal("malloc(LE_planes failed\n");

    info("\nLees-Edwards boundary conditions are active:\n");

    le_params_.dx_sep = L(X) / N_LE_total;
    le_params_.dx_min = 0.5*le_params_.dx_sep;

    for (n = 0; n < N_LE_total; n++) {
      LeesEdw[n].loc = le_params_.dx_min + n*le_params_.dx_sep;
      LeesEdw[n].vel = le_params_.uy_plane;
      LeesEdw[n].disp = 0.0;
      LeesEdw[n].frac = 0.0;
      LeesEdw[n].rank = n;
      info("LE plane %d is at x = %d with speed %f\n", n+1, LeesEdw[n].loc,
	   LeesEdw[n].vel);
    }

    le_params_.shear_rate = le_params_.uy_plane*N_LE_total/L(X);
    info("Overall shear rate = %f\n", le_params_.shear_rate);
  }

  LE_init_original();

  /* Only allow initialisation at t = 0 */

  if (get_step() == 0) {

    RUN_get_int_parameter("LE_init_profile", &n);

    if (n == 1) {
      info("Initialising shear profile\n");
      le_init_shear_profile();
    }
  }

  le_init_tables();

  initialised_ = 1;

  return;
}

/*****************************************************************************
 *
 *  le_init_tables
 *
 *  Initialise the buffer look up tables.
 *
 *****************************************************************************/

static void le_init_tables() {

  int ib, ic, ip, n, nb, nh, np;
  int nlocal[3];

  get_N_local(nlocal);

  /* Look up table for buffer -> real index */

  /* For each 'x' location in the buffer region, work out the corresponding
   * x index in the real system:
   *   - for each boundary there are 2*nhalo_ buffer planes
   *   - the locations extend nhalo_ points either side of the boundary.
   */

  n = 2*nhalo_*N_LE_plane;
  le_params_.nxbuffer = n;
  le_params_.index_buffer_to_real = (int *) malloc(n*sizeof(int));
  if (le_params_.index_buffer_to_real == NULL) fatal("malloc(le) failed\n");

  ib = 0;
  for (n = 0; n < N_LE_plane; n++) {
    ic = LeesEdw_plane[n].loc - (nhalo_ - 1);
    for (nh = 0; nh < 2*nhalo_; nh++) {
      assert(ib < 2*nhalo_*N_LE_plane);
      le_params_.index_buffer_to_real[ib] = ic + nh;
      ib++;
    }
  }

  /* Look up table for real -> buffer index */

  /* For each x location in the real system, work out the index of
   * the appropriate x-location in the buffer region. This is more
   * complex, as it depends on whether you are looking across an
   * LE boundary, and if so, in which direction.
   * ie., we need a look up table = function(x, +/- dx).
   * Note that this table exists when no planes are present, ie.,
   * there is no transformation, ie., f(x, dx) = x + dx for all dx.
   */

  n = (nlocal[X] + 2*nhalo_)*(2*nhalo_ + 1);
  le_params_.index_real_nbuffer = n;
  le_params_.index_real_to_buffer = (int *) malloc(n*sizeof(int));
  if (le_params_.index_real_to_buffer == NULL) fatal("malloc(le) failed\n");

  /* Set table in abscence of planes. */
  /* Note the elements of the table at the extreme edges of the local
   * system point outside the system. Accesses must take care. */

   for (ic = 1 - nhalo_; ic <= nlocal[X] + nhalo_; ic++) {
     for (nh = -nhalo_; nh <= nhalo_; nh++) {
       n = (ic + nhalo_ - 1)*(2*nhalo_+1) + (nh + nhalo_);
       assert(n >= 0 && n < (nlocal[X] + 2*nhalo_)*(2*nhalo_ + 1));
       le_params_.index_real_to_buffer[n] = ic + nh;
     }
   }

   /* For each position in the buffer, add appropriate
    * corrections in the table. */

   nb = nlocal[X] + nhalo_ + 1;

   for (ib = 0; ib < le_params_.nxbuffer; ib++) {
     np = ib / (2*nhalo_);
     ip = LeesEdw_plane[np].loc;

     /* This bit of logical chooses the first nhalo_ points of the
      * buffer region for each plane as the 'downward' looking part */

     if ((ib - np*2*nhalo_) < nhalo_) {

       /* Looking across the plane in the -ve x-direction */

       for (ic = ip + 1; ic <= ip + nhalo_; ic++) {
	 for (nh = -nhalo_; nh <= -1; nh++) {
	   if (ic + nh == le_params_.index_buffer_to_real[ib]) {
	     n = (ic + nhalo_ - 1)*(2*nhalo_+1) + (nh + nhalo_);
	     assert(n >= 0 && n < (nlocal[X] + 2*nhalo_)*(2*nhalo_ + 1));
	     le_params_.index_real_to_buffer[n] = nb+ib;
	   }
	 }
       }
     }
     else {
       /* looking across the plane in the +ve x-direction */

       for (ic = ip - (nhalo_ - 1); ic <= ip; ic++) {
	 for (nh = 1; nh <= nhalo_; nh++) {
	   if (ic + nh == le_params_.index_buffer_to_real[ib]) {
	     n = (ic + nhalo_ - 1)*(2*nhalo_+1) + (nh + nhalo_);
	     assert(n >= 0 && n < (nlocal[X] + 2*nhalo_)*(2*nhalo_ + 1));
	     le_params_.index_real_to_buffer[n] = nb+ib;	   
	   }
	 }
       }
     }
     /* Next buffer point */
   }

   
   /* Buffer velocity jumps. When looking from the real system across
    * a boundary into a given buffer, what is the associated velocity
    * jump? The boundary velocities are constant in time. */

   n = le_params_.nxbuffer;
   le_params_.buffer_duy = (double *) malloc(n*sizeof(double));
   if (le_params_.buffer_duy == NULL) fatal("malloc(buffer_duy) failed\n");

  ib = 0;
  for (n = 0; n < N_LE_plane; n++) {
    for (nh = 0; nh < nhalo_; nh++) {
      assert(ib < le_params_.nxbuffer);
      le_params_.buffer_duy[ib] = -LeesEdw_plane[n].vel;
      ib++;
    }
    for (nh = 0; nh < nhalo_; nh++) {
      assert(ib < le_params_.nxbuffer);
      le_params_.buffer_duy[ib] = +LeesEdw_plane[n].vel;
      ib++;
    }
  }

  return;
}

/*----------------------------------------------------------------------------*/
/*!
 * Applies Lees-Edwards: transformation of site contents across LE walls: 
 * and the components of the distribution functions which cross LE walls
 *
 *- \c Options:   _TRACE_
 *- \c Arguments: void
 *- \c Returns:   void
 *- \c Buffers:   no dependence
 *- \c Version:   2.0
 *- \c Last \c updated: 04/01/2003 by JCD
 *- \c Authors:   JC Desplat
 *- \c See \c also: LE_update_buffers()
 *                  LE_init(), MODEL_get_gradients()
 *- \c Note:      this routine is compatible with the serial and MPI
 *                implementations. 
 */
/*----------------------------------------------------------------------------*/

void LE_apply_LEBC( void )
{
  int     plane,xfac,yfac,xfac2,yfac2,zfac2,jj,kk,ind,ind0,ind1,integ;
  int     LE_loc;
  int     i,j;
  double   displ,LE_vel,LE_frac;
  int     N[3];

  double *f, *g;
  double rho, phi, ds[3][3], dsphi[3][3], udotc, jdotc, sdotq, sphidotq;
  double u[3], jphi[3], du[3], djphi[3];
  int ia, ib;

  const double r2rcs4 = 4.5;         /* The constant 1 / 2 c_s^4 */

  int p,side;

  if (N_LE_plane == 0) {
    VERBOSE(("LE_apply_LEBC(): no walls present\n"));
    return;
  }

  TIMER_start(TIMER_LE);

  get_N_local(N);
  yfac  =  N[Z]+2*nhalo_;
  xfac  = (N[Y]+2*nhalo_) * (N[Z]+2*nhalo_);
  zfac2 = LE_N_VEL_XING * 2;  /* final x2 because 2 dist functions (f and g) */
  yfac2 = yfac * zfac2;
  xfac2 = xfac * zfac2; 

  /* Stage 1: update LE displacement and weights for linear interpolation */

  for (plane = 0; plane < N_LE_plane; plane++) {
    /* 
     * Note that this summation is prone to numerical round-offs. However, we
     * cannot use disp = disp0 + step*vel as this would preclude the use of
     * non-constant LE velocities (not currently supported by might be in a
     * future release)
     */
    LeesEdw_plane[plane].disp += LeesEdw_plane[plane].vel;

    /* No. The plane velocities are constant... */
    LeesEdw_plane[plane].disp = LeesEdw_plane[plane].vel*get_step();

    /* -gbl.N_total.y < displ < gbl.N_total.y */
    displ = fmod(LeesEdw_plane[plane].disp,1.0*N_total(Y));
    /* -gbl.N_total.y <= integ <= gbl.N_total.y-1 */
    integ = (int)floor(displ); 
    /* 0 <= displ < 1 */
    displ -= integ;
    /* 0 < LE_frac <= 1 */
    LE_frac = 1.0-displ;
    LeesEdw_plane[plane].frac = LE_frac;
  }


  /* Stage 2: use Ronojoy's scheme to update fs and gs */

  for (plane = 0; plane < N_LE_plane; plane++) {
    for (side = 0; side < 2; side++) {

      /* Start with plane below Lees-Edwards BC */

      if (side == 0) {
	LE_vel =-LeesEdw_plane[plane].vel;
	LE_loc = LeesEdw_plane[plane].loc;
      }
      else {       /* Finally, deal with plane above LEBC */
	LE_vel =+LeesEdw_plane[plane].vel;
	LE_loc = LeesEdw_plane[plane].loc + 1;
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
	  /* (where u_LE[X]=u_LE[Z]=0; u_LE[Y]=LeesEdw_plane[plane].vel) */
	  
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

  LE_update_buffers(SITE_AND_PHI);

  /* Stage 4: apply translation on fs and gs crossing LE planes */

  for (plane = 0; plane < N_LE_plane; plane++) {
    LE_loc  = LeesEdw_plane[plane].loc;
      
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

  TIMER_stop(TIMER_LE);

  return;
}

/*----------------------------------------------------------------------------*/
/*!
 * Initialise Lees-Edwards boundary conditions
 *
 *- \c Options:   _TRACE_, _MPI_
 *- \c Arguments: void
 *- \c Returns:   void
 *- \c Buffers:   no dependence
 *- \c Version:   2.0b1
 *- \c Last \c updated: 03/03/2002 by JCD
 *- \c Authors:   JC Desplat
 *- \c See \c also: MODEL_init(), LE_update_buffers(), LE_apply_LEBC() 
 */
/*----------------------------------------------------------------------------*/

void LE_init_original( void )
{
  int     plane, integ, i, ny2z2;
  double   displ, frac;
  int     N[3];
    
#ifdef _MPI_ /* Parallel (MPI) implementation */

  int     gblflag,  LE_rank, *intbuff, ny3z2;
  int     offset[3];
  int     gr_rank;
  int     flag;

  get_N_local(N);
  get_N_offset(offset);
  gr_rank = cart_rank();

  ny2z2 = (N[Y]+2*nhalo_)*(N[Z]+2*nhalo_);
  ny3z2 = (N[Y]+2*nhalo_+1)*(N[Z]+2*nhalo_);

  /* Set up Lees-Edwards specific MPI datatypes */
  
  /* 
   * Build list of local LE planes from global list: unset flag (set to FALSE)
   * if a plane is located at the edge of the local box (i.e., X==0 or X==N[X]):
   * not allowed!
   */
  flag = 1;

  N_LE_plane = 0;
  LeesEdw_plane = (LE_Plane *)NULL;

  for(i=0; i < N_LE_total; i++) {
       
    /* 
     * Ignore the LE walls placed outside of the local box (but include those
     * placed at the PE boundary even if not allowed: abort later) 
     */
 
    if ((offset[X] <= LeesEdw[i].loc) && (LeesEdw[i].loc <= (offset[X]+N[X])))
      {
	N_LE_plane++;
	if((LeesEdw_plane = (LE_Plane *)
	    realloc(LeesEdw_plane,N_LE_plane*sizeof(LE_Plane))) == NULL)
	  {
	    fatal("LE_init(): failed to (re)allocate %d bytes for LE\n",
		  N_LE_plane*sizeof(LE_Plane));
	  }
	  
	  /* 
	   * Now simply copy/set all the attributes to the local list. Note
	   * that the location is now expressed in local co-ordinates
	   * The LE (global) rank will be set later 
	   */
	  (LeesEdw_plane+N_LE_plane-1)->loc  = LeesEdw[i].loc-offset[X];
	  (LeesEdw_plane+N_LE_plane-1)->vel  = LeesEdw[i].vel;
	  (LeesEdw_plane+N_LE_plane-1)->frac = LeesEdw[i].frac;
	  (LeesEdw_plane+N_LE_plane-1)->disp = LeesEdw[i].disp;
	  (LeesEdw_plane+N_LE_plane-1)->peX  = cart_coords(X);
	  
	  /* Unset flag if LE Plane located at a PE boundary */
	  if( (LeesEdw[i].loc == offset[X]) ||
	      (LeesEdw[i].loc == (offset[X]+N[X])))
	    {
	      flag = 0;
	      /* One warning per offending wall is sufficient! */
	      if( (cart_coords(Y)==0) && (cart_coords(Z)==0) ){
		verbose("LE_init(): Lees-Edwards plane at a PE boundary!\n");
		verbose("Offending wall located at X = %d\n", 
			LeesEdw[i].loc);
	      }
	    }
	}
    }  

  /* Abort if any of the PEs has one or more walls at its domain boundary */
  MPI_Allreduce(&flag, &gblflag, 1, MPI_INT, MPI_LAND, cart_comm());
  if(gblflag == 0){
    fatal("LE_init(): wall at domain boundary\n");
  }
  
  /* 
   * Set communicator for Lees-Edwards communications: all LE planes have a
   * fixed X location and therefore span all the PEs for a given cart_coords(X) 
   */
  MPI_Comm_split(cart_comm(), cart_coords(X), cart_rank(), &LeesEdw_Comm);

  if(N_LE_plane)
    {
      /* Initialise (global) rank of LE walls (local list) */
      for (i=0; i < N_LE_total; i++){
	/* Get global rank of first local LE wall */
	if((LeesEdw_plane[0].loc+offset[X]) == LeesEdw[i].loc){
	  LeesEdw_plane[0].rank = LeesEdw[i].rank;
	  break;
	}
      }
      /* Then, simply increment (LE walls have been sorted earlier) */
      for(i=1; i<N_LE_plane; i++){
	LeesEdw_plane[i].rank = LeesEdw_plane[0].rank+i;
      }
           
      /* 
       * Create look-up table to translate global ranks (in cart_comm()) into
       * local ranks (in LeesEdw_Comm), i.e.:
       * LE_rank[<rank in cart_comm()>] = <rank in LeesEdw_Comm> 
       */
      MPI_Comm_rank(LeesEdw_Comm,&LE_rank);

      LE_ranks = (int *)malloc(pe_size()*sizeof(int));
      intbuff  = (int *)malloc(cart_size(Y)*cart_size(Z)*sizeof(int));
      if((LE_ranks==NULL) || (intbuff==NULL))
	{
	  fatal("COM_Init(): failed to allocate %d bytes ranks of LE walls\n",
		(pe_size() + cart_size(Y)*cart_size(Z))*sizeof(int));
	}
      
      /* -1 denotes cartesian ranks belonging to external LE communicators */
      for(i=0; i<pe_size(); i++){
	LE_ranks[i] = -1;
      }
      
      /* For any given LE communicator, gather all cartesian ranks (in order) */
      MPI_Gather(&gr_rank,1,MPI_INT,&intbuff[0],1,MPI_INT,0,LeesEdw_Comm);
      /* Then set up the look-up table on each LE root PE */
      if(LE_rank == 0){
	for(i=0; i<(cart_size(Y)*cart_size(Z)); i++){
	  LE_ranks[intbuff[i]] = i;
	}
      }
      /* Broadcast look-up table to other PEs in LE communicator (same cart_coords(X)) */
      MPI_Bcast(LE_ranks,pe_size(),MPI_INT,0,LeesEdw_Comm);
      free(intbuff);
      
      /* 
       * Allocate memory for Lees-Edwards buffers ("translated" values): see 
       * LE_update_buffers(). Storage required for LeesEdw_site[] is 
       * 2*N_LE_planes (one plane on each side of LE wall) 
       * * ny2z2 (number of sites in a YZ plane, including halos)
       * * 2 (because there are two distribution functions f and g) 
       * * LE_N_VEL_XING (simply save components crossing the LE wall) 
       */
      LeesEdw_site = (double *)
	malloc(2*N_LE_plane*ny2z2*2*LE_N_VEL_XING*sizeof(double));

      if(LeesEdw_site==NULL)
	{
	  fatal("LE_Init(): failed to allocate %d bytes buffers\n",
		(2*N_LE_plane*sizeof(double)*ny2z2*(2*LE_N_VEL_XING+1)));
	}
    }
  
#else  /* Serial */

  get_N_local(N);
  
  ny2z2 = (N[Y]+2*nhalo_)*(N[Z]+2*nhalo_);
  N_LE_plane = N_LE_total;

  /* LEBC parameters/properties will be stored in LeesEdw_plane */
  if(N_LE_plane > 0){
    if((LeesEdw_plane = (LE_Plane *)
	realloc(LeesEdw_plane,N_LE_plane*sizeof(LE_Plane))) == NULL)
      {
	fatal("LE_init(): failed to allocate %d bytes for LE parameters\n",
	      N_LE_plane*sizeof(LE_Plane));
      }
  }
  else{
    LeesEdw_plane = (LE_Plane *)NULL;
  }

  for(i = 0; i < N_LE_total; i++) {
    (LeesEdw_plane+i)->loc  = LeesEdw[i].loc;
    (LeesEdw_plane+i)->vel  = LeesEdw[i].vel;
    (LeesEdw_plane+i)->frac = LeesEdw[i].frac;
    (LeesEdw_plane+i)->disp = LeesEdw[i].disp;
    (LeesEdw_plane+i)->peX  = 0;
  }
  
  /* Initialise rank of LE walls */
  for(i=0; i< N_LE_total; i++){
    (LeesEdw_plane+i)->rank  =  LeesEdw[i].rank = i;
  }
  
  /* 
   * Allocate memory for Lees-Edwards buffers ("translated" values) and 
   * "unrolled" phis: see LE_update_buffers() and LE_unroll_phi().
   * Storage required for LeesEdw_site[] is:
   *   2*N_LE_planes   one plane on each side of LE wall
   *   * ny2z2         number of sites in a YZ plane, including halos
   *   * 2             because there are two distribution functions f and g
   *   * LE_N_VEL_XING simply save components crossing the LE wall 
   *   * sizeof(double) because site components are doubles
   */

  if (N_LE_plane > 0) {
    LeesEdw_site = (double *)
      malloc(2*N_LE_plane*ny2z2*2*LE_N_VEL_XING*sizeof(double));
  
    if(LeesEdw_site==NULL)
      {
	fatal("LE_Init(): failed to allocate %d bytes for LE buffers\n",
	      2*(2*LE_N_VEL_XING+1)*N_LE_plane*ny2z2*sizeof(double));
      }
  }
#endif /* _MPI_ */
  
  /* Not executed if there are no LE walls */
  for(plane=0; plane<N_LE_plane; plane++)
    {
      /* 
       * Sets initial Lees-Edwards struct members (displacement, weight, etc).
       * Note that the following bounds:
       * -gbl.N_total.y < displ < gbl.N_total.y
       * -gbl.N_total.y <=integ <=gbl.N_total.y-1
       *  and       0.0 < LeesEdw_plane[plane].frac <= 1.0
       */
      displ = fmod(LeesEdw_plane[plane].disp,1.0*N_total(Y));
      integ = (int)floor(displ);
      displ -= integ;                            /* now   0.0 <= displ < 1.0 */
      frac = 1.0 - displ;                        /* hence 0.0 <  frac  <=1.0 */
      
      /* 
       * LeesEdw_plane[plane].frac contains the weight of the left-hand site,
       * i.e., the site corresponding to floor(LeesEdw_plane[plane].disp)
       */
      LeesEdw_plane[plane].frac = frac;
    }

}

/*----------------------------------------------------------------------------*/
/*!
 * Copy (remote) shifted YZ LE plane into local buffers LeesEdw_phi[] and
 * LeesEdw_site[] for use by LE_apply_LEBC(). This routine uses linear
 * interpolation
 *
 *- \c Options:   _TRACE_, _VERBOSE_, _MPI_
 *- \c Arguments: void
 *- \c Returns:   void
 *- \c Buffers:   no dependence
 *- \c Version:   2.0b1
 *- \c Last \c updated: 03/03/2002 by JCD
 *- \c Authors:   JC Desplat
 *- \c See \c also: LE_update_buffers(), LE_update_buffers_cubic(), 
 *                LE_apply_LEBC() 
 *- \c Note:      compatible with serial and MPI implementations with no
 *                particular restrictions. 
 */
/*----------------------------------------------------------------------------*/
#define TAG_LE_START 3102
#define TAG_LE_END   3103


void LE_update_buffers( int target_buff )
{
  int     jj, kk, ind, ind0, ind1, ind2, xfac, yfac, xfac2, yfac2, zfac2;
  int     integ, LE_loc, plane, vel_ind;
  int     disp_j1,disp_j2;
  double   LE_frac;
  int     N[3];

#ifdef _MPI_

  int      nsites, nsites1, nsites2;
  int      source_rank1, source_rank2;
  int      target_rank1, target_rank2, target_pe1[3], target_pe2[3];
  int      i, start_y;
  int      offset[3];
  Site    *buff_site;
  MPI_Request req[4];
  MPI_Status status[4];
#endif /* _MPI_ */

  if (N_LE_total == 0) {
    VERBOSE(("LE_update_buffers(): no planes present\n"));
    return;
  }

  /* Only requires communications if the Y axis is distributed in the PE grid */
  if (cart_size(Y) > 1) {
  
#ifdef _MPI_ /* Parallel (MPI) section */

    get_N_local(N);
    get_N_offset(offset);
    yfac = N[Z]+2*nhalo_;
    xfac = (N[Y]+2*nhalo_)*(N[Z]+2*nhalo_);
    nsites = (N[Y]+2*nhalo_+1)*(N[Z]+2*nhalo_);

    /* 
     * Set up buffer of translated sites (by linear interpolation):
     * 1. Copy (N[Y]+3)*(N[Z]+2) translated sites in buff_site[]
     * 2. Perform linear interpolation and copy to LeesEdw_site[]
     */

    /* Strides for LeesEdw_site[] */
    zfac2 = LE_N_VEL_XING * 2;  /* x2 because 2 distrib funcs (f and g) */
    yfac2 = yfac * zfac2;
    xfac2 = xfac * zfac2; 

      /* JCD: the space of the buffer looks wrongly set !!! CHECK !!!*/
      /* Allocate memory for buffering sites */
      if((buff_site = (Site *)malloc(2*nsites*sizeof(Site))) == NULL)
	{
	  fatal("LE_update_buffers(): could not allocate %d bytes\n",
		2*nsites*sizeof(Site));
	}

      for(i=0; i<N_LE_plane; i++)
	{
	  LE_frac =     LeesEdw_plane[i].frac;
	  LE_loc  =     LeesEdw_plane[i].loc;
	  integ = floor(LeesEdw_plane[i].disp);
	  integ = integ%N_total(Y);

	  /*
	   * Plane below (going down): +ve displacement:
	   * LeesEdw_site[i] =
	   *            frac*buff_site[i+integ] + (1-frac)*buff_site[i+integ+1]
	   */

	  /* Starting y coordinate (global address): range 1->N_total.y */
	  start_y = ((offset[Y]+integ+2*N_total(Y)-1) % N_total(Y)) + 1;
      
	  /* Get ranks of both target PEs (target_rank1 and target_rank2) */
	  /* Note PEi responsible for start_y in (local) range 0->N[Y]-1 */
	  target_pe1[Y] = (start_y / N[Y]) % cart_size(Y);
	  target_pe2[Y] = (target_pe1[Y]+1) % cart_size(Y);
	  target_pe1[X] = target_pe2[X] = cart_coords(X);
	  target_pe1[Z] = target_pe2[Z] = cart_coords(Z);
	  MPI_Cart_rank(cart_comm(),target_pe1,&target_rank1);
	  MPI_Cart_rank(cart_comm(),target_pe2,&target_rank2);
	  target_rank1 = LE_ranks[target_rank1];
	  target_rank2 = LE_ranks[target_rank2];
      
	  /* Starting y coordinate (now local address on PE target_rank1) */
	  /* Valid values for start_y are in the range 0->N[Y]-1 */
	  /* Obviously remainder starts at y=1 on PE target_rank2 */
	  start_y = start_y % N[Y];
      
	  /* Number of sites to fetch from target_rank1 and target_rank2 */
	  /* Note that nsites = nsites1+nsites2 = (N[Y]+3)*(N[Z]+2) */
	  nsites1 = (N[Y]-start_y+1)*(N[Z]+2*nhalo_);
	  nsites2 =     (start_y+2)*(N[Z]+2*nhalo_);
      
        /* Use point-to-point communication */

        source_rank1 = target_rank1;
        source_rank2 = target_rank2;
        target_pe1[Y] = (cart_coords(Y) -
                         ((target_pe1[Y]-cart_coords(Y) + cart_size(Y)) 
                          % cart_size(Y))
                         + cart_size(Y)) % cart_size(Y);
        target_pe2[Y] = (target_pe1[Y] - 1 + cart_size(Y)) % cart_size(Y);
        target_pe1[X] = target_pe2[X] = cart_coords(X);
        target_pe1[Z] = target_pe2[Z] = cart_coords(Z);
        MPI_Cart_rank(cart_comm(), target_pe1, &target_rank1);
        MPI_Cart_rank(cart_comm(), target_pe2, &target_rank2);
        target_rank1 = LE_ranks[target_rank1];
        target_rank2 = LE_ranks[target_rank2];

        MPI_Irecv(&buff_site[0].f[0], nsites1, DT_Site, source_rank1,
                  TAG_LE_START, LeesEdw_Comm, &req[0]);
        MPI_Irecv(&buff_site[nsites1].f[0], nsites2, DT_Site, source_rank2,
                  TAG_LE_END, LeesEdw_Comm, &req[1]);
        MPI_Issend(&site[LE_loc*xfac+start_y*yfac].f[0], nsites1, DT_Site,
                   target_rank1, TAG_LE_START, LeesEdw_Comm, &req[2]);
        MPI_Issend(&site[LE_loc*xfac+yfac].f[0], nsites2, DT_Site,
                   target_rank2, TAG_LE_END, LeesEdw_Comm, &req[3]);
        MPI_Waitall(4,req,status);

	  /* Plane above (going up): -ve displacement */
	  /* buff[i] = (1-frac)*phi[i-(integ+1)] + frac*phi[i-integ] */
      
	  /* Starting y coordinate (global address): range 1->N_total.y */
	  start_y = ((offset[Y]-integ+2*N_total(Y)-2) % N_total(Y)) + 1;
      
	  /* Get ranks of both target PEs (target_rank1 and target_rank2) */
	  /* Note PEi responsible for start_y in (local) range 0->N[Y]-1 */
	  target_pe1[Y] = (start_y / N[Y]) % cart_size(Y);
	  target_pe2[Y] = (target_pe1[Y]+1) % cart_size(Y);
	  target_pe1[X] = target_pe2[X] = cart_coords(X);
	  target_pe1[Z] = target_pe2[Z] = cart_coords(Z);
	  MPI_Cart_rank(cart_comm(),target_pe1,&target_rank1);
	  MPI_Cart_rank(cart_comm(),target_pe2,&target_rank2);
	  target_rank1 = LE_ranks[target_rank1];
	  target_rank2 = LE_ranks[target_rank2];
	  
	  /* Starting y coordinate (now local address on PE target_rank1) */
	  /* Valid values for start_y are in the range 0->N[Y]-1 */
	  /* Obviously remainder starts at y=1 on PE target_rank2 */
	  start_y = start_y % N[Y];
	  
	  /* Number of sites to fetch from target_rank1 and target_rank2 */
	  /* Note that nsites = nsites1+nsites2 = (N[Y]+3)*(N[Z]+2) */
	  nsites1 = (N[Y]-start_y+1)*(N[Z]+2*nhalo_);
	  nsites2 =     (start_y+2)*(N[Z]+2*nhalo_);
	  
        /* Use point-to-point communication */

        source_rank1 = target_rank1;
        source_rank2 = target_rank2;
        target_pe1[Y] = (cart_coords(Y) -
                         ((target_pe1[Y]-cart_coords(Y) + cart_size(Y))
                          % cart_size(Y))
                         + cart_size(Y)) % cart_size(Y);
        target_pe2[Y] = (target_pe1[Y] - 1 + cart_size(Y)) % cart_size(Y);
        target_pe1[X] = target_pe2[X] = cart_coords(X);
        target_pe1[Z] = target_pe2[Z] = cart_coords(Z);
        MPI_Cart_rank(cart_comm(), target_pe1, &target_rank1);
        MPI_Cart_rank(cart_comm(), target_pe2, &target_rank2);
        target_rank1 = LE_ranks[target_rank1];
        target_rank2 = LE_ranks[target_rank2];

        MPI_Irecv(&buff_site[nsites].f[0], nsites1, DT_Site, source_rank1,
                  TAG_LE_START, LeesEdw_Comm, &req[0]);
        MPI_Irecv(&buff_site[nsites+nsites1].f[0], nsites2, DT_Site,
                  source_rank2, TAG_LE_END, LeesEdw_Comm, &req[1]);
        MPI_Issend(&site[(LE_loc+1)*xfac+start_y*yfac].f[0], nsites1,
                   DT_Site,
                   target_rank1, TAG_LE_START, LeesEdw_Comm, &req[2]);
        MPI_Issend(&site[(LE_loc+1)*xfac+yfac].f[0], nsites2, DT_Site,
                   target_rank2, TAG_LE_END, LeesEdw_Comm, &req[3]);
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
	  for(jj=0; jj<=N[Y]+1; jj++)
	    for(kk=0; kk<=N[Z]+1; kk++)
	      {
		ind = 2*i*xfac2 + jj*yfac2 + kk*zfac2;
		ind0 =            jj*yfac  + kk;
		ind1 =        (jj+1)*yfac  + kk;
		ind2 = 0; 
		for(vel_ind=0; vel_ind<NVEL; vel_ind++)	      
		  if(cv[vel_ind][X]==1)
		    {
		      LeesEdw_site[ind+2*ind2] =
			LE_frac       * buff_site[ind0].f[vel_ind] + 
			(1.0-LE_frac) * buff_site[ind1].f[vel_ind];
		      LeesEdw_site[ind+2*ind2+1] =
			LE_frac       * buff_site[ind0].g[vel_ind] + 
			(1.0-LE_frac) * buff_site[ind1].g[vel_ind];
		      ind2++;
		    }
	      }
	  
	  /* Plane above */
	  for(jj=0; jj<=N[Y]+1; jj++)
	    for(kk=0; kk<=N[Z]+1; kk++)
	      {
		ind = (2*i+1)*xfac2 +  jj   *yfac2 + kk*zfac2;
		ind0 =       nsites +  jj   *yfac  + kk;
		ind1 =       nsites + (jj+1)*yfac  + kk;
		ind2 = 0;
		for(vel_ind=0; vel_ind<NVEL; vel_ind++)	      
		  if(cv[vel_ind][X]==-1)
		    {
		      LeesEdw_site[ind+2*ind2] =
			LE_frac       * buff_site[ind1].f[vel_ind] + 
			(1.0-LE_frac) * buff_site[ind0].f[vel_ind];
		      LeesEdw_site[ind+2*ind2+1] =
			LE_frac       * buff_site[ind1].g[vel_ind] + 
			(1.0-LE_frac) * buff_site[ind0].g[vel_ind];
		      ind2++;
		    }
	      }
	}
      
      free(buff_site);

#endif /* _MPI_ */
  }
  else {
    /* Serial, or MPI with cart_size(Y) = 1 */

    get_N_local(N);
    yfac  =  N[Z]+2*nhalo_;
    xfac  = (N[Y]+2*nhalo_) * (N[Z]+2*nhalo_);
    zfac2 = LE_N_VEL_XING * 2;  /* final x2 because fs and gs as well! */
    yfac2 = yfac * zfac2;
    xfac2 = xfac * zfac2; 

    /* Set up buffer of translated sites (only include velocities crossing the
	 LE walls) */
      for(plane=0; plane<N_LE_plane; plane++)
	{
	  LE_frac =     LeesEdw_plane[plane].frac;
	  LE_loc  =     LeesEdw_plane[plane].loc;
	  integ = floor(LeesEdw_plane[plane].disp);
	  integ = integ%N[Y];

	  /* Plane below (going down): +ve displacement */
	  /* site_buff[i] = frac*site[i+integ] + (1-frac)*site[i+(integ+1)] */
	  for(jj=1; jj<=N[Y]; jj++)
	    {
	      disp_j1 = ((jj+integ+2*N[Y]-1) % N[Y]) + 1;
	      disp_j2 = (disp_j1 % N[Y]) + 1;
	      for(kk=1; kk<=N[Z]; kk++)
		{
		  ind  = 2*plane*xfac2 +      jj*yfac2 + kk*zfac2;
		  ind0 = LE_loc *xfac  + disp_j1*yfac  + kk;
		  ind1 = LE_loc *xfac  + disp_j2*yfac  + kk;

		  ind0 = ADDR(LE_loc, disp_j1, kk);
		  ind1 = ADDR(LE_loc, disp_j2, kk);
		  
		  /* For each velocity intersecting the LE plane (up, ie X+1) */
		  /* [JCD] Not very smart... could be seriously optimised */
		  ind2 = 0; 
		  for(vel_ind=0; vel_ind<NVEL; vel_ind++)
		    if(cv[vel_ind][X]==1)
		      {
			LeesEdw_site[ind+2*ind2] = 
			        LE_frac*site[ind0].f[vel_ind] +
			  (1.0-LE_frac)*site[ind1].f[vel_ind];
			LeesEdw_site[ind+2*ind2+1] = 
			        LE_frac*site[ind0].g[vel_ind] +
			  (1.0-LE_frac)*site[ind1].g[vel_ind];
			ind2++;
		      }
		}
	    }
	  
	  /* Plane above: -ve displacement */
	  /* site[i] = frac*site[i-integ] + (1-frac)*site[i-(integ+1)] */
	  /* buff[i] = site[i-(integ+1)] */
	  for(jj=1; jj<=N[Y]; jj++)
	    {
	      disp_j1 = ((jj-integ+2*N[Y]-2) % N[Y]) + 1;
	      disp_j2 = ((disp_j1+N[Y]) % N[Y]) + 1;
	      for(kk=1; kk<=N[Z]; kk++)
		{
		  ind = (2*plane+1)*xfac2 +      jj*yfac2 + kk*zfac2;
		  ind0 = (LE_loc+1)*xfac  + disp_j1*yfac  + kk;
		  ind1 = (LE_loc+1)*xfac  + disp_j2*yfac  + kk;

		  ind0 = ADDR(LE_loc+1, disp_j1, kk);
		  ind1 = ADDR(LE_loc+1, disp_j2, kk);
		  
		  /* For each velocity intersecting the LE plane (up, ie X-1) */
		  /* [JCD] Not very smart... could be seriously optimised */
		  ind2 = 0;
		  for(vel_ind=0; vel_ind<NVEL; vel_ind++)	      
		    if(cv[vel_ind][X]==-1)
		      {
			LeesEdw_site[ind+2*ind2] = 
			        LE_frac*site[ind1].f[vel_ind] +
			  (1.0-LE_frac)*site[ind0].f[vel_ind];
			LeesEdw_site[ind+2*ind2+1] = 
                                LE_frac*site[ind1].g[vel_ind] +
			  (1.0-LE_frac)*site[ind0].g[vel_ind];
			ind2++;
		      }
		}
	    }
	}
  }

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
  int i, j, p;
  int N[3];
  double rho, u[ND], gradu[ND][ND];
  double eta;

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

  gradu[X][Y] = le_params_.shear_rate;

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

  return;
}

/*****************************************************************************
 *
 *  le_get_steady_uy
 *
 *  Return the velocity expected for steady shear profile at
 *  position x (dependent on x-direction only). Takes a local index.
 *
 *****************************************************************************/

double le_get_steady_uy(int ic) {

  int offset[3];
  int nplane;
  double xglobal, uy;

  get_N_offset(offset);

  /* The shear profile is linear, so the local velocity is just a
   * function of position, modulo the number of planes encountered
   * since the origin. The planes are half way between sites giving
   * the - 0.5. */

  xglobal = offset[X] + (double) ic - 0.5;
  nplane = (int) ((le_params_.dx_min + xglobal)/le_params_.dx_sep);

  uy = xglobal*le_params_.shear_rate - le_params_.uy_plane*nplane;
 
  return uy;
}

/*****************************************************************************
 *
 *  le_get_nplane
 *
 *  Return the local number of planes (block boundaries).
 *
 *****************************************************************************/

int le_get_nplane() {

  return N_LE_plane;
}

/*****************************************************************************
 *
 *  le_plane_location
 *
 *  Return location (x-coordinte - 0.5) of plane n.
 *  It is erroneous to call this if no planes.
 *
 *****************************************************************************/

int le_plane_location(const int n) {

  assert(initialised_);
  assert(n >= 0 && n < N_LE_plane);

  return LeesEdw_plane[n].loc;
}

/*****************************************************************************
 *
 *  le_get_nxbuffer
 *
 *  Return the size (x-direction) of the buffer required to hold
 *  cross-boundary interpolated quantities.
 *
 *****************************************************************************/

int le_get_nxbuffer() {

  assert(initialised_);

  return le_params_.nxbuffer;
}

/*****************************************************************************
 *
 *  le_index_real_to_buffer
 *
 *  For x index and step size di, return the x index of the translated
 *  buffer.
 *
 *****************************************************************************/

int le_index_real_to_buffer(const int ic, const int di) {

  int ib;

  assert(initialised_);
  assert(di >= -nhalo_ && di <= +nhalo_);

  ib = (ic + nhalo_ - 1)*(2*nhalo_ + 1) + di + nhalo_;

  assert(ib >= 0 && ib < le_params_.index_real_nbuffer);

  return le_params_.index_real_to_buffer[ib];
}

/*****************************************************************************
 *
 *  le_index_buffer_to_real
 *
 *  For x index in the buffer region, return the corresponding
 *  x index in the real system.
 *
 *****************************************************************************/

int le_index_buffer_to_real(int ib) {

  assert(initialised_);
  assert(ib >=0 && ib < le_params_.nxbuffer);

  return le_params_.index_buffer_to_real[ib];
}

/*****************************************************************************
 *
 *  le_buffer_displacement
 *
 *  Return the current displacement dy = du_y t for the buffer plane
 *  with x location ib.
 *
 *****************************************************************************/

double le_buffer_displacement(int ib) {

  /* The minus one is to ensure the regression test doesn't fail. The
   * displacement oringally updated between the phi and f_i
   * transformations */
  double dt = get_step() - 1.0;

  assert(initialised_);
  assert(ib >= 0 && ib < le_params_.nxbuffer);

  return dt*le_params_.buffer_duy[ib];
}

/*****************************************************************************
 *
 *  le_communicator
 *
 *  Return the handle to the Lees Edwards communicator.
 *
 *****************************************************************************/

MPI_Comm le_communicator() {

  assert(initialised_);
  return LeesEdw_Comm;
}

/*****************************************************************************
 *
 *  le_displacement_ranks
 *
 *  For a given  displacement, work out which two LE ranks
 *  are required for communication.
 *
 *****************************************************************************/

void le_displacement_ranks(const double dy, int recv[2], int send[2]) {

  int nlocal[3];
  int noffset[3];
  int pe1_cart[3];
  int pe2_cart[3];
  MPI_Comm cartesian = cart_comm();
  int jdy, j1;

  assert(initialised_);
  assert(LE_ranks);

  get_N_local(nlocal);
  get_N_offset(noffset);

  jdy = floor(fmod(dy, L(Y)));
  j1 = 1 + (noffset[Y] + 1 - nhalo_ - jdy - 2 + 2*N_total(Y)) % N_total(Y);

  pe1_cart[X] = cart_coords(X);
  pe1_cart[Y] = j1 / nlocal[Y];
  pe1_cart[Z] = cart_coords(Z);
  pe2_cart[X] = pe1_cart[X];
  pe2_cart[Y] = pe1_cart[Y] + 1;
  pe2_cart[Z] = pe1_cart[Z];

  MPI_Cart_rank(cartesian, pe1_cart, recv);
  MPI_Cart_rank(cartesian, pe2_cart, recv + 1);

  recv[0] = LE_ranks[recv[0]];
  recv[1] = LE_ranks[recv[1]];

  /* Send to ... */

  pe1_cart[Y] = cart_coords(Y) - (pe1_cart[Y] - cart_coords(Y));
  pe2_cart[Y] = pe1_cart[Y] - 1;

  MPI_Cart_rank(cartesian, pe1_cart, send);
  MPI_Cart_rank(cartesian, pe2_cart, send + 1);

  send[0] = LE_ranks[send[0]];
  send[1] = LE_ranks[send[1]];

  return;
}


/*****************************************************************************
 *
 *  le_site_index
 *
 *  Compute the one-dimensional index from coordinates ic, jc, kc.
 *  This differs from get_site_index only in construction (not in result).
 *
 *  Where performance is important, prefer macro version via NDEBUG.
 *
 *****************************************************************************/

int le_site_index(const int ic, const int jc, const int kc) {

  int nlocal[3];
  int index;

  get_N_local(nlocal);

  assert(initialised_);
  assert(ic >= 1-nhalo_);
  assert(jc >= 1-nhalo_);
  assert(kc >= 1-nhalo_);
  assert(ic <= nlocal[X] + nhalo_ + le_params_.nxbuffer);
  assert(jc <= nlocal[Y] + nhalo_);
  assert(kc <= nlocal[Z] + nhalo_);

  index = (nlocal[Y] + 2*nhalo_)*(nlocal[Z] + 2*nhalo_)*(nhalo_ + ic - 1)
    +                            (nlocal[Z] + 2*nhalo_)*(nhalo_ + jc - 1)
    +                                                    nhalo_ + kc - 1;

  return index;
}
