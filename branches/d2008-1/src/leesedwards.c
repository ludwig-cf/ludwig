
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
#include "phi.h"

extern Site * site;
extern double * phi_site;


#ifdef _MPI_

extern MPI_Datatype DT_FVector;
extern MPI_Datatype DT_Site;

static MPI_Comm     LeesEdw_Comm;/* Communicator for Lees-Edwards (LE) RMA */
static MPI_Comm     X_PEs_Comm;  /* Communicator for Lees-Edwards (X row) */
static MPI_Info     LeesEdw_Info;/* Information for RMAs (LE) */
static MPI_Datatype DT_plane_LE_Float;/* MPI datatype defining LE plane */
static MPI_Datatype DT_plane_LE_Float_exclhalo; /* For unwrapping phi (LE) */

#ifdef _MPI_2_
static MPI_Win      LE_Site_Win; /* MPI window for LE buffering (Sites) */
static MPI_Win      LE_Float_Win;/* MPI window for LE buffering (Floats) */
#endif

#endif /* _MPI_ */

static double      *LeesEdw_site;
static int        *LE_distrib;
static int        *LE_ranks;


static LE_Plane   *LeesEdw_plane;
static double      *LeesEdw_phi;

static int        N_LE_plane;
static int        N_LE_total = 0;
static LE_Plane * LeesEdw = NULL;

static int    LE_cmpLEBC( const void *, const void * );
static void   LE_print_LEbuffers( void );
static void   LE_init_original(void);
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
} le_params_;

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

    if (nhalo_ > 1) fatal("nahlo = %d in le code\n", nhalo_);
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

  return;
}


/*----------------------------------------------------------------------------*/
/*!
 * Applies Lees-Edwards: transformation of site contents across LE walls: 
 * applies to both phi (for the computation of the gradients across LE walls)
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
  char filename[256]; /* DBG */

  double *f, *g;
  double rho, phi, ds[3][3], dsphi[3][3], udotc, jdotc, sdotq, sphidotq;
  FVector u,du,jphi,djphi;

  const double r2rcs4 = 4.5;         /* The constant 1 / 2 c_s^4 */

  int p,side;

  if (N_LE_plane == 0) {
    VERBOSE(("LE_apply_LEBC(): no walls present\n"));
    return;
  }

  TIMER_start(TIMER_LE);

  get_N_local(N);
  yfac  =  N[Z]+2;
  xfac  = (N[Y]+2) * (N[Z]+2);
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
	LE_vel = -LE_vel;
	LE_loc++;
      }      

      /* First, for plane `below' LE plane, ie, crossing LE plane going up */

      for (jj = 1; jj <= N[Y]; jj++) {
	for (kk = 1; kk <= N[Z]; kk++) {
	  
	  ind = LE_loc*xfac + jj*yfac + kk;
	    	  
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
	  u.x = 0.0;
	  u.y = 0.0;
	  u.z = 0.0;

	  jphi.x = 0.0;
	  jphi.y = 0.0;
	  jphi.z = 0.0;
	  for (p = 1; p < NVEL; p++) {
	    rho    += f[p];
	    u.x    += f[p]*cv[p][X];
	    u.y    += f[p]*cv[p][Y];
	    u.z    += f[p]*cv[p][Z];
	    phi    += g[p];
	    jphi.x += g[p]*cv[p][0];
	    jphi.y += g[p]*cv[p][1];
	    jphi.z += g[p]*cv[p][2];
	  }
 
	  /* Include correction for Lees-Edwards BC: first for the stress */
	  /* NOTE: use the original u.y and jphi[1], i.e., befoer LE fix */
	  ds[0][0] = 0.0;
	  ds[0][1] = LE_vel*u.x*rho;
	  ds[0][2] = 0.0;
	  ds[1][0] = LE_vel*u.x*rho;
	  ds[1][1] = (rho*LE_vel*(2*u.y+LE_vel));
	  ds[1][2] = LE_vel*u.z*rho;
	  ds[2][0] = 0.0;
	  ds[2][1] = LE_vel*u.z*rho;
	  ds[2][2] = 0.0;
	    
	  dsphi[0][0] = 0.0; 
	  dsphi[0][1] = LE_vel*jphi.x; 
	  dsphi[0][2] = 0.0; 
	  dsphi[1][0] = LE_vel*jphi.x; 
	  dsphi[1][1] = LE_vel*(2*jphi.y+LE_vel*phi);
	  dsphi[1][2] = LE_vel*jphi.z; 
	  dsphi[2][0] = 0.0;
	  dsphi[2][1] = LE_vel*jphi.z; 
	  dsphi[2][2] = 0.0; 

	  /* ... then for the momentum (note that whilst jphi[] represents */
	  /* the moment, u only represents the momentum) */
	  du.x    = 0.0;
	  du.y    = LE_vel; 
	  du.z    = 0.0;
	  djphi.x = 0.0;
	  djphi.y = phi*LE_vel;
	  djphi.z = 0.0;

	  /* Now update the distribution */
	  for (p = 0; p < NVEL; p++) {
	      
	    udotc =    du.y * cv[p][1];
	    jdotc = djphi.y * cv[p][1];
	      
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
  LE_update_buffers(SITE_AND_PHI);

  /* Stage 4: apply translation on fs and gs crossing LE planes */

  for (plane = 0; plane < N_LE_plane; plane++) {
    LE_frac = LeesEdw_plane[plane].frac;
    LE_loc  = LeesEdw_plane[plane].loc;
      
    /* First, for plane 'below' LE plane: displacement = +displ */
    for (jj = 1; jj <= N[Y]; jj++) {
      for (kk = 1; kk <= N[Z]; kk++) {
	ind  =  LE_loc*xfac  + jj*yfac  + kk;
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
	ind  =  (LE_loc+1)*xfac  + jj*yfac  + kk;
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
  int     flag;
  int     plane, side, ind, xfac, yfac, integ, i, j, k, N_sites, ny2z2;
  double   displ, frac, LE_vel;
  int     N[3];
    
#ifdef _MPI_ /* Parallel (MPI) implementation */
  int     size1, size2, colour;
  int     gblflag, rank, LE_rank, *intbuff, ny3z2;
  int     offset[3];
  int     gr_rank;

  get_N_local(N);
  get_N_offset(offset);
  gr_rank = cart_rank();

  ny2z2 = (N[Y]+2)*(N[Z]+2);
  ny3z2 = (N[Y]+3)*(N[Z]+2);
  xfac = (N[Y]+2)*(N[Z]+2);
  yfac = (N[Z]+2);

  N_sites = (N[X]+2)*(N[Y]+2)*(N[Z]+2);
  
  /* Set up Lees-Edwards specific MPI datatypes */
  
  /*
   * (YZ) plane for Lees-Edwards: etype is double instead of Site: one
   * contiguous block of (N[Y]+2)*(N[Z]+2) doubles. No extra row here as
   * interpolation is carried out before buffering 
   */
  MPI_Type_contiguous(ny2z2, MPI_DOUBLE,&DT_plane_LE_Float);
  MPI_Type_commit(&DT_plane_LE_Float);

  /* 
   * (YZ) plane for Lees-Edwards (unrolling): N[Y] blocks of N[Z] Floats with 
   * stride (yfac=N[Z]+2)
   */
  MPI_Type_vector(N[Y],N[Z],(N[Z]+2), MPI_DOUBLE,&DT_plane_LE_Float_exclhalo);
  MPI_Type_commit(&DT_plane_LE_Float_exclhalo);
  
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

  /* Set communicator along X row (only 1 PE per cart_coords(X) is sufficient!) */
  colour = ((cart_coords(Y)==0) && (cart_coords(Z)==0)) ? 0 : 1;
  /* Note: cart_coords(X) and rank in new communicator will be identical */
  MPI_Comm_split(cart_comm(), colour, cart_coords(X), &X_PEs_Comm);


  if(N_LE_plane)
    {
      /* Sort list of local LE planes by increasing (X) position */
      qsort((LE_Plane *) LeesEdw_plane, N_LE_plane, sizeof(LE_Plane),
	    LE_cmpLEBC);

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
      
      /* Set info parameters for RMA windows. See LE_update_buffers() */
      MPI_Info_create(&LeesEdw_Info);
      MPI_Info_set(LeesEdw_Info, "no_locks", "true");
      
      /* 
       * Allocate memory for Lees-Edwards buffers ("translated" values): see 
       * LE_update_buffers(). Storage required for LeesEdw_site[] is 
       * 2*N_LE_planes (one plane on each side of LE wall) 
       * * ny2z2 (number of sites in a YZ plane, including halos)
       * * 2 (because there are two distribution functions f and g) 
       * * LE_N_VEL_XING (simply save components crossing the LE wall) 
       * * sizeof(Float) (because site components are Floats) 
       */
      LeesEdw_site = (double *)
	malloc(2*N_LE_plane*ny2z2*2*LE_N_VEL_XING*sizeof(double));
      LeesEdw_phi = (double *)malloc(2*N_LE_plane*ny2z2*sizeof(double));
      if((LeesEdw_site==NULL) || (LeesEdw_phi==NULL))
	{
	  fatal("LE_Init(): failed to allocate %d bytes buffers\n",
		(2*N_LE_plane*sizeof(double)*ny2z2*(2*LE_N_VEL_XING+1)));
	}
    }
  
#else  /* Serial */

  get_N_local(N);
  
  N_sites = (N[X]+2)*(N[Y]+2)*(N[Z]+2);
  ny2z2 = (N[Y]+2)*(N[Z]+2);
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
    LeesEdw_phi = (double *)malloc(2*N_LE_plane*ny2z2*sizeof(double));
  
    if((LeesEdw_site==NULL) || (LeesEdw_phi==NULL))
      {
	fatal("LE_Init(): failed to allocate %d bytes for LE buffers\n",
	      N_sites*sizeof(double) +
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

#ifdef _MPI_ /* Specific to MPI implementation */
  /* 
   * Set RMA window for LE_update_buffers(): required for MPI_Get/MPI_Put
   * This needs to take place *AFTER* the memory has been allocated.
   */

  /* Only required if there is a LE plane intersecting the local domain */
  /* Two types of RMA operations will take place: for the Sites, and for Phis */
  if(N_LE_plane) {
  
      /* Now, set RMA windows */
      size1 = (N[X]+2)*(N[Y]+2)*(N[Z]+2)*sizeof(Site);
      size2 = (N[X]+2)*(N[Y]+2)*(N[Z]+2)*sizeof(double);
#ifdef _MPI_2_
      MPI_Win_create(&site[0].f[0],size1,sizeof(Site),LeesEdw_Info,
		     LeesEdw_Comm,&LE_Site_Win);
      MPI_Win_create(&phi_site[0],size2,sizeof(double),LeesEdw_Info,
		     LeesEdw_Comm,&LE_Float_Win);
#endif
    }
  
  /* 
   * Initialise utility arrays: store the distribution of planes on each PE:
   * LE_distrib[i] will contain the number of LE planes for any PE located at
   * cart_coords(X) == i
   */

  if((LE_distrib = (int *)malloc(cart_size(X)*sizeof(int))) == NULL)
    {
      fatal("LE_init(): failed to allocate %d bytes LE utility array\n",
	    cart_size(X)*sizeof(int));
    }

  /* Only requires 1PE for each value of cart_coords(X) (colour == 0) */
  colour = ((cart_coords(Y)==0) && (cart_coords(Z)==0)) ? 0 : 1;
  if(colour==0)
    {
      MPI_Allgather(&N_LE_plane,1,MPI_INT,&LE_distrib[0],1,MPI_INT,X_PEs_Comm);
      MPI_Barrier(X_PEs_Comm);
    }
  /* ... and now broadcast to all PEs */
  MPI_Bcast(&LE_distrib[0],cart_size(X),MPI_INT,0,cart_comm());

  /* Required to ensure that the RMS windows have been declared before entering
   LE_update_buffers() */
  MPI_Barrier(MPI_COMM_WORLD);
  
#endif /* _MPI_ */

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
  int     i, jj, kk, ind, ind0, ind1, ind2, xfac, yfac, xfac2, yfac2, zfac2;
  int     start_y, target_rank1, target_rank2, target_pe1[3], target_pe2[3];
  int     integ, LE_loc, plane, nsites, nsites1, nsites2, vel_ind;
  int     disp_j1,disp_j2;
  int     source_rank1, source_rank2;
  double   LE_frac, LE_vel, rho, phi, *buff_phi;
  int     N[3];
  int     offset[3];
  Site    *buff_site;

#ifdef _MPI_
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
    yfac = N[Z]+2;
    xfac = (N[Y]+2)*(N[Z]+2);
    nsites = (N[Y]+3)*(N[Z]+2);   /* extra row for linear interpolation */

    switch ( target_buff ) {

      /* 
       * Update both LeesEdw_site[] and LeesEdw_phi[]. Both are computed by
       * linear interpolation. Note that it *may* be more efficient to derive
       * the latter from LeesEdw_site[] instead. Case typically called by 
       * LE_apply_LEBC()
       */
    case SITE_AND_PHI:

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
	  nsites1 = (N[Y]-start_y+1)*(N[Z]+2);
	  nsites2 =     (start_y+2)*(N[Z]+2);
      
	  /* Using MPI_MODE_NOPRECEDE may make a difference */
#ifdef _MPI_2_
	  MPI_Win_fence(0,LE_Site_Win);
	  MPI_Get(&buff_site[0].f[0],nsites1,DT_Site,target_rank1,
		  LE_loc*xfac+start_y*yfac,nsites1,DT_Site,LE_Site_Win);
	  MPI_Get(&buff_site[nsites1].f[0],nsites2,DT_Site,target_rank2,
		  LE_loc*xfac+yfac,nsites2,DT_Site,LE_Site_Win);
#else
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

#endif
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
	  nsites1 = (N[Y]-start_y+1)*(N[Z]+2);
	  nsites2 =     (start_y+2)*(N[Z]+2);
	  
	  /* Then, the complete sites themselves (.fs and .gs) */
#ifdef _MPI_2_
	  MPI_Get(&buff_site[nsites].f[0],nsites1,DT_Site,target_rank1,
		  (LE_loc+1)*xfac+start_y*yfac,nsites1,DT_Site,LE_Site_Win);
	  MPI_Get(&buff_site[nsites+nsites1].f[0],nsites2,DT_Site,target_rank2,
		  (LE_loc+1)*xfac+yfac,nsites2,DT_Site,LE_Site_Win);
	  MPI_Win_fence(0,LE_Site_Win);
#else
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

#endif


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
      
      /* Now compute LeesEdw_phi[]... simply by not using a break statement! */

      /* 
       * Only update LeesEdw_phi[] (typically required by MODEL_get_gradients()
       */
    case PHI_ONLY:
      
      /* 
       * Set up buffer of translated phis (by linear interpolation):
       * 1. Copy (N[Y]+3)*(N[Z]+2) translated phis in buff_phi[]
       * 2. Perform linear interpolation and copy to LeesEdw_phi[]
       */
      
      /* JCD: size of buffer could be brought down!!! CHECK!!!*/
      /* Allocate memory for buffering phis */
      if((buff_phi = (double *)malloc(2*nsites*sizeof(double))) == NULL)
	{
	  fatal("LE_update_buffers(): could not allocate %d bytes\n",
		2*nsites*sizeof(double));
	}

      for(i=0; i<N_LE_plane; i++)
	{
	  LE_frac =     LeesEdw_plane[i].frac;
	  LE_loc  =     LeesEdw_plane[i].loc;
	  integ = floor(LeesEdw_plane[i].disp);
	  integ = integ%N_total(Y);
      
	  /*
	   * Plane below (going down): +ve displacement:
	   * LeesEdw_phi[i] =
	   *            frac*buff_phi[i+integ] + (1-frac)*buff_phi[i+integ+1]
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
	  nsites1 = (N[Y]-start_y+1)*(N[Z]+2);
	  nsites2 =     (start_y+2)*(N[Z]+2);
      
	  /* Using MPI_MODE_NOPRECEDE may make a difference */
#ifdef _MPI_2_
	  MPI_Win_fence(0,LE_Float_Win);
	  MPI_Get(&buff_phi[0],nsites1, MPI_DOUBLE,target_rank1,
		  LE_loc*xfac+start_y*yfac,nsites1,MPI_DOUBLE,LE_Float_Win);
	  MPI_Get(&buff_phi[nsites1],nsites2, MPI_DOUBLE,target_rank2,
		  LE_loc*xfac+yfac,nsites2,MPI_DOUBLE,LE_Float_Win);
#else
          source_rank1 = target_rank1;
          source_rank2 = target_rank2;
          target_pe1[Y] = (cart_coords(Y) -
                           ((target_pe1[Y] - cart_coords(Y) + cart_size(Y))
                            %cart_size(Y))
                           + cart_size(Y)) % cart_size(Y);
          target_pe2[Y] = (target_pe1[Y]-1+cart_size(Y)) % cart_size(Y);
          target_pe1[X] = target_pe2[X] = cart_coords(X);
          target_pe1[Z] = target_pe2[Z] = cart_coords(Z);
          MPI_Cart_rank(cart_comm(), target_pe1, &target_rank1);
          MPI_Cart_rank(cart_comm(), target_pe2, &target_rank2);
          target_rank1 = LE_ranks[target_rank1];
          target_rank2 = LE_ranks[target_rank2];

          MPI_Irecv(&buff_phi[0],nsites1, MPI_DOUBLE, source_rank1,
                    TAG_LE_START, LeesEdw_Comm, &req[0]);
          MPI_Irecv(&buff_phi[nsites1], nsites2, MPI_DOUBLE, source_rank2,
                    TAG_LE_END, LeesEdw_Comm, &req[1]);
          MPI_Issend(&phi_site[LE_loc*xfac+start_y*yfac], nsites1, MPI_DOUBLE,
                     target_rank1, TAG_LE_START, LeesEdw_Comm, &req[2]);
          MPI_Issend(&phi_site[LE_loc*xfac+yfac], nsites2, MPI_DOUBLE,
                     target_rank2, TAG_LE_END, LeesEdw_Comm, &req[3]);
          MPI_Waitall(4,req,status);

#endif      
	  /*
	   * Plane above (going up): -ve displacement:
	   * LeesEdw_phi[i] =
	   *           (1-frac)*buff_phi[i-(integ+1)] + frac*buff_phi[i-integ]
	   */

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
	  nsites1 = (N[Y]-start_y+1)*(N[Z]+2);
	  nsites2 =     (start_y+2)*(N[Z]+2);
#ifdef _MPI_2_      
	  MPI_Get(&buff_phi[nsites],nsites1,MPI_DOUBLE,target_rank1,
		  (LE_loc+1)*xfac+start_y*yfac,nsites1,MPI_DOUBLE,
		  LE_Float_Win);
	  MPI_Get(&buff_phi[nsites+nsites1],nsites2,MPI_DOUBLE,target_rank2,
		  (LE_loc+1)*xfac+yfac,nsites2,MPI_DOUBLE,LE_Float_Win);
	  MPI_Win_fence(0,LE_Float_Win);
#else
          source_rank1 = target_rank1;
          source_rank2 = target_rank2;
          target_pe1[Y] = (cart_coords(Y) -
                           ((target_pe1[Y] - cart_coords(Y) + cart_size(Y))
                            %cart_size(Y))
                           + cart_size(Y)) % cart_size(Y);
          target_pe2[Y] = (target_pe1[Y]-1+cart_size(Y)) % cart_size(Y);
          target_pe1[X] = target_pe2[X] = cart_coords(X);
          target_pe1[Z] = target_pe2[Z] = cart_coords(Z);
          MPI_Cart_rank(cart_comm(), target_pe1, &target_rank1);
          MPI_Cart_rank(cart_comm(), target_pe2, &target_rank2);
          target_rank1 = LE_ranks[target_rank1];
          target_rank2 = LE_ranks[target_rank2];
          MPI_Irecv(&buff_phi[nsites],nsites1, MPI_DOUBLE, source_rank1,
                    TAG_LE_START,LeesEdw_Comm,&req[0]);
          MPI_Irecv(&buff_phi[nsites+nsites1],nsites2, MPI_DOUBLE,
                    source_rank2,
                    TAG_LE_END, LeesEdw_Comm, &req[1]);
          MPI_Issend(&phi_site[(LE_loc+1)*xfac+start_y*yfac],nsites1,
                     MPI_DOUBLE,
                     target_rank1,TAG_LE_START,LeesEdw_Comm,&req[2]);
          MPI_Issend(&phi_site[(LE_loc+1)*xfac+yfac],nsites2,MPI_DOUBLE,
                     target_rank2,TAG_LE_END,LeesEdw_Comm,&req[3]);
          MPI_Waitall(4,req,status);

#endif      
	  /* Perform linear interpolation on buffer of phis */
	  /* Plane below */
	  for(jj=0; jj<=N[Y]+1; jj++)
	    for(kk=0; kk<=N[Z]+1; kk++)
	      {
		ind = 2*i*xfac + jj*yfac + kk;
		ind0 =           jj*yfac + kk;
		ind1 =       (jj+1)*yfac + kk;
		LeesEdw_phi[ind] =
		  LE_frac*buff_phi[ind0] + (1.0-LE_frac)*buff_phi[ind1];
	      }
	  /* Plane above */
	  for(jj=0; jj<=N[Y]+1; jj++)
	    for(kk=0; kk<=N[Z]+1; kk++)
	      {
		ind = (2*i+1)*xfac +     jj*yfac + kk;
		ind0 =      nsites +     jj*yfac + kk;
		ind1 =      nsites + (jj+1)*yfac + kk;
		LeesEdw_phi[ind] =
		  (1.0-LE_frac)*buff_phi[ind0] + LE_frac*buff_phi[ind1];
	      }
	}

      free(buff_phi);
      
      break;
      
      /* We definitely shouldn't be here... */
    default:
      fatal("LE_update_buffers(): argument not recognised.\n");
    }

#endif /* _MPI_ */
  }
  else {
    /* Serial, or MPI with cart_size(Y) = 1 */

    get_N_local(N);
    yfac  =  N[Z]+2;
    xfac  = (N[Y]+2) * (N[Z]+2);
    zfac2 = LE_N_VEL_XING * 2;  /* final x2 because fs and gs as well! */
    yfac2 = yfac * zfac2;
    xfac2 = xfac * zfac2; 

    switch( target_buff ) {

      /* 
       * Update both LeesEdw_site[] and LeesEdw_phi[]. Both are computed by
       * linear interpolation. Note that it *may* be more efficient to derive
       * the latter from LeesEdw_site[] instead. Case typically called by 
       * LE_apply_LEBC()
       */
    case SITE_AND_PHI:
      
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
      
      /* Now compute LeesEdw_phi[]... simply by not using a break statement! */

      /* Only update of LeesEdw_phi[], typically required by
	 MODEL_get_gradients() */
    case PHI_ONLY:
      
      /* Set up buffer of translated phis (by linear interpolation) */
      for(plane=0; plane<N_LE_plane; plane++)
	{
	  LE_frac =     LeesEdw_plane[plane].frac;
	  LE_loc  =     LeesEdw_plane[plane].loc;
	  integ = floor(LeesEdw_plane[plane].disp);
	  integ = integ%N[Y];
	  
	  /* Plane below (going down): +ve displacement */
	  /* buff[i] = frac*phi[i+integ] + (1-frac)*phi[i+integ+1] */
	  for(jj=0; jj<=N[Y]+1; jj++)
	    {
	      disp_j1 = ((jj+integ+2*N[Y]-1) % N[Y]) + 1;
	      disp_j2 = (disp_j1 % N[Y]) + 1;
	      for(kk=0; kk<=N[Z]+1; kk++)
		{
		  ind = 2*plane*xfac +      jj*yfac + kk;
		  ind0 = LE_loc*xfac + disp_j1*yfac + kk;
		  ind1 = LE_loc*xfac + disp_j2*yfac + kk;
		  LeesEdw_phi[ind] = LE_frac*phi_site[ind0] + 
		    (1.0-LE_frac)*phi_site[ind1];
		}
	    }
	  
	  /* Plane above (going up): -ve displacement */
	  /* buff[i] = (1-frac)*phi[i-(integ+1)] + frac*phi[i-integ] */
	  for(jj=0; jj<=N[Y]+1; jj++)
	    {
	      disp_j1 = ((jj-integ+2*N[Y]-2) % N[Y]) + 1;
	      disp_j2 = ((disp_j1+N[Y]) % N[Y]) + 1;
	      for(kk=0; kk<=N[Z]+1; kk++)
		{
		  ind = (2*plane+1)*xfac + jj*yfac + kk;
		  ind0 = (LE_loc+1)*xfac + disp_j1*yfac + kk;
		  ind1 = (LE_loc+1)*xfac + disp_j2*yfac + kk;
		  LeesEdw_phi[ind] = (1.0-LE_frac)*phi_site[ind0] +
		    LE_frac*phi_site[ind1];
		}
	    }
	}
      
      break;
      
      /* We definitely shouldn't be here... */
    default:
      fatal("LE_update_buffers(): argument not recognised.\n");
    }
  }

  return;
}

/*----------------------------------------------------------------------------*/
/*!
 * Print LE parameters to STDOUT for monitoring or restart. The format used 
 * follows that used in the input file, e.g., loc_vel_disp
 *
 *- \c Options:   _TRACE_, _MPI_
 *- \c Arguments: void
 *- \c Returns:   void
 *- \c Buffers:   no dependence
 *- \c Version:   2.0
 *- \c Last \c updated: 14/01/2002 by JCD
 *- \c Authors:   JC Desplat
 *- \c See \c also:
 *- \c Note:      typically called at the end of a run (or whenever doing a
 *                checkpointing) to print out the status of the LE walls to
 *                set a potential re-start
 */
/*----------------------------------------------------------------------------*/

void LE_print_params( void )
{
#ifdef _MPI_ /* Parallel (MPI) section */
  int rank, colour, *displ;
  int nplanes, i, j;

  /* Abort if no LE walls are present */
  if (N_LE_total==0) {
    VERBOSE(("LE_print_params(): no walls present\n"));
    return;
  }
  
  if((displ = (int *)malloc(cart_size(X)*sizeof(int))) == NULL)
    {
      fatal("LE_print_params(): failed to allocate %d bytes\n",
	    cart_size(X)*sizeof(int));
    }

  /* displ[i] contains the total number of planes on the LHS of any PE
     located at cart_coords(X) == i */
  displ[0] = 0;
  for(i=1; i<cart_size(X); i++){
    displ[i] = displ[i-1] + LE_distrib[i-1];
  }
  
  /* X_PEs_Comm is a communicator spanning the X row (only 1 PE per cart_coords(X)). See
     LE_init(). Note: cart_coords(X) and rank in X_PEs_Comm should be identical! */
  MPI_Comm_rank(X_PEs_Comm, &rank);
  
  /* Only get local values from one row (minimise communications) */
  colour = ((cart_coords(Y)==0) && (cart_coords(Z)==0)) ? 0 : 1;

  if(colour == 0)
    {
#ifdef KS_TO_SORT
      MPI_Gatherv(&LeesEdw_plane[0], N_LE_plane, DT_LE_GBL, &LeesEdw[0],
		  &LE_distrib[0], &displ[0], DT_LE_GBL, 0, X_PEs_Comm);
#endif
      /* Correct gbl.LeesEdw.loc (local -> global co-ordinates) */
      if(rank == 0)
	{
#ifdef KS_TO_SORT
	  for(i=0; i < N_LE_total; i++){
	    LeesEdw[i].loc += LeesEdw[i].peX*gbl.N[X];
	  }
#endif	  
	  /* Now, print current LE parameters in a form suitable for input,
	     i.e., Lees_Edwards location_velocity_displacement */
	  printf("Lees-Edwards parameters at step %d are "
		 "(location_velocity_displacement):\n", (get_step()-1));
	  for(i=0; i < N_LE_total; i++){
	    printf("\tLees_Edwards %d_%lg_%lg\n", LeesEdw[i].loc,
		   (double) LeesEdw[i].vel, (double) LeesEdw[i].disp);
	  }

	}
    }
  
  free(displ);

#else /* Serial */

  int i;

  /* No walls, no output */
  if (N_LE_total == 0){
    return;
  }

  printf("Lees Edwards parameters at step %d are "
	 "(location_velocity_displacement):\n", get_step());
  for(i=0; i< N_LE_total; i++){
    printf("\tLees_Edwards %d_%lg_%lg\n",LeesEdw_plane[i].loc,
	   (double)LeesEdw_plane[i].vel,(double)LeesEdw_plane[i].disp);
  }
#endif /* _MPI_ */
}

/*----------------------------------------------------------------------------*/
/*!
 * Comparison function for qsort() (see sorting of LE planes in LE_init())
 * This function return an integer less than, equal to, or greater than zero
 * if the first argument is considered to be respectively less than, equal to,
 * or greater than the second.
 *
 *- \c Options:   none
 *- \c Arguments:
 *  -# \c void \c *le1: first LE wall for comparison
 *  -# \c void \c *le2: second LE wall for comparison
 *- \c Returns:   int: see description above (1, -1, or 0)
 *- \c Buffers:   no dependence
 *- \c Version:   2.0
 *- \c Last \c updated: 14/01/2003 by JCD
 *- \C Authors:   JC Desplat
 *- \c See \c also: LE_init()
 *- \c Note:      none
 */
/*----------------------------------------------------------------------------*/

int LE_cmpLEBC(const void *le1, const void *le2)
{
  if (((LE_Plane *)le1)->loc > ((LE_Plane *)le2)->loc){
    return (1);
  }
  if (((LE_Plane *)le1)->loc < ((LE_Plane *)le2)->loc){
    return (-1);
  }
  return (0);
}


/*--------------------------------------------------------------------------*/
/* Utility routines for debugging */
void LE_print_LEbuffers()
{
  int i,j,k,vel,gind,ind,LE_loc;
  FILE *fp;
  char fname[256];
  int N[3];
  int offset[3];

  get_N_local(N);
  get_N_offset(offset);

#ifdef _MPI_
  sprintf(fname, "LEbuffs_%6.6d.%2.2d", get_step(), cart_rank());
#else
  sprintf(fname, "LEbuffs_%6.6d", get_step());
#endif
  fp = fopen(fname,"w");

  /* Start with phi (excluding halos) */
  for(i=0; i< N_LE_plane; i++){
    LE_loc = LeesEdw_plane[i].loc;
    for(j=1; j<=N[Y]; j++){
      for(k=1; k<=N[Z]; k++){
	gind = (N_total(Y)+2)*(N_total(Z)+2)*(LE_loc+offset[X])
	  + (N_total(Z)+2)*(j+offset[Y]) 
	  + (k+offset[Z]);
	ind = (N[Y]+2)*(N[Z]+2)*(2*i) + (N[Z]+2)*j + k;
	fprintf(fp, "DBG: LE_PHI STEP %5.5d buff[%6.6d] = %lg\n",
	       get_step(), gind,LeesEdw_phi[ind]);fflush(NULL);
	gind = (N_total(Y)+2)*(N_total(Z)+2)*(LE_loc+1+offset[X])
	  + (N_total(Z)+2)*(j+offset[Y]) 
	  + (k+offset[Z]);
	ind = (N[Y]+2)*(N[Z]+2)*(2*i+1) + (N[Z]+2)*j + k;
	fprintf(fp, "DBG: LE_PHI STEP %5.5d buff[%6.6d] = %lg\n",
	       get_step(), gind, LeesEdw_phi[ind]);
	fflush(NULL);
      }
    }
  }

  /* Then, deal with sites (still excluding halos) */
  for(i=0; i< N_LE_plane; i++){
    LE_loc = LeesEdw_plane[i].loc;
    for(j=1; j<=N[Y]; j++){
      for(k=1; k<=N[Z]; k++){
	for(vel=0; vel<2*LE_N_VEL_XING; vel++){
	  gind = (N_total(Y)+2)*(N_total(Z)+2)*(LE_loc+offset[X])
	    + (N_total(Z)+2)*(j+offset[Y]) 
	    + (k+offset[Z]);
	  ind = (N[Y]+2)*(N[Z]+2)*(2*i)*LE_N_VEL_XING*2 
	    + (N[Z]+2)*j*LE_N_VEL_XING*2
	    + k*LE_N_VEL_XING*2
	    + vel;
	  fprintf(fp, "DBG: LE_SITE STEP %5.5d buff[%6.6d_%d] = %lg\n",
		 get_step(), gind,vel,LeesEdw_site[ind]);fflush(NULL);
	  gind = (N_total(Y)+2)*(N_total(Z)+2)*(LE_loc+1+offset[X])
	    + (N_total(Z)+2)*(j+offset[Y]) 
	    + (k+offset[Z]);
	  ind = (N[Y]+2)*(N[Z]+2)*(2*i+1)*LE_N_VEL_XING*2 
	    + (N[Z]+2)*j*LE_N_VEL_XING*2
	    + k*LE_N_VEL_XING*2
	    + vel;
	  fprintf(fp, "DBG: LE_SITE STEP %5.5d buff[%6.6d_%d] = %lg\n",
		 get_step(), gind, vel, LeesEdw_site[ind]);
	  fflush(NULL);
	}
      }
    }
  }
  
  fclose(fp);
}

/*----------------------------------------------------------------------------*/
/*!
 * Computes and store gradients of phi
 *
 *- \c Options:   _TRACE_
 *- \c Arguments: void
 *- \c Returns:   void
 *- \c Buffers:   uses .phi
 *- \c Version:   2.0
 *- \c Last \c updated: 27/01/2002 by JCD
 *- \c Authors:   P. Bladon and JC Desplat
 *- \c See \c also: MODEL_calc_phi(), LE_apply_LEBC()
 *- \c Note:      this routine is suitable for all systems, including those with
 *                Lees-Edwards walls. Note that values will be computed for all
 *                sites, including solid sites (for which obviously the values
 *                stored have no meaning)
 */
/*----------------------------------------------------------------------------*/

void MODEL_get_gradients( void )
{
  int     i,i0,i1,i2,j,k,plane,ind,ind0,xfac,yfac,LE_loc;
  double   f1,f2,phi[9];
  double delsq_phi, grad_phi[3];
  int     N[3];

  get_N_local(N);
  xfac = (N[Y]+2)*(N[Z]+2);
  yfac = (N[Z]+2);
  f1 = 1.0/18.0;
  f2 = 1.0/9.0;

  /* Calculate phi everywhere */
  /* KS done in collision MODEL_calc_phi(); */
  assert(nhalo_ == 1);

  /* WARNING: phi_site[] must be up-to-date! */

  /* First, compute gradients in the bulk (away from LE planes) */
  for(plane=0; plane<N_LE_plane; plane++) {
	  
    i0 = LeesEdw_plane[plane].loc+1;
    i1 = LeesEdw_plane[(plane+1)%N_LE_plane].loc-1;
	  
    if(plane == (N_LE_plane-1)){
      i1 += N[X];
    }
    for(i=i0; i<i1; i++) {
      i2 = (i%N[X])+1;
      for(j=1; j<=N[Y]; j++)
	for(k=1; k<=N[Z]; k++) {
	  ind = i2*xfac + j*yfac + k;
	  grad_phi[X] = 
	    f1*(phi_site[ind+xfac       ]-phi_site[ind-xfac       ] +
		phi_site[ind+xfac+yfac+1]-phi_site[ind-xfac+yfac+1] +
		phi_site[ind+xfac-yfac+1]-phi_site[ind-xfac-yfac+1] +
		phi_site[ind+xfac+yfac-1]-phi_site[ind-xfac+yfac-1] +
		phi_site[ind+xfac-yfac-1]-phi_site[ind-xfac-yfac-1] +
		phi_site[ind+xfac+yfac  ]-phi_site[ind-xfac+yfac  ] +
		phi_site[ind+xfac-yfac  ]-phi_site[ind-xfac-yfac  ] +
		phi_site[ind+xfac     +1]-phi_site[ind-xfac     +1] +
		phi_site[ind+xfac     -1]-phi_site[ind-xfac     -1]);
		    
	  grad_phi[Y] = 
	    f1*(phi_site[ind     +yfac  ]-phi_site[ind     -yfac  ] +
		phi_site[ind+xfac+yfac+1]-phi_site[ind+xfac-yfac+1] +
		phi_site[ind-xfac+yfac+1]-phi_site[ind-xfac-yfac+1] +
		phi_site[ind+xfac+yfac-1]-phi_site[ind+xfac-yfac-1] +
		phi_site[ind-xfac+yfac-1]-phi_site[ind-xfac-yfac-1] +
		phi_site[ind+xfac+yfac  ]-phi_site[ind+xfac-yfac  ] +
		phi_site[ind-xfac+yfac  ]-phi_site[ind-xfac-yfac  ] +
		phi_site[ind     +yfac+1]-phi_site[ind     -yfac+1] +
		phi_site[ind     +yfac-1]-phi_site[ind     -yfac-1]);
		    
	  grad_phi[Z] = 
	    f1*(phi_site[ind          +1]-phi_site[ind          -1] +
		phi_site[ind+xfac+yfac+1]-phi_site[ind+xfac+yfac-1] +
		phi_site[ind-xfac+yfac+1]-phi_site[ind-xfac+yfac-1] +
		phi_site[ind+xfac-yfac+1]-phi_site[ind+xfac-yfac-1] +
		phi_site[ind-xfac-yfac+1]-phi_site[ind-xfac-yfac-1] +
		phi_site[ind+xfac     +1]-phi_site[ind+xfac     -1] +
		phi_site[ind-xfac     +1]-phi_site[ind-xfac     -1] +
		phi_site[ind     +yfac+1]-phi_site[ind     +yfac-1] +
		phi_site[ind     -yfac+1]-phi_site[ind     -yfac-1]);
		    
	  delsq_phi      = f2*(phi_site[ind+xfac       ] + 
			       phi_site[ind-xfac       ] +
			       phi_site[ind     +yfac  ] + 
			       phi_site[ind     -yfac  ] +
			       phi_site[ind          +1] + 
			       phi_site[ind          -1] +
			       phi_site[ind+xfac+yfac+1] + 
			       phi_site[ind+xfac+yfac-1] + 
			       phi_site[ind+xfac-yfac+1] + 
			       phi_site[ind+xfac-yfac-1] + 
			       phi_site[ind-xfac+yfac+1] + 
			       phi_site[ind-xfac+yfac-1] + 
			       phi_site[ind-xfac-yfac+1] + 
			       phi_site[ind-xfac-yfac-1] +
			       phi_site[ind+xfac+yfac  ] + 
			       phi_site[ind+xfac-yfac  ] + 
			       phi_site[ind-xfac+yfac  ] + 
			       phi_site[ind-xfac-yfac  ] + 
			       phi_site[ind+xfac     +1] + 
			       phi_site[ind+xfac     -1] + 
			       phi_site[ind-xfac     +1] + 
			       phi_site[ind-xfac     -1] +
			       phi_site[ind     +yfac+1] + 
			       phi_site[ind     +yfac-1] + 
			       phi_site[ind     -yfac+1] + 
			       phi_site[ind     -yfac-1] -
			       26.0*phi_site[ind]);
	  phi_set_grad_phi_site(ind, grad_phi);
	  phi_set_delsq_phi_site(ind, delsq_phi);
	}
    }
  }
      
  /* Make sure LE buffers for interpolation are up-to-date */
  LE_update_buffers( PHI_ONLY );

  /* WARNING: LeesEdw_phi[] must be up-to-date! */
      
  for(plane=0; plane<N_LE_plane; plane++)
    {
      LE_loc = LeesEdw_plane[plane].loc;
	  
      /* Second, compute gradients from plane `above' LE plane */
      /* Crossing LE plane going down (X-1): +ve displacement */
      for(j=1; j<=N[Y]; j++)
	for(k=1; k<=N[Z]; k++)
	  {
	    ind  = (LE_loc+1)*xfac + j*yfac + k;
	    ind0 = 2*plane*xfac + j*yfac + k;
		  
	    phi[0] =  LeesEdw_phi[ind0-yfac-1];
	    phi[1] =  LeesEdw_phi[ind0-yfac  ];
	    phi[2] =  LeesEdw_phi[ind0-yfac+1];
	    phi[3] =  LeesEdw_phi[ind0     -1];
	    phi[4] =  LeesEdw_phi[ind0       ];
	    phi[5] =  LeesEdw_phi[ind0     +1];
	    phi[6] =  LeesEdw_phi[ind0+yfac-1];
	    phi[7] =  LeesEdw_phi[ind0+yfac  ];
	    phi[8] =  LeesEdw_phi[ind0+yfac+1];
		  
	    grad_phi[X] = 
	      f1*(phi_site[ind+xfac-yfac-1] + phi_site[ind+xfac-yfac  ] +
		  phi_site[ind+xfac-yfac+1] + phi_site[ind+xfac     -1] +
		  phi_site[ind+xfac       ] + phi_site[ind+xfac     +1] +
		  phi_site[ind+xfac+yfac-1] + phi_site[ind+xfac+yfac  ] +
		  phi_site[ind+xfac+yfac+1] -
		  (phi[0]+phi[1]+phi[2]+phi[3]+phi[4]+phi[5]+phi[6]+phi[7]+
		   phi[8]));
		  
	    grad_phi[Y] = 
	      f1*(phi_site[ind     +yfac  ] - phi_site[ind     -yfac  ] +
		  phi_site[ind+xfac+yfac+1] - phi_site[ind+xfac-yfac+1] + 
		  phi[6]                    - phi[0]                    + 
		  phi_site[ind+xfac+yfac-1] - phi_site[ind+xfac-yfac-1] + 
		  phi[7]                    - phi[1]                    +
		  phi_site[ind+xfac+yfac  ] - phi_site[ind+xfac-yfac  ] + 
		  phi[8]                    - phi[2]                    + 
		  phi_site[ind     +yfac+1] - phi_site[ind     -yfac+1] + 
		  phi_site[ind     +yfac-1] - phi_site[ind     -yfac-1]);
		  
	    grad_phi[Z] = 
	      f1*(phi_site[ind          +1] - phi_site[ind          -1] +
		  phi_site[ind+xfac+yfac+1] - phi_site[ind+xfac+yfac-1] + 
		  phi[2]                    - phi[0]                    + 
		  phi_site[ind+xfac-yfac+1] - phi_site[ind+xfac-yfac-1] + 
		  phi[5]                    - phi[3]                    +
		  phi_site[ind+xfac     +1] - phi_site[ind+xfac     -1] + 
		  phi[8]                    - phi[6]                    + 
		  phi_site[ind     +yfac+1] - phi_site[ind     +yfac-1] + 
		  phi_site[ind     -yfac+1] - phi_site[ind     -yfac-1]); 
		  
	    delsq_phi = f2*(phi[0]+phi[1]+phi[2]+phi[3]+phi[4]+phi[5]
				 +phi[6]+phi[7]+phi[8]+
				 phi_site[ind     -yfac-1] + 
				 phi_site[ind     -yfac  ] + 
				 phi_site[ind     -yfac+1] + 
				 phi_site[ind          -1] + 
				 phi_site[ind          +1] +
				 phi_site[ind     +yfac-1] + 
				 phi_site[ind     +yfac  ] + 
				 phi_site[ind     +yfac+1] + 
				 phi_site[ind+xfac-yfac-1] + 
				 phi_site[ind+xfac-yfac  ] + 
				 phi_site[ind+xfac-yfac+1] + 
				 phi_site[ind+xfac     -1] + 
				 phi_site[ind+xfac       ] +
				 phi_site[ind+xfac     +1] + 
				 phi_site[ind+xfac+yfac-1] + 
				 phi_site[ind+xfac+yfac  ] + 
				 phi_site[ind+xfac+yfac+1] -
				 26.0*phi_site[ind]);
	  phi_set_grad_phi_site(ind, grad_phi);
	  phi_set_delsq_phi_site(ind, delsq_phi);
	  }
	  
      /* Last, compute gradients from plane `beneath' LE plane */
      /* Crossing LE plane going up (X+1): -ve displacement */
      for(j=1; j<=N[Y]; j++)
	for(k=1; k<=N[Z]; k++)
	  {
	    ind  =      LE_loc*xfac + j*yfac + k;
	    ind0 = (2*plane+1)*xfac + j*yfac + k;
		  
	    phi[0] =  LeesEdw_phi[ind0-yfac-1];
	    phi[1] =  LeesEdw_phi[ind0-yfac  ];
	    phi[2] =  LeesEdw_phi[ind0-yfac+1];
	    phi[3] =  LeesEdw_phi[ind0     -1];
	    phi[4] =  LeesEdw_phi[ind0       ];
	    phi[5] =  LeesEdw_phi[ind0     +1];
	    phi[6] =  LeesEdw_phi[ind0+yfac-1];
	    phi[7] =  LeesEdw_phi[ind0+yfac  ];
	    phi[8] =  LeesEdw_phi[ind0+yfac+1];
		  
	    grad_phi[X] = 
	      f1*(phi[0]+phi[1]+phi[2]+phi[3]+phi[4]+phi[5]+phi[6]+phi[7]+
		  phi[8]-
		  (phi_site[ind-xfac-yfac-1] + phi_site[ind-xfac-yfac  ] +
		   phi_site[ind-xfac-yfac+1] + phi_site[ind-xfac     -1] +
		   phi_site[ind-xfac       ] + phi_site[ind-xfac     +1] +
		   phi_site[ind-xfac+yfac-1] + phi_site[ind-xfac+yfac  ] +
		   phi_site[ind-xfac+yfac+1]));
		
	    grad_phi[Y] = 
	      f1*(phi_site[ind     +yfac  ] - phi_site[ind     -yfac  ] +
		  phi[6]                    - phi[0]                    + 
		  phi_site[ind-xfac+yfac+1] - phi_site[ind-xfac-yfac+1] + 
		  phi[7]                    - phi[1]                    + 
		  phi_site[ind-xfac+yfac-1] - phi_site[ind-xfac-yfac-1] +
		  phi[8]                    - phi[2]                    + 
		  phi_site[ind-xfac+yfac  ] - phi_site[ind-xfac-yfac  ] + 
		  phi_site[ind     +yfac+1] - phi_site[ind     -yfac+1] + 
		  phi_site[ind     +yfac-1] - phi_site[ind     -yfac-1]);
		  
	    grad_phi[Z] = 
	      f1*(phi_site[ind          +1] - phi_site[ind          -1] +
		  phi[2]                    - phi[0]                    + 
		  phi_site[ind-xfac+yfac+1] - phi_site[ind-xfac+yfac-1] + 
		  phi[5]                    - phi[3]                    + 
		  phi_site[ind-xfac-yfac+1] - phi_site[ind-xfac-yfac-1] +
		  phi[8]                    - phi[6]                    +
		  phi_site[ind-xfac     +1] - phi_site[ind-xfac     -1] + 
		  phi_site[ind     +yfac+1] - phi_site[ind     +yfac-1] + 
		  phi_site[ind     -yfac+1] - phi_site[ind     -yfac-1]); 
		  
	    delsq_phi = f2*(phi[0]+phi[1]+phi[2]+phi[3]+phi[4]+
				 phi[5]+
				 phi[6]+phi[7]+phi[8]+
				 phi_site[ind     -yfac-1] + 
				 phi_site[ind     -yfac  ] + 
				 phi_site[ind     -yfac+1] + 
				 phi_site[ind          -1] + 
				 phi_site[ind          +1] +
				 phi_site[ind     +yfac-1] + 
				 phi_site[ind     +yfac  ] + 
				 phi_site[ind     +yfac+1] + 
				 phi_site[ind-xfac-yfac-1] + 
				 phi_site[ind-xfac-yfac  ] + 
				 phi_site[ind-xfac-yfac+1] + 
				 phi_site[ind-xfac     -1] + 
				 phi_site[ind-xfac       ] +
				 phi_site[ind-xfac     +1] + 
				 phi_site[ind-xfac+yfac-1] + 
				 phi_site[ind-xfac+yfac  ] + 
				 phi_site[ind-xfac+yfac+1] -
				 26.0*phi_site[ind]);
	  phi_set_grad_phi_site(ind, grad_phi);
	  phi_set_delsq_phi_site(ind, delsq_phi);
	  }
    }


  if(N_LE_plane == 0)
    {

      for(i=1; i<=N[X]; i++)
	for(j=1; j<=N[Y]; j++)
	  for(k=1; k<=N[Z]; k++)
	    {
	      ind = i*xfac + j*yfac + k;
	      
	      grad_phi[X] = 
		f1*(phi_site[ind+xfac       ] - phi_site[ind-xfac       ] +
		    phi_site[ind+xfac+yfac+1] - phi_site[ind-xfac+yfac+1] +
		    phi_site[ind+xfac-yfac+1] - phi_site[ind-xfac-yfac+1] + 
		    phi_site[ind+xfac+yfac-1] - phi_site[ind-xfac+yfac-1] + 
		    phi_site[ind+xfac-yfac-1] - phi_site[ind-xfac-yfac-1] +
		    phi_site[ind+xfac+yfac  ] - phi_site[ind-xfac+yfac  ] +
		    phi_site[ind+xfac-yfac  ] - phi_site[ind-xfac-yfac  ] +
		    phi_site[ind+xfac     +1] - phi_site[ind-xfac     +1] +
		    phi_site[ind+xfac     -1] - phi_site[ind-xfac     -1]);
	      
	      grad_phi[Y] = 
		f1*(phi_site[ind     +yfac  ] - phi_site[ind     -yfac  ] +
		    phi_site[ind+xfac+yfac+1] - phi_site[ind+xfac-yfac+1] + 
		    phi_site[ind-xfac+yfac+1] - phi_site[ind-xfac-yfac+1] + 
		    phi_site[ind+xfac+yfac-1] - phi_site[ind+xfac-yfac-1] + 
		    phi_site[ind-xfac+yfac-1] - phi_site[ind-xfac-yfac-1] +
		    phi_site[ind+xfac+yfac  ] - phi_site[ind+xfac-yfac  ] + 
		    phi_site[ind-xfac+yfac  ] - phi_site[ind-xfac-yfac  ] + 
		    phi_site[ind     +yfac+1] - phi_site[ind     -yfac+1] + 
		    phi_site[ind     +yfac-1] - phi_site[ind     -yfac-1]);
	      
	      grad_phi[Z] = 
		f1*(phi_site[ind          +1] - phi_site[ind          -1] +
		    phi_site[ind+xfac+yfac+1] - phi_site[ind+xfac+yfac-1] + 
		    phi_site[ind-xfac+yfac+1] - phi_site[ind-xfac+yfac-1] + 
		    phi_site[ind+xfac-yfac+1] - phi_site[ind+xfac-yfac-1] + 
		    phi_site[ind-xfac-yfac+1] - phi_site[ind-xfac-yfac-1] +
		    phi_site[ind+xfac     +1] - phi_site[ind+xfac     -1] + 
		    phi_site[ind-xfac     +1] - phi_site[ind-xfac     -1] + 
		    phi_site[ind     +yfac+1] - phi_site[ind     +yfac-1] + 
		    phi_site[ind     -yfac+1] - phi_site[ind     -yfac-1]); 
	      
	      delsq_phi = f2*(phi_site[ind+xfac       ] + 
				   phi_site[ind-xfac       ] +
				   phi_site[ind     +yfac  ] + 
				   phi_site[ind     -yfac  ] +
				   phi_site[ind          +1] + 
				   phi_site[ind          -1] +
				   phi_site[ind+xfac+yfac+1] + 
				   phi_site[ind+xfac+yfac-1] + 
				   phi_site[ind+xfac-yfac+1] + 
				   phi_site[ind+xfac-yfac-1] + 
				   phi_site[ind-xfac+yfac+1] + 
				   phi_site[ind-xfac+yfac-1] + 
				   phi_site[ind-xfac-yfac+1] + 
				   phi_site[ind-xfac-yfac-1] +
				   phi_site[ind+xfac+yfac  ] + 
				   phi_site[ind+xfac-yfac  ] + 
				   phi_site[ind-xfac+yfac  ] + 
				   phi_site[ind-xfac-yfac  ] + 
				   phi_site[ind+xfac     +1] + 
				   phi_site[ind+xfac     -1] + 
				   phi_site[ind-xfac     +1] + 
				   phi_site[ind-xfac     -1] +
				   phi_site[ind     +yfac+1] + 
				   phi_site[ind     +yfac-1] + 
				   phi_site[ind     -yfac+1] + 
				   phi_site[ind     -yfac-1] -
				   26.0*phi_site[ind]);
	  phi_set_grad_phi_site(ind, grad_phi);
	  phi_set_delsq_phi_site(ind, delsq_phi);
	    }
    }

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
