#include "globals.h"

#include "pe.h"
#include "timer.h"
#include "coords.h"
#include "cartesian.h"

static void COM_halo_rho( void );
static void COM_read_input_file( char * );
static void COM_process_input_file( Input_Data *, Input_Data * );
static void COM_process_options(Input_Data *);

/* MPI-specific functions and variables */
#ifdef _MPI_

static IVector COM_decomp( int );
static void    COM_Add_FVector( FVector *, FVector *, int *, MPI_Datatype * );

static MPI_Datatype DT_plane_XY; /* MPI datatype defining XY plane */
static MPI_Datatype DT_plane_XZ; /* MPI datatype defining XZ plane */
static MPI_Datatype DT_plane_YZ; /* MPI datatype defining YZ plane */
static MPI_Datatype DT_Float_plane_XY;/* MPI Datatype: XY plane of Floats */
static MPI_Datatype DT_Float_plane_XZ;/* MPI Datatype: XZ plane of Floats */
static MPI_Datatype DT_Float_plane_YZ;/* MPI Datatype: YZ plane of Floats */

MPI_Comm     IO_Comm;     /* Communicator for parallel IO groups */

MPI_Datatype DT_FVector;  /* MPI Datatype for type FVector */
MPI_Datatype DT_Site;     /* MPI Datatype for type Site */
MPI_Op       OP_AddFVector;     /* (user-def) MPI operator to add FVectors */


#endif /* _MPI_ */

IO_Param     io_grp;      /* Parameters of current IO group */


/*---------------------------------------------------------------------------*\
 * void COM_halo( void )                                                     * 
 *                                                                           *
 * Halos data to neighbouring PEs -- or if serial code, to its periodic      *
 * image                                                                     *
 *                                                                           *
 * Version: 2.0                                                              *
 * Options: _MPI_, _TRACE_                                                   *
 *                                                                           *
 * Last Updated: 04/01/2002 by JCD                                           *
\*---------------------------------------------------------------------------*/

void COM_halo( void )
{
  int i, j, k;
  int  xfac, yfac;
  int N[3];

#ifdef _MPI_

  int nyz2, nxy2z2;

  MPI_Request req[4];
  MPI_Status status[4];

  LUDWIG_ENTER("COM_halo()");

  TIMER_start(TIMER_HALO_LATTICE);

  get_N_local(N);
  yfac = N[Z]+2;
  nyz2 = N[Y]*yfac;
  xfac = (N[Y]+2)*yfac;
  nxy2z2 = N[X]*xfac;
  
  /* Swap (YZ) plane: one contiguous block in memory */
  
  /* Particular case where source_PE == destination_PE (although MPI should
     cope with it by itself) */

  if(cart_size(X) == 1) {
    for(j=0;j<=N[Y]+1;j++)
      for(k=0;k<=N[Z]+1;k++) {
	site[                j*yfac + k] = site[N[X]*xfac + j*yfac + k];
	site[(N[X]+1)*xfac + j*yfac + k] = site[     xfac + j*yfac + k];
      }
  }
  else {   
    MPI_Issend(&site[xfac].f[0], 1, DT_plane_YZ, cart_neighb(BACKWARD,X),
	       TAG_HALO_SWP_X_BWD, cart_comm(), &req[0]);
    MPI_Irecv(&site[(N[X]+1)*xfac].f[0], 1, DT_plane_YZ,
	      cart_neighb(FORWARD,X),
	      TAG_HALO_SWP_X_BWD, cart_comm(), &req[1]);
    MPI_Issend(&site[nxy2z2].f[0], 1, DT_plane_YZ, cart_neighb(FORWARD,X),
	       TAG_HALO_SWP_X_FWD, cart_comm(), &req[2]);
    MPI_Irecv(&site[0].f[0], 1, DT_plane_YZ, cart_neighb(BACKWARD,X),
	      TAG_HALO_SWP_X_FWD, cart_comm(), &req[3]);
    MPI_Waitall(4,req,status); /* Necessary to propagate corners */
  }
  
  /* Swap (XZ) plane: N[X]+2 blocks of N[Z]+2 sites (stride=(N[Y]+2)*(N[Z]+2)) */

  if(cart_size(Y) == 1)
    {
      for(i=0;i<=N[X]+1;i++)
	for(k=0;k<=N[Z]+1;k++)
	  {
	    site[i*xfac                 + k] = site[i*xfac + N[Y]*yfac + k];
	    site[i*xfac + (N[Y]+1)*yfac + k] = site[i*xfac +      yfac + k];
	  }
    }
  else
    {
      MPI_Issend(&site[yfac].f[0], 1, DT_plane_XZ,
		 cart_neighb(BACKWARD,Y), TAG_HALO_SWP_Y_BWD, cart_comm(),
		 &req[0]);
      MPI_Irecv(&site[(N[Y]+1)*yfac].f[0], 1, DT_plane_XZ,
		cart_neighb(FORWARD,Y), TAG_HALO_SWP_Y_BWD, cart_comm(),
		&req[1]);
      MPI_Issend(&site[nyz2].f[0], 1, DT_plane_XZ, cart_neighb(FORWARD,Y),
		 TAG_HALO_SWP_Y_FWD, cart_comm(), &req[2]);
      MPI_Irecv(&site[0].f[0], 1, DT_plane_XZ, cart_neighb(BACKWARD,Y),
		TAG_HALO_SWP_Y_FWD, cart_comm(), &req[3]);
      MPI_Waitall(4,req,status); /* Necessary to propagate corners */
    }
  
  /* Swap (XY) plane: (N[X]+2)*(N[Y]+2) blocks of 1 site (stride=N[Z]+2) */
  if(cart_size(Z) == 1)
    {
      for(i=0;i<=N[X]+1;i++)
	for(j=0;j<=N[Y]+1;j++)
	  {
	    site[i*xfac + j*yfac           ] = site[i*xfac + j*yfac + N[Z]];
	    site[i*xfac + j*yfac + N[Z] + 1] = site[i*xfac + j*yfac + 1   ];
	  }
    }
  else
    {
      MPI_Issend(site[1].f, 1, DT_plane_XY, cart_neighb(BACKWARD,Z),
		 TAG_HALO_SWP_Z_BWD, cart_comm(), &req[0]);
      MPI_Irecv(site[N[Z]+1].f, 1, DT_plane_XY, cart_neighb(FORWARD,Z),
		TAG_HALO_SWP_Z_BWD, cart_comm(), &req[1]);
      MPI_Issend(site[N[Z]].f, 1, DT_plane_XY, cart_neighb(FORWARD,Z),
		 TAG_HALO_SWP_Z_FWD, cart_comm(), &req[2]);  
      MPI_Irecv(site[0].f, 1, DT_plane_XY, cart_neighb(BACKWARD,Z),
		TAG_HALO_SWP_Z_FWD, cart_comm(), &req[3]);
      MPI_Waitall(4,req,status);
    }

#else /* Serial section */
  
  LUDWIG_ENTER("COM_halo()");
  TIMER_start(TIMER_HALO_LATTICE);
  
  get_N_local(N);
  yfac = N[Z]+2;
  xfac = (N[Y]+2)*yfac;
  
  for (i = 1; i <= N[X]; i++)
    for (j = 1; j <= N[Y]; j++) {
      site[i*xfac + j*yfac           ] = site[i*xfac + j*yfac + N[Z]];
      site[i*xfac + j*yfac + N[Z] + 1] = site[i*xfac + j*yfac + 1   ];
    }
    
  for (i = 1; i <= N[X]; i++)
    for (k = 0; k <= N[Z]+1; k++) {
      site[i*xfac                 + k] = site[i*xfac + N[Y]*yfac + k];
      site[i*xfac + (N[Y]+1)*yfac + k] = site[i*xfac +      yfac + k];
    }
    
  for (j = 0; j <= N[Y]+1; j++)
    for(k = 0; k <= N[Z]+1; k++) {
      site[                j*yfac + k] = site[N[X]*xfac + j*yfac + k];
      site[(N[X]+1)*xfac + j*yfac + k] = site[xfac      + j*yfac + k];
    }

#endif /* _MPI_ */

  TIMER_stop(TIMER_HALO_LATTICE);

  return;
}

/*---------------------------------------------------------------------------*\
 * void COM_halo_rho( void )                                                 *
 *                                                                           *
 * Halos rho_site to neighbouring PEs -- or if serial code, to its periodic  *
 * image                                                                     *
 *                                                                           *
 * Version: 2.0                                                              *
 * Options: _MPI_, _TRACE_                                                   *
 *                                                                           *
 * Last Updated: 04/01/2002 by JCD                                           *
\*---------------------------------------------------------------------------*/

void COM_halo_rho( void )
{
  int i, j, k;
  int xfac, yfac;
  int N[3];

  extern double * rho_site;
  
#ifdef _MPI_
  int nyz2, nxy2z2;

  MPI_Request req[4];
  MPI_Status status[4];
  
  LUDWIG_ENTER("COM_halo_rho()");
  TIMER_start(TIMER_HALO_LATTICE);

  get_N_local(N);
  yfac = N[Z]+2;
  nyz2 = N[Y]*yfac;
  xfac = (N[Y]+2)*yfac;
  nxy2z2 = N[X]*xfac;
  
  /* Swap (YZ) plane: one contiguous block in memory */
  
  /* Particular case where source_PE == destination_PE (although MPI would */
  /* cope with it by itself) */
  if(cart_size(X) == 1)
    {
      for(j=0;j<=N[Y]+1;j++)
	for(k=0;k<=N[Z]+1;k++)
	  {
	    rho_site[             j*yfac+k] = rho_site[N[X]*xfac+j*yfac+k];
	    rho_site[(N[X]+1)*xfac+j*yfac+k] = rho_site[xfac    +j*yfac+k];
	  }
    }
  else
    {   
      MPI_Issend(&rho_site[xfac], 1, DT_Float_plane_YZ,
		 cart_neighb(BACKWARD,X), TAG_HALO_SWP_X_BWD, cart_comm(),
		 &req[0]);
      MPI_Irecv(&rho_site[(N[X]+1)*xfac], 1, DT_Float_plane_YZ,
		cart_neighb(FORWARD,X), TAG_HALO_SWP_X_BWD, cart_comm(),
		&req[1]);
      MPI_Issend(&rho_site[nxy2z2], 1, DT_Float_plane_YZ,
		 cart_neighb(FORWARD,X), TAG_HALO_SWP_X_FWD, cart_comm(),
		 &req[2]);
      MPI_Irecv(&rho_site[0], 1, DT_Float_plane_YZ,
		cart_neighb(BACKWARD,X), TAG_HALO_SWP_X_FWD, cart_comm(),
		&req[3]);
      MPI_Waitall(4,req,status); /* Necessary to propagate corners */
    }
  
  /* Swap (XZ) plane: N[X]+2 blocks of N[Z]+2 sites (stride=(N[Y]+2)*(N[Z]+2)) */
  if(cart_size(Y) == 1)
    {
      for(i=0;i<=N[X]+1;i++)
	for(k=0;k<=N[Z]+1;k++)
	  {
	    rho_site[i*xfac             +k] = rho_site[i*xfac+N[Y]*yfac+k];
	    rho_site[i*xfac+(N[Y]+1)*yfac+k] = rho_site[i*xfac+    yfac+k];
	  }
    }
  else
    {
      MPI_Issend(&rho_site[yfac], 1, DT_Float_plane_XZ,
		 cart_neighb(BACKWARD,Y), TAG_HALO_SWP_Y_BWD, cart_comm(),
		 &req[0]);
      MPI_Irecv(&rho_site[(N[Y]+1)*yfac], 1, DT_Float_plane_XZ,
		cart_neighb(FORWARD,Y),
		TAG_HALO_SWP_Y_BWD, cart_comm(), &req[1]);
      MPI_Issend(&rho_site[nyz2], 1, DT_Float_plane_XZ, cart_neighb(FORWARD,Y),
		 TAG_HALO_SWP_Y_FWD, cart_comm(), &req[2]);
      MPI_Irecv(&rho_site[0], 1, DT_Float_plane_XZ, cart_neighb(BACKWARD,Y),
		TAG_HALO_SWP_Y_FWD, cart_comm(), &req[3]);
      MPI_Waitall(4,req,status); /* Necessary to propagate corners */
    }
  
  /* Swap (XY) plane: (N[X]+2)*(N[Y]+2) blocks of 1 site (stride=N[Z]+2) */
  if(cart_size(Z) == 1)
    {
      for(i=0;i<=N[X]+1;i++)
	for(j=0;j<=N[Y]+1;j++)
	  {
	    rho_site[i*xfac+j*yfac      ] = rho_site[i*xfac+j*yfac+N[Z]];
	    rho_site[i*xfac+j*yfac+N[Z]+1] = rho_site[i*xfac+j*yfac+1  ];
	  }
    }
  else
    {
      MPI_Issend(&rho_site[1], 1, DT_Float_plane_XY, cart_neighb(BACKWARD,Z),
		 TAG_HALO_SWP_Z_BWD, cart_comm(), &req[0]);
      MPI_Irecv(&rho_site[N[Z]+1], 1, DT_Float_plane_XY,
		cart_neighb(FORWARD,Z),
		TAG_HALO_SWP_Z_BWD, cart_comm(), &req[1]);
      MPI_Issend(&rho_site[N[Z]], 1, DT_Float_plane_XY, cart_neighb(FORWARD,Z),
		 TAG_HALO_SWP_Z_FWD, cart_comm(), &req[2]);  
      MPI_Irecv(&rho_site[0], 1, DT_Float_plane_XY, cart_neighb(BACKWARD,Z),
		TAG_HALO_SWP_Z_FWD, cart_comm(), &req[3]);
      MPI_Waitall(4,req,status);
    }
  
#else /* Serial section */
  
  LUDWIG_ENTER("COM_halo_rho()");
  TIMER_start(TIMER_HALO_LATTICE);

  get_N_local(N);
  yfac = N[Z]+2;
  xfac = (N[Y]+2)*yfac;
  
  for(i=1;i<=N[X];i++)
    for(j=1;j<=N[Y];j++) {
      rho_site[i*xfac + j*yfac          ] = rho_site[i*xfac + j*yfac + N[Z]];
      rho_site[i*xfac + j*yfac + N[Z] + 1] = rho_site[i*xfac + j*yfac + 1  ];
    }
    
  for(i=1;i<=N[X];i++)
    for(k=0;k<=N[Z]+1;k++) {
      rho_site[i*xfac                + k] = rho_site[i*xfac + N[Y]*yfac + k];
      rho_site[i*xfac + (N[Y]+1)*yfac + k] = rho_site[i*xfac +     yfac + k];
    }
    
  for(j=0;j<=N[Y]+1;j++)
    for(k=0;k<=N[Z]+1;k++) {
      rho_site[               j*yfac + k] = rho_site[N[X]*xfac + j*yfac + k];
      rho_site[(N[X]+1)*xfac + j*yfac + k] = rho_site[xfac     + j*yfac + k];
    }

#endif /* _MPI_ */

  TIMER_stop(TIMER_HALO_LATTICE);

  return;
}

/*---------------------------------------------------------------------------*\
 * void COM_halo_phi( void )                                                 *
 *                                                                           *
 * Halos phi_site to neighbouring PEs -- or if serial code, to its periodic  *
 * image                                                                     *
 *                                                                           *
 * Version: 2.0                                                              *
 * Options: _MPI_, _TRACE_                                                   *
 *                                                                           *
 * Last Updated: 04/01/2002 by JCD                                           *
\*---------------------------------------------------------------------------*/

void COM_halo_phi( void )
{
  int i, j, k;
  int xfac, yfac;
  int N[3];

  extern double * phi_site;
  
#ifdef _MPI_
  int nyz2, nxy2z2;

  MPI_Request req[4];
  MPI_Status status[4];
  
  LUDWIG_ENTER("COM_halo_phi()");
  TIMER_start(TIMER_HALO_LATTICE);

  get_N_local(N);
  yfac = N[Z]+2;
  nyz2 = N[Y]*yfac;
  xfac = (N[Y]+2)*yfac;
  nxy2z2 = N[X]*xfac;
  
  /* Swap (YZ) plane: one contiguous block in memory */
  
  /* Particular case where source_PE == destination_PE (although MPI would */
  /* cope with it by itself) */
  if(cart_size(X) == 1)
    {
      for(j=0;j<=N[Y]+1;j++)
	for(k=0;k<=N[Z]+1;k++)
	  {
	    phi_site[             j*yfac+k] = phi_site[N[X]*xfac+j*yfac+k];
	    phi_site[(N[X]+1)*xfac+j*yfac+k] = phi_site[xfac    +j*yfac+k];
	  }
    }
  else
    {   
      MPI_Issend(&phi_site[xfac], 1, DT_Float_plane_YZ,
		 cart_neighb(BACKWARD,X),
		 TAG_HALO_SWP_X_BWD, cart_comm(), &req[0]);
      MPI_Irecv(&phi_site[(N[X]+1)*xfac], 1, DT_Float_plane_YZ,
		cart_neighb(FORWARD,X),
		TAG_HALO_SWP_X_BWD, cart_comm(), &req[1]);
      MPI_Issend(&phi_site[nxy2z2], 1, DT_Float_plane_YZ,
		 cart_neighb(FORWARD,X),
		 TAG_HALO_SWP_X_FWD, cart_comm(), &req[2]);
      MPI_Irecv(&phi_site[0], 1, DT_Float_plane_YZ, cart_neighb(BACKWARD,X),
		TAG_HALO_SWP_X_FWD, cart_comm(), &req[3]);
      MPI_Waitall(4,req,status); /* Necessary to propagate corners */
    }
  
  /* Swap (XZ) plane: N[X]+2 blocks of N[Z]+2 sites (stride=(N[Y]+2)*(N[Z]+2)) */
  if(cart_size(Y) == 1)
    {
      for(i=0;i<=N[X]+1;i++)
	for(k=0;k<=N[Z]+1;k++)
	  {
	    phi_site[i*xfac              +k] = phi_site[i*xfac+N[Y]*yfac+k];
	    phi_site[i*xfac+(N[Y]+1)*yfac+k] = phi_site[i*xfac+    yfac+k];
	  }
    }
  else
    {
      MPI_Issend(&phi_site[yfac], 1, DT_Float_plane_XZ,
		 cart_neighb(BACKWARD,Y),
		 TAG_HALO_SWP_Y_BWD, cart_comm(), &req[0]);
      MPI_Irecv(&phi_site[(N[Y]+1)*yfac], 1, DT_Float_plane_XZ,
		cart_neighb(FORWARD,Y),
		TAG_HALO_SWP_Y_BWD, cart_comm(), &req[1]);
      MPI_Issend(&phi_site[nyz2], 1, DT_Float_plane_XZ, cart_neighb(FORWARD,Y),
		 TAG_HALO_SWP_Y_FWD, cart_comm(), &req[2]);
      MPI_Irecv(&phi_site[0], 1, DT_Float_plane_XZ, cart_neighb(BACKWARD,Y),
		TAG_HALO_SWP_Y_FWD, cart_comm(), &req[3]);
      MPI_Waitall(4,req,status); /* Necessary to propagate corners */
    }
  
  /* Swap (XY) plane: (N[X]+2)*(N[Y]+2) blocks of 1 site (stride=N[Z]+2) */
  if(cart_size(Z) == 1)
    {
      for(i=0;i<=N[X]+1;i++)
	for(j=0;j<=N[Y]+1;j++)
	  {
	    phi_site[i*xfac+j*yfac      ] = phi_site[i*xfac+j*yfac+N[Z]];
	    phi_site[i*xfac+j*yfac+N[Z]+1] = phi_site[i*xfac+j*yfac+1  ];
	  }
    }
  else
    {
      MPI_Issend(&phi_site[1], 1, DT_Float_plane_XY, cart_neighb(BACKWARD,Z),
		 TAG_HALO_SWP_Z_BWD, cart_comm(), &req[0]);
      MPI_Irecv(&phi_site[N[Z]+1], 1, DT_Float_plane_XY,
		cart_neighb(FORWARD,Z),
		TAG_HALO_SWP_Z_BWD, cart_comm(), &req[1]);
      MPI_Issend(&phi_site[N[Z]], 1, DT_Float_plane_XY, cart_neighb(FORWARD,Z),
		 TAG_HALO_SWP_Z_FWD, cart_comm(), &req[2]);  
      MPI_Irecv(&phi_site[0], 1, DT_Float_plane_XY, cart_neighb(BACKWARD,Z),
		TAG_HALO_SWP_Z_FWD, cart_comm(), &req[3]);
      MPI_Waitall(4,req,status);
    }

#else /* Serial section (or OpenMP as appropriate) */

  LUDWIG_ENTER("COM_halo_phi()");
  TIMER_start(TIMER_HALO_LATTICE);

  get_N_local(N);
  yfac = N[Z]+2;
  xfac = (N[Y]+2)*yfac;
  
  for(i=1;i<=N[X];i++)
    for(j=1;j<=N[Y];j++) {
      phi_site[i*xfac + j*yfac           ] = phi_site[i*xfac + j*yfac + N[Z]];
      phi_site[i*xfac + j*yfac + N[Z] + 1] = phi_site[i*xfac + j*yfac + 1   ];
    }
    
  for(i=1;i<=N[X];i++)
    for(k=0;k<=N[Z]+1;k++) {
      phi_site[i*xfac                 + k] = phi_site[i*xfac + N[Y]*yfac + k];
      phi_site[i*xfac + (N[Y]+1)*yfac + k] = phi_site[i*xfac +      yfac + k];
    }
    
  for(j=0;j<=N[Y]+1;j++)
    for(k=0;k<=N[Z]+1;k++) {
      phi_site[                j*yfac + k] = phi_site[N[X]*xfac + j*yfac + k];
      phi_site[(N[X]+1)*xfac + j*yfac + k] = phi_site[xfac      + j*yfac + k];
    }
#endif /* _MPI_ */

  TIMER_stop(TIMER_HALO_LATTICE);

  return;
}


/*---------------------------------------------------------------------------*\
 * void COM_init( int argc, char **argv )                                    *
 *                                                                           *
 * Initialises communication routines                                        *
 *                                                                           *
 * Version: 2.0                                                              *
 * Options: _MPI_, _TRACE_                                                   *
 *                                                                           *
 * Arguments                                                                 *
 * - argc:    same as the main() routine's argc                              *
 * - argv:    same as the main() routine's argv                              *
 *                                                                           *
 * Last Updated: 06/01/2002 by JCD                                           *
\*---------------------------------------------------------------------------*/

void COM_init( int argc, char **argv )
{
#ifdef _MPI_ /* Parallel (MPI) section */

  int nx2, ny2, ny3, nz2, nx2y2, ny2z2, ny3z2;
  int N_sites, colour;
  int N[3];

  MPI_Aint disp[2];
  MPI_Aint size1,size2;
  MPI_Datatype DT_Global, DT_tmp, type[2];

  LUDWIG_ENTER("COM_init()");

  get_N_local(N);

  /* PE0 reads parameters from file (and set default values if needed): */
  /* this should only take place -after- MPI has been initialised (due */
  /* to restrictions on the maximum number of files which can be opened */
  /* simultaneously); root PE broadcasts parameters to other PEs */
  if( argc > 1 ){
    COM_read_input_file(argv[1]);
  }
  else{
    COM_read_input_file("input");
  }

  /* Set-up MPI datatype for structure Global */
  MPI_Type_contiguous(sizeof(Global),MPI_BYTE,&DT_Global);
  MPI_Type_commit(&DT_Global);


  MPI_Bcast(&io_grp.n_io, 1, MPI_INT, 0, MPI_COMM_WORLD);

  /* Compute extents to define new datatypes */

  nx2 = N[X]+2;
  ny2 = N[Y]+2;
  ny3 = N[Y]+3;
  nz2 = N[Z]+2;
  nx2y2 = nx2*ny2;
  ny2z2 = ny2*nz2;
  ny3z2 = ny3*nz2;
  N_sites = (N[X]+2)*(N[Y]+2)*(N[Z]+2);


  /* Set-up parallel IO parameters (rank and root) */

  io_grp.size = pe_size() / io_grp.n_io;

  if((cart_rank()%io_grp.size) == 0) {
    io_grp.root = TRUE;
  }
  else {
    io_grp.root = FALSE;
  }

  /* Set-up filename suffix for each parallel IO file */

  io_grp.file_ext = (char *) malloc(16*sizeof(char));

  if (io_grp.file_ext == NULL) fatal("malloc(io_grp.file_ext) failed\n");

  io_grp.index = cart_rank()/io_grp.size + 1;   /* Start IO indexing at 1 */
  sprintf(io_grp.file_ext, ".%d-%d", io_grp.n_io, io_grp.index);

  if (io_grp.n_io >= FOPEN_MAX) {
    verbose("Trying to open %d files (max is %d)\n", io_grp.n_io, FOPEN_MAX);
    verbose("Set MAX_IO_NODE (now %d) to a smaller value\n", MAX_IO_NODE);
    fatal("");
  }  

  /* Create communicator for each IO group, and get rank within IO group */
  MPI_Comm_split(cart_comm(), io_grp.index, cart_rank(), &IO_Comm);
  MPI_Comm_rank(IO_Comm, &io_grp.rank);




  /* Set-up MPI datatype for structure Site */
  MPI_Type_contiguous(sizeof(Site),MPI_BYTE,&DT_Site);
  MPI_Type_commit(&DT_Site);

  /* Set-up datatypes for XY, XZ and YZ planes (including halos!) */
  /* (XY) plane: (N[X]+2)*(N[Y]+2) blocks of 1 site (stride=N[Z]+2) */
  MPI_Type_vector(nx2y2,1,nz2,DT_Site,&DT_plane_XY);
  MPI_Type_commit(&DT_plane_XY);

  /* (XZ) plane: N[X]+2 blocks of N[Z]+2 sites (stride=(N[Y]+2)*(N[Z]+2)) */
  MPI_Type_hvector(nx2,nz2,ny2z2*sizeof(Site),DT_Site,&DT_plane_XZ);
  MPI_Type_commit(&DT_plane_XZ);

  /* (YZ) plane: one contiguous block of (N[Y]+2)*(N[Z]+2) sites */
  MPI_Type_contiguous(ny2z2,DT_Site,&DT_plane_YZ);
  MPI_Type_commit(&DT_plane_YZ);

  /* Set-up datatypes for XY, XZ and YZ planes of Floats (including halos) */
  /* (XY) plane: (N[X]+2)*(N[Y]+2) blocks of 1 Float (stride=N[Z]+2) */
  MPI_Type_vector(nx2y2,1,nz2, MPI_DOUBLE, &DT_Float_plane_XY);
  MPI_Type_commit(&DT_Float_plane_XY);

  /* (XZ) plane: N[X]+2 blocks of N[Z]+2 Floats (stride=(N[Y]+2)*(N[Z]+2)) */
  MPI_Type_hvector(nx2,nz2,ny2z2*sizeof(Float),MPI_DOUBLE,&DT_Float_plane_XZ);
  MPI_Type_commit(&DT_Float_plane_XZ);

  /* (YZ) plane: one contiguous block of (N[Y]+2)*(N[Z]+2) Floats */
  MPI_Type_contiguous(ny2z2, MPI_DOUBLE,&DT_Float_plane_YZ);
  MPI_Type_commit(&DT_Float_plane_YZ);

  /* Set-up MPI datatype for FVector */
  MPI_Type_contiguous(sizeof(FVector), MPI_BYTE, &DT_FVector);
  MPI_Type_commit(&DT_FVector);

  /* Initialise parallel (user-defined) operators to add FVector, IVector */
  MPI_Op_create((MPI_User_function *)COM_Add_FVector, TRUE, &OP_AddFVector);

  MPI_Barrier(cart_comm());

#else /* Serial section */

  LUDWIG_ENTER("COM_init()");

  /* Setting-up parameters */
  if( argc > 1 ){
    COM_read_input_file(argv[1]);
  }
  else{
    COM_read_input_file("input");
  }

  /* Serial definition of io_grp (used in cio) */
  io_grp.root  = TRUE;
  io_grp.n_io  = 1;
  io_grp.size  = 1;
  io_grp.index = 0;
  io_grp.rank  = 0;
  io_grp.file_ext = (char *) malloc(16*sizeof(char));
  if (io_grp.file_ext == NULL) fatal("malloc(io_grp.file_ext) failed\n");
  sprintf(io_grp.file_ext, ""); /* Nothing required in serial*/

#endif /* _MPI_ */

}

/*---------------------------------------------------------------------------*\
 * void COM_finish(  )                                                       *
 *                                                                           *
 * Close communications (if required). In practice, not much is required     *
 * besides flushing STDOUT/STDERR and calling MPI_Finalize() (for the MPI    *
 * implementation) and exiting with a suitable error code                    *
 *                                                                           *
 * Version: 2.0                                                              *
 * Options: _TRACE_, _MPI_                                                   *
 *                                                                           *
 * Arguments                                                                 *
 *                                                                           *
 * Last Updated: 06/01/2002 by JCD                                           *
\*---------------------------------------------------------------------------*/

void COM_finish( )
{

  LUDWIG_ENTER("COM_finish()");
  
#ifdef _MPI_
  
  /* Free-up MPI-specific resources */
  
  MPI_Op_free(&OP_AddFVector);

  MPI_Barrier(MPI_COMM_WORLD);
  MPI_Finalize();

#endif /* _MPI_ */

  /* Make sure STDOUT/STDERR are flushed before exiting */

  fflush(NULL);

}

/*---------------------------------------------------------------------------*\
 * void COM_read_site( char *filename, int (*func)( FILE * ) )               *
 *                                                                           *
 * Reads `something' into each lattice site from filename. The actual        *
 * reading is done by a user supplied function passed as argument            *
 *                                                                           *
 * Version: 2.0                                                              *
 * Options: _TRACE_, _MPI_                                                   *
 *                                                                           *
 * Arguments                                                                 *
 * - filename:    file from which the sites will be read from                *
 * - func:        pointer to user-supplied routine which will perform the IO *
 *                                                                           *
 * Last Updated: 06/07/2002 by JCD                                           *
\*---------------------------------------------------------------------------*/

void COM_read_site( char *filename, void (*func)( FILE * ))
{
#ifdef _MPI_ /* Parallel (MPI) version */
  
  int     i,j,k,io_size,io_index;
  int     N[3], nloc[3];
  IVector N_read;
  char    io_filename[FILENAME_MAX];
  FILE    *fp;
  long    foffset;
  MPI_Status status;
  
  LUDWIG_ENTER("COM_read_site()");

  get_N_local(N);

  /* Parallel input:  io_grp.n_io concurrent inputs will take place */
  
  /* Set correct filename from function argument and local rank */
  sprintf(io_filename, "%s%s", filename,io_grp.file_ext);
  if(io_grp.root == TRUE)
    {
      /* Open file for input */
      if( (fp = fopen(io_filename,"r")) == NULL )
	{
	  fatal("COM_read_site(): could not open data file %s\n", io_filename);
	}

      /* Ensure that simulation parameters are in the header */
      if( fscanf(fp,"%d %d %d %d %d %d %d %d\n\f",&nloc[X],&nloc[Y],&nloc[Z],
		 &N_read.x,&N_read.y,&N_read.z,&io_size,&io_index) != 8 )
	{
	  fatal("COM_read_site(): input file %s not standard\n",io_filename);
	}
      
      if( nloc[X] != N[X] || nloc[Y] != N[Y] || nloc[Z] != N[Z] ||
	  N_total(X) != N_read.x || N_total(Y) != N_read.y ||
	  N_total(Z) != N_read.z )
	  {
	    fatal("COM_read_site(): size of system mismatch\n");
	  }

      /* Check that files are compatible with current parallel IO settings */
      if( io_grp.size != io_size || io_grp.index != io_index )
	{
	  fatal("COM_read_link(): %s is not compatible current parallel IO\n",
		  io_filename);
	}
    } /* end of section for root IO PE */
    
  /* Each PE in the current IO group will read its own data in turn (by rank
     order). The use of blocking receives makes sure they don't `overtake' each
     other. Offset in file is passed using a ring communication over the local
     IO group. All IO groups access their own file concurrently (parallel IO) */
  
    /* Initiate ring communication / synchronisation within local IO group */
  else				/* i.e. non IO PE (io_grp.root==FALSE) */
    {
      /* Wait until previous PE in local group has completed its IO (use
	 blocking receives: similar mechanism to COM_read_link()) */
      MPI_Recv(&foffset, 1, MPI_LONG, io_grp.rank-1, TAG_IO, IO_Comm, &status);
      if((fp=fopen(io_filename,"r")) == NULL)
	{
	  fatal("COM_read_site(): could not open file %s\n", io_filename);
	}
      rewind(fp);

      /* Jumps to the correct location in file */
      fseek(fp, foffset, SEEK_SET);
    }

  /* Read each entry (simply scan all local sites) */
  for(i=1;i<=N[X];i++)
    for(j=1;j<=N[Y];j++)
      for(k=1;k<=N[Z];k++){
	func(fp);
      }

  foffset = ftell(fp);
  
  /* Pass on current offset (foffset) to next PE in local IO ring */
  if((io_grp.rank+1) != io_grp.size){
    MPI_Ssend(&foffset, 1, MPI_LONG, io_grp.rank+1, TAG_IO, IO_Comm);
  }

#else /* Serial version */

  int     i,j,k,io_size,io_index;
  int     N[3], nloc[3];
  IVector N_read;
  char    io_filename[FILENAME_MAX];
  FILE    *fp;

  LUDWIG_ENTER("COM_read_site()");

  get_N_local(N);

  strcpy(io_filename, filename);

  /* Open file for input */
  if( (fp = fopen(io_filename,"r")) == NULL )
    {
      fatal("COM_read_site(): could not open file %s\n", io_filename);
    }
    
  /* Ensure that simulation parameters are in the header */
  if( fscanf(fp,"%d %d %d %d %d %d %d %d\n\f",&nloc[X],&nloc[Y],&nloc[Z],
	     &N_read.x,&N_read.y,&N_read.z,&io_size,&io_index) != 8 )
    {
      fatal("COM_read_site(): input file %s not standard\n",io_filename);
    }

  if( nloc[X] != N[X] || nloc[Y] != N[Y] || nloc[Z] != N[Z] ||
      N_total(X) != N_read.x || N_total(Y) != N_read.y ||
      N_total(Z) != N_read.z )
    {
      fatal("COM_read_site(): size of system mismatch\n");
    }

  /* Read each entry (simply scan all local sites) */
  for(i=1;i<=N[X];i++)
    for(j=1;j<=N[Y];j++)
      for(k=1;k<=N[Z];k++){
	func(fp);
      }

  fclose(fp);

#endif /* _MPI_ */
}

/*---------------------------------------------------------------------------*\
 * void COM_write_site( char *filename, void (*func)( FILE *, int, int ))    *
 *                                                                           *
 * Writes `something' from each lattice site to filename. The actual         *
 * writing is done by a user supplied function passed as argument            *
 *                                                                           *
 * Version: 2.0                                                              *
 * Options: _TRACE_, _MPI_                                                   *
 *                                                                           *
 * Arguments                                                                 *
 * - filename:    file to which the sites will be written to                 *
 * - func:        pointer to user-supplied routine which will perform the IO *
 *                                                                           *
 * Last Updated: 15/07/2002 by JCD                                           *
\*---------------------------------------------------------------------------*/

void COM_write_site( char *filename, void (*func)( FILE *, int, int ))
{
#ifdef _MPI_ /* Parallel (MPI) Section */
  
  int     i,j,k,ind,xfac,yfac,gxfac,gyfac,g_ind,io_size,io_index;
  int     N[3];
  int     offset[3];
  char    io_filename[FILENAME_MAX];
  FILE    *fp;
  int     msg=0;
  MPI_Status status;

  LUDWIG_ENTER("COM_write_site()");

  get_N_local(N);
  get_N_offset(offset);

  sprintf(io_filename, "%s%s", filename,io_grp.file_ext);
  io_size = io_grp.size;
  io_index = io_grp.index;
  if(io_grp.root == TRUE)
    {
      /* Open file for output */
      if( (fp = fopen(io_filename,"w")) == NULL )
	{
	  fatal("COM_write_site(): could not open file %s\n", io_filename);
	}
      
      /* Print header information */
      fprintf(fp,"%d %d %d %d %d %d %d %d\n\f",N[X],N[Y],N[Z],
	      N_total(X),N_total(Y),N_total(Z),io_size,io_index);

    }  /* end of section for root IO PE */
  
  /* Each PE in the current IO group will write its own data in turn (by rank
     order). The use of blocking receives makes sure they don't `overtake' each
     other. All IO groups access their own file concurrently (parallel IO) */
  
  /* Initiate ring communication / synchronisation within local IO group */
  else				/* i.e. non IO root PE */
    {
      MPI_Recv(&msg, 1, MPI_INT, io_grp.rank-1, TAG_IO, IO_Comm, &status);
      if((fp=fopen(io_filename,"a+b")) == NULL)
	{
	  fatal("COM_write_site(): could not open file %s\n", io_filename);

	}
    }

  /* Write all local sites */

  xfac  = (N[Y]+2)*(N[Z]+2);
  yfac  = (N[Z]+2);

  gxfac = (N_total(Y) + 2)*(N_total(Z) + 2);
  gyfac = (N_total(Z) + 2);

  for(i=1;i<=N[X];i++)
    for(j=1;j<=N[Y];j++)
      for(k=1;k<=N[Z];k++)
	{
	  /* local index */
	  ind = i*xfac + j*yfac + k;
	  
	  /* global index */
	  g_ind = ((i + offset[X])*gxfac + (j + offset[Y])*gyfac + 
		   (k + offset[Z]));
	  func(fp,ind,g_ind);
	}
  fclose(fp);

  /* Inform next PE in local IO ring that it can now proceed with its own IO */
  if((io_grp.rank+1) != io_grp.size)
    MPI_Ssend(&msg, 1, MPI_INT, io_grp.rank+1, TAG_IO, IO_Comm);
  
#else /* Serial version */
  
  int     i,j,k,ind,xfac,yfac,io_size,io_index;
  int     N[3];
  char    io_filename[FILENAME_MAX];
  FILE    *fp;
  
  LUDWIG_ENTER("COM_write_site()");


  get_N_local(N);
  xfac = (N[Y]+2)*(N[Z]+2);
  yfac = (N[Z]+2);

  strcpy(io_filename, filename);
  io_size = 1; 
  io_index = 1;
  
  /* Open file for output */
  if( (fp = fopen(io_filename,"w")) == NULL )
    {
      fatal("COM_write_site(): could not open file %s\n", io_filename);
    }
  
  /* Print header information */

  fprintf(fp,"%d %d %d %d %d %d %d %d\n\f",N[X],N[Y],N[Z],N[X],N[Y],N[Z],
	  io_size, io_index);
  
  /* Write all sites */
  for(i=1; i<=N[X]; i++)
    for(j=1; j<=N[Y]; j++)
      for(k=1; k<=N[Z]; k++)
	{
	  /* local (and global) index */
	  ind = i*xfac + j*yfac + k;
	  
	  /* in the serial implementation, local index == global index */
	  func(fp,ind,ind);
	}
  
  fclose(fp);
  
#endif /* _MPI_ */
}

/*---------------------------------------------------------------------------*\
 * void COM_read_input_file( char *filename )                                *
 *                                                                           *
 * Reads in input file to doubly linked list for later processing            *
 *                                                                           *
 * Version: 2.0                                                              *
 * Options: _TRACE_, _MPI_                                                   *
 *                                                                           *
 * Arguments                                                                 *
 * - filename: input from which the parameter list will be read              *
 *                                                                           *
 * Last Updated: 15/07/2002 by JCD                                           *
\*---------------------------------------------------------------------------*/

void COM_read_input_file( char *filename )
{
  Input_Data *h,*p,*tmp;
  int  i,ii,nlines;
  char str[256], *strout;
  FILE *fp;
  
#ifdef _MPI_
  MPI_Datatype DT_InputData;
#endif /* _MPI_ */
  
  LUDWIG_ENTER("COM_read_input_file()");

  /* Placeholder for head of list */
  h = (Input_Data*)malloc(sizeof(Input_Data));
  h->last = h->next = NULL;
  
#ifdef _MPI_
  /* Set-up MPI datatype for structure Input_Data */
  MPI_Type_contiguous(sizeof(h->str),MPI_BYTE,&DT_InputData);
  MPI_Type_commit(&DT_InputData);
#endif /* _MPI_ */
  
  /* Open input files (if possible) else, fall back on default "input" */

  /* MPI implementation: root PE reads data, builds list and broadcasts to
     other PEs (restriction on max number of files open) */

  if(pe_rank() == 0)
    {
      nlines = 0;
      if( (fp = fopen(filename,"r")) == NULL )
	{
	  info("COM_read_input_file(): couldn't open input file %s\n",
	       filename);
	  
	  if( (fp = fopen("input","r")) == NULL ){
	    info("COM_read_input_file(): Using defaults\n");
	  }
	}
      
      /* Read input file into linked list */
      if( fp != NULL )
	{
	  while( (strout=fgets(str,256,fp)) != NULL )
	    {
	      nlines++;
	      /* Get new link */
	      p = (Input_Data*)malloc(sizeof(Input_Data));
	      
	      if( h->next != NULL ) h->next->last = p;
	      
	      p->next = h->next;
	      p->last = h;
	      h->next = p;
	      
	      /* Copy into string */
	      strcpy(p->str,str);
	    }
	  
	  /* Close down file */
	  fclose(fp);
	}
    }
  
  /* Root PE broadcasts Input Data to other PEs */
#ifdef _MPI_
  MPI_Bcast(&nlines, 1, MPI_INT, 0, MPI_COMM_WORLD);
#endif /* _MPI_ */

  p = h->next;
  for( i=0; i<nlines; i++ )
    {
      if( pe_rank() != 0 )
	{
	  /* Get new link */
	  p = (Input_Data*)malloc(sizeof(Input_Data));
	  
	  if( h->next != NULL ) h->next->last = p;
	  
	  p->next = h->next; 
	  p->last = h;
	  h->next = p;
	}
#ifdef _MPI_
      MPI_Bcast(p->str, 1, DT_InputData, 0, MPI_COMM_WORLD);
#endif /* _MPI_ */
      p = p->next;
    }
  
  /* Process COM specific options -- parallel things probably */
  COM_process_options( h );
  
  /* Process model dependent options */
  MODEL_process_options( h );

  /* Report on unused lines in input file */
  /* Read out list */
  p = h->next;
  
  /* Read out unused options */
  while( p != NULL )
    {

      /* Free memory */
      tmp = p;
      p = p->next;
      free(tmp);
    }
  
  free(h);
}
/*---------------------------------------------------------------------------*\
 * void COM_process_options( Input_Data *h )                                 *
 *                                                                           *
 * Process parallel specific options                                         *
 *                                                                           *
 * Version: 2.0                                                              *
 * Options: _TRACE_, _MPI_                                                   *
 *                                                                           *
 * Arguments                                                                 *
 * - h: pointer to start of input parameter list                             *
 *                                                                           *
 * Last Updated: 15/07/2002 by JCD                                           *
\*---------------------------------------------------------------------------*/

void COM_process_options( Input_Data *h )
{
#ifdef _MPI_

  int        i,flag,nthreads,max_io;
  Input_Data *p,*tmp;
  char       parameter[256],value[256];

  LUDWIG_ENTER("COM_process_options()");

  /* Set up defaults */
  /* Sets default number of I/O channels to their optimal value */

  for(i=1; i<=MAX_IO_NODE; i++)
    {
      if(((pe_size()%i) == 0) && ((pe_size()/i) > 1)){
	io_grp.n_io = i;
      }
    }
  
  /* Read out list */
  p = h->next;
  while( p != NULL )
    {
      /* Get strings */
      sscanf(p->str,"%s %s",parameter,value);
      
      flag = FALSE;
      
      /* Process strings */
      if( strcmp("n_io_nodes",parameter) == 0 )
	{
	  io_grp.n_io = imin(io_grp.n_io,atoi(value));
	  flag = TRUE;
	}
      
      /* If got an option ok, remove from list */
      if( flag )
	{
	  if( p->next != NULL ) p->next->last = p->last;
	  if( p->last != NULL ) p->last->next = p->next;		
	  
	  tmp = p->next;
	  free(p);
	  p = tmp;
	}
      else
	{
	  /* Next String */
	  p = p->next;
	}
    }
  
#endif /* _MPI_ */

  return;
}
/*---------------------------------------------------------------------------*\
 * int COM_local_index( int g_ind )                                          *
 *                                                                           *
 * Translates a global index g_ind into a local index ind                    *
 * Returns the local index                                                   *
 *                                                                           *
 * Version: 2.0                                                              *
 * Options: none                                                             *
 *                                                                           *
 * Arguments                                                                 *
 * - index: local index                                                      *
 *                                                                           *
 * Last Updated: 15/07/2002 by JCD                                           *
\*---------------------------------------------------------------------------*/

int COM_local_index( int g_ind )
{
  IVector coord;
  int N[3];
  int offset[3];
  int g_xfac, g_yfac, ind;

  get_N_local(N);
  get_N_offset(N);

  g_yfac = (N_total(Z) + 2);
  g_xfac = (N_total(Y) + 2) * g_yfac;
  
  coord.x =  g_ind / g_xfac;
  coord.y = (g_ind % g_xfac) / g_yfac;
  coord.z =  g_ind % g_yfac;
  
  coord.x -= offset[X];
  coord.y -= offset[Y];
  coord.z -= offset[Z];
  
  ind = coord.x*(N[Y]+2)*(N[Z]+2) + coord.y*(N[Z]+2) + coord.z;
  
  return(ind); 
}
/*---------------------------------------------------------------------------*\
 * IVector COM_index2coord( int index )                                      *
 *                                                                           *
 * Translates a local index index to local coordinates (x,y,z)               *
 * Returns the local co-ordinates                                            *
 *                                                                           *
 * Version: 2.0                                                              *
 * Options: none                                                             *
 *                                                                           *
 * Arguments                                                                 *
 * - index: local index                                                      *
 *                                                                           *
 * Last Updated: 15/07/2002 by JCD                                           *
 \*---------------------------------------------------------------------------*/

IVector COM_index2coord( int index )
{
  IVector coord;
  int N[3];
  int xfac,yfac;

  get_N_local(N);

  yfac = N[Z]+2;
  xfac = (N[Y]+2)*yfac;
  
  coord.x = index/xfac;
  coord.y = (index%xfac)/yfac;
  coord.z = index%yfac;
  return(coord);
}

/*------------------------- MPI-specific routines -------------------------*/
#ifdef _MPI_

/*---------------------------------------------------------------------------*\
 * IVector COM_decomp( IVector gblgrid, int npe )                            *
 *                                                                           *
 * Performs regular domain decomposition. Returns extents of local grid      *
 *                                                                           *
 * Version: 2.0                                                              *
 * Options: _TRACE_, _MPI_                                                   *
 *                                                                           *
 * Arguments                                                                 *
 * - gblgrid: extents of global grid                                         *
 * - npe:     number of processors (MPI only)                                *
 *                                                                           *
 *                                                                           *
 * Last Updated: 15/07/2002 by JCD                                           *
\*---------------------------------------------------------------------------*/

IVector COM_decomp( int npe )
{
  IVector locgrid;
  long    minsurf, surf;
  int     i, j, k, triplet=0;
  
  LUDWIG_ENTER("COM_decomp()");

  minsurf = LONG_MAX;
  for (i=1; i <= N_total(X); i++)
    for(j=1; j <= N_total(Y); j++)
      for(k=1; k <= N_total(Z); k++)
	{
	  if((N_total(X) % i == 0) && (N_total(Y)%j==0)&&(N_total(Z)%k==0)&&
	     ((N_total(X)/i)*(N_total(Y)/j)*(N_total(Z)/k)==npe))
	    {
	      triplet=1;
	      surf = i*j + i*k + j*k;
	      if(surf<minsurf)
		{
		  locgrid.x = (int)i; 
		  locgrid.y = (int)j;
		  locgrid.z = (int)k;
		  minsurf = surf;
		}
	    }
	}
  
  if (!triplet) {
    fatal("Could not decompose system with current parameters!\n");
  }

  return locgrid;
}

/*----- Utility routines for Global (user-defined) reduction operations ----*/

/*--------------------------------------------------------------------------*\
 * Global sum of FVectors                                                   *
 \*--------------------------------------------------------------------------*/

void COM_Add_FVector(FVector *invec, FVector *inoutvec, int *len,
		     MPI_Datatype *DT_Vector)
{
  int i;
  FVector fvec;

  for(i=0;i<(*len);++i)
    {
      fvec.x = invec->x + inoutvec->x;
      fvec.y = invec->y + inoutvec->y;
      fvec.z = invec->z + inoutvec->z;
      *inoutvec = fvec;
      invec++;
      inoutvec++;
    }
}

#endif /* _MPI_ */
/*--------------------------------------------------------------------------*/
