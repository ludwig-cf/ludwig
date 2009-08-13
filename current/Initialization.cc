void randomizeQ(void)
{
  int i,j,k,l,u;
  double phase,phase2,amplitude;

  // Global position (double)
  double ic, jc, kc;
  double ioff, joff, koff;

  ioff = Lx*pe_cartesian_coordinates_[0]/pe_cartesian_size_[0];
  joff = Ly*pe_cartesian_coordinates_[1]/pe_cartesian_size_[1];
  koff = Lz*pe_cartesian_coordinates_[2]/pe_cartesian_size_[2];



//========================//
// BLUEHAAR configuration //
//========================//
if(BLUEHAAR==1){

   // number and radius of DTCs	
   int Ndt=256;
   double Rdt=3;
   // rotation angles
   double *theta,*phi,di,dj,dk;
   // center coordinates of DTC
   double *idt,*jdt,*kdt;
   // rotated coordinate indices
   int ir,jr,kr;

   theta=new double[Ndt];
   phi=new double[Ndt];

   idt=new double[Ndt];
   jdt=new double[Ndt];
   kdt=new double[Ndt];


// DTC centers and rotation angles are set

   for(u=0; u<Ndt; u++){

      theta[u]=TwoPi*drand48();
      phi[u]=TwoPi*drand48();
      idt[u]=Lx*drand48();
      jdt[u]=Ly*drand48();
      kdt[u]=Lz*drand48();

   }


     for (i=ix1; i<ix2; i++) {
       for (j=jy1; j<jy2; j++) {
	 for (k=kz1; k<kz2; k++) {

	    Qxx[i][j][k]= 0.0;
	    Qxy[i][j][k]= 0.0;
	    Qyy[i][j][k]= 0.0;
	    Qxz[i][j][k]= 0.0;
	    Qyz[i][j][k]= 0.0;

	    ic = i-1 + ioff;
	    jc = j-1 + joff;
	    kc = k-1 + koff;

	    density[i][j][k]=densityinit;

	    amplitude=0.3;


	    for(u=0; u<Ndt; u++){

		di=ic-idt[u];
		dj=jc-jdt[u];
		dk=kc-kdt[u];

// If the current site is in an ROI

		if(di*di+dj*dj+dk*dk<Rdt*Rdt){

// a double rotation around DTC centers is performed.

		  ir = (int)floor(idt[u]+1-ioff + \
			di*cos(phi[u])-dj*cos(theta[u])*sin(phi[u])+dk*sin(theta[u])*sin(phi[u]));
		  jr = (int)floor(jdt[u]+1-joff + \
			di*sin(phi[u])+dj*cos(theta[u])*cos(phi[u])-dk*sin(theta[u])*cos(phi[u]));
		  kr = (int)floor(kdt[u]+1-koff + \
			dj*sin(theta[u])+dk*cos(theta[u]));


		     if((0<=ir && ir<Lx2) && (0<=jr && jr<Ly2) && (0<=kr && kr<Lz2)){  

		      Qxx[ir][jr][kr]=amplitude*(-cos(2*q0*jc));
		      Qxxinit[ir][jr][kr]=Qxx[ir][jr][kr];
		      Qxy[ir][jr][kr]=0.0;
		      Qxyinit[ir][jr][kr]=Qxy[ir][jr][kr];
		      Qxz[ir][jr][kr]=amplitude*sin(2.0*q0*jc);
		      Qxzinit[ir][jr][kr]=Qxz[ir][jr][kr];

		      Qyy[ir][jr][kr]=amplitude*(-cos(2.0*q0*ic));
		      Qyz[ir][jr][kr]=-amplitude*sin(2.0*q0*ic);
		      Qyyinit[ir][jr][kr]=Qyy[ir][jr][kr];
		      Qyzinit[ir][jr][kr]=Qyz[ir][jr][kr];

		     }

		  }

	       }

	    for (l=0; l<15; l++) {
		 f[i][j][k][l]=density[i][j][k]/15.0;
	    }

	    }
	 }
     }


  exchangeMomentumAndQTensor();



// All other lattice sites are set to either isotropic liquid or cholesteric LC

     for (i=ix1; i<ix2; i++) {
       for (j=jy1; j<jy2; j++) {
	 for (k=kz1; k<kz2; k++) {

	    if(Qxx[i][j][k]<1e-9 && Qxy[i][j][k]<1e-9 && Qxz[i][j][k]<1e-9 && Qyy[i][j][k]<1e-9 && Qyz[i][j][k]<1e-9){

	    ic = i-1 + ioff;
	    jc = j-1 + joff;
	    kc = k-1 + koff;

	    amplitude=0.3;


// isotropic liquid

/*
		   Qxx[i][j][k]= 1e-4/2.0;
		   Qxy[i][j][k]= 0.0;
		   Qyy[i][j][k]= -1e-4;
		   Qxz[i][j][k]= 0.0;
		   Qyz[i][j][k]= 0.0;
*/
// cholesteric environment

		   Qxx[i][j][k]=0.2723/2.0+amplitude*cos(2.0*q0*jc);
		   Qxy[i][j][k]= 0.0;
		   Qyy[i][j][k]= -0.2723;
		   Qxz[i][j][k]= -amplitude*(sin(2.0*q0*jc));
		   Qyz[i][j][k]= 0.0;

	       }

	    }
	 }
     }

 }
//================================================//
// initial configurations different from BLUEHAAR //
//================================================//
if(BLUEHAAR!=1){

  for (i=ix1; i<ix2; i++) {
    for (j=jy1; j<jy2; j++) {
      for (k=kz1; k<kz2; k++) {

	  ic = i-1 + ioff;
	  jc = j-1 + joff;
	  kc = k-1 + koff;

	density[i][j][k]=densityinit;

	amplitude=0.2;
	if (O8MSTRUCT == 1) amplitude=-0.2;

        if (TWIST == 1){

      amplitude=(0.546-0.2723/2.0);
//      amplitude=0.40985;

// cholesteric LC



      Qxx[i][j][k]=0.2723/2.0+amplitude*cos(2.0*q0*jc);
      Qxy[i][j][k]= 0.0;
      Qyy[i][j][k]= -0.2723;
      Qxz[i][j][k]= -amplitude*(sin(2.0*q0*jc));
      Qyz[i][j][k]= 0.0;

         }


 // phase=2.0*q0*k;


	if (DTSTRUCT == 1) {
          amplitude=0.3;

	  Qxx[i][j][k]=amplitude*(-cos(2*q0*jc));
	  Qxxinit[i][j][k]=Qxx[i][j][k];
	  Qxy[i][j][k]=0.0;
	  Qxyinit[i][j][k]=Qxy[i][j][k];
	  Qxz[i][j][k]=amplitude*sin(2.0*q0*jc);
	  Qxzinit[i][j][k]=Qxz[i][j][k];

	  Qyy[i][j][k]=amplitude*(-cos(2.0*q0*ic));
	  Qyz[i][j][k]=-amplitude*sin(2.0*q0*ic);
	  Qyyinit[i][j][k]=Qyy[i][j][k];
	  Qyzinit[i][j][k]=Qyz[i][j][k];

	}


	if (O2STRUCT == 1) {

	  amplitude=0.3; 

	  Qxx[i][j][k]=amplitude*(cos(2.0*q0*kc)-cos(2.0*q0*jc));
	  Qxxinit[i][j][k]=Qxx[i][j][k];
	  Qxy[i][j][k]=amplitude*sin(2.0*q0*kc);
	  Qxyinit[i][j][k]=Qxy[i][j][k];
	  Qxz[i][j][k]=amplitude*sin(2.0*q0*jc);

	  Qyy[i][j][k]=amplitude*(cos(2.0*q0*ic)-cos(2.0*q0*kc));
	  Qyyinit[i][j][k]=Qyy[i][j][k];
	  Qyz[i][j][k]=amplitude*sin(2.0*q0*ic);
 	  Qyzinit[i][j][k]=Qyz[i][j][k];

}

	if (O5STRUCT == 1) {

	  Qxx[i][j][k]=amplitude*
	    (2.0*cos(sqrt(2.0)*q0*jc)*cos(sqrt(2.0)*q0*kc)-
	         cos(sqrt(2.0)*q0*ic)*cos(sqrt(2.0)*q0*kc)-
	         cos(sqrt(2.0)*q0*ic)*cos(sqrt(2.0)*q0*jc));
	  Qyy[i][j][k]=amplitude*
	    (2.0*cos(sqrt(2.0)*q0*ic)*cos(sqrt(2.0)*q0*kc)-
	         cos(sqrt(2.0)*q0*jc)*cos(sqrt(2.0)*q0*ic)-
	         cos(sqrt(2.0)*q0*jc)*cos(sqrt(2.0)*q0*kc));
	  Qxy[i][j][k]=amplitude*
	    (sqrt(2.0)*cos(sqrt(2.0)*q0*jc)*sin(sqrt(2.0)*q0*kc)-
	     sqrt(2.0)*cos(sqrt(2.0)*q0*ic)*sin(sqrt(2.0)*q0*kc)-
	     sin(sqrt(2.0)*q0*ic)*sin(sqrt(2.0)*q0*jc));
	  Qxz[i][j][k]=amplitude*
	    (sqrt(2.0)*cos(sqrt(2.0)*q0*ic)*sin(sqrt(2.0)*q0*jc)-
	     sqrt(2.0)*cos(sqrt(2.0)*q0*kc)*sin(sqrt(2.0)*q0*jc)-
	     sin(sqrt(2.0)*q0*ic)*sin(sqrt(2.0)*q0*kc));
	  Qyz[i][j][k]=amplitude*
	    (sqrt(2.0)*cos(sqrt(2.0)*q0*kc)*sin(sqrt(2.0)*q0*ic)-
	     sqrt(2.0)*cos(sqrt(2.0)*q0*jc)*sin(sqrt(2.0)*q0*ic)-
	     sin(sqrt(2.0)*q0*jc)*sin(sqrt(2.0)*q0*kc));
	}

	if ((O8STRUCT == 1) || (O8MSTRUCT == 1)) {
	    Qxx[i][j][k]=amplitude*
		(-2.0*cos(sqrt(2.0)*q0*jc)*sin(sqrt(2.0)*q0*kc)+
		 sin(sqrt(2.0)*q0*ic)*cos(sqrt(2.0)*q0*kc)+
		 cos(sqrt(2.0)*q0*ic)*sin(sqrt(2.0)*q0*jc));
	  Qyy[i][j][k]=amplitude*
	    (-2.0*sin(sqrt(2.0)*q0*ic)*cos(sqrt(2.0)*q0*kc)+
	     sin(sqrt(2.0)*q0*jc)*cos(sqrt(2.0)*q0*ic)+
	     cos(sqrt(2.0)*q0*jc)*sin(sqrt(2.0)*q0*kc));
	  Qxy[i][j][k]=amplitude*
	    (sqrt(2.0)*cos(sqrt(2.0)*q0*jc)*cos(sqrt(2.0)*q0*kc)+
	     sqrt(2.0)*sin(sqrt(2.0)*q0*ic)*sin(sqrt(2.0)*q0*kc)-
	     sin(sqrt(2.0)*q0*ic)*cos(sqrt(2.0)*q0*jc));
	  Qxz[i][j][k]=amplitude*
	    (sqrt(2.0)*cos(sqrt(2.0)*q0*ic)*cos(sqrt(2.0)*q0*jc)+
	     sqrt(2.0)*sin(sqrt(2.0)*q0*kc)*sin(sqrt(2.0)*q0*jc)-
	     cos(sqrt(2.0)*q0*ic)*sin(sqrt(2.0)*q0*kc));
	  Qyz[i][j][k]=amplitude*
	    (sqrt(2.0)*cos(sqrt(2.0)*q0*kc)*cos(sqrt(2.0)*q0*ic)+
	     sqrt(2.0)*sin(sqrt(2.0)*q0*jc)*sin(sqrt(2.0)*q0*ic)-
	     sin(sqrt(2.0)*q0*jc)*cos(sqrt(2.0)*q0*kc));


	}


   if (HEXPLANAR == 1) {

/* twist is along z-direction */

     Qxx[i][j][k]=amplitude*(-1.5*cos(q0*ic)*cos(q0*sqrt(3.0)*jc));
     Qxy[i][j][k]=amplitude*(-0.5*sqrt(3.0)*sin(q0*ic)*sin(q0*sqrt(3.0)*jc));
     Qxz[i][j][k]=amplitude*(sqrt(3.0)*cos(q0*ic)*sin(q0*sqrt(3.0)*jc));
     Qyy[i][j][k]=amplitude*(-cos(2.0*q0*ic)-0.5*cos(q0*ic)*cos(q0*sqrt(3.0)*jc));
     Qyz[i][j][k]=amplitude*(-sin(2.0*q0*ic)-sin(q0*ic)*cos(q0*sqrt(3.0)*jc));

   }



   if(HEX3DA == 1) {

     Qxx[i][j][k]=amplitude*(-1.5*cos(q0*ic)*cos(q0*sqrt(3.0)*jc)+0.25*cos(q0*Lx/Lz*kc));
     Qxy[i][j][k]=amplitude*(-0.5*sqrt(3.0)*sin(q0*ic)*sin(q0*sqrt(3.0)*jc)+0.25*sin(q0*Lx/Lz*kc));
     Qxz[i][j][k]=amplitude*(sqrt(3.0)*cos(q0*ic)*sin(q0*sqrt(3.0)*jc));
     Qyy[i][j][k]=amplitude*(-cos(2.0*q0*ic)-0.5*cos(q0*ic)*cos(q0*sqrt(3.0)*jc)-0.25*cos(q0*Lx/Lz*kc));
     Qyz[i][j][k]=amplitude*(-sin(2.0*q0*ic)-sin(q0*ic)*cos(q0*sqrt(3.0)*jc));

   }


     if(HEX3DB == 1) {

       Qxx[i][j][k]=amplitude*(1.5*cos(q0*ic)*cos(q0*sqrt(3.0)*jc)+0.25*cos(q0*Lx/Lz*kc));
       Qxy[i][j][k]=amplitude*(0.5*sqrt(3.0)*sin(q0*ic)*sin(q0*sqrt(3.0)*jc)+0.25*sin(q0*Lx/Lz*kc));
       Qxz[i][j][k]=amplitude*(-sqrt(3.0)*cos(q0*ic)*sin(q0*sqrt(3.0)*jc));
       Qyy[i][j][k]=amplitude*(cos(2.0*q0*ic)+0.5*cos(q0*ic)*cos(q0*sqrt(3.0)*jc)-0.25*cos(q0*Lx/Lz*kc));
       Qyz[i][j][k]=amplitude*(sin(2.0*q0*ic)+sin(q0*ic)*cos(q0*sqrt(3.0)*jc));

     }



	if (RANDOM == 1) {
		
	  amplitude=1e-2;

	  phase= 2.0/5.0*Pi*(0.5-drand48());  
	  phase2= Pi/2.0+Pi/5.0*(0.5-drand48());

	  Qxx[i][j][k]= amplitude*
	    (3.0/2.0*sin(phase2)*sin(phase2)*cos(phase)*cos(phase)-1.0/2.0);
	  Qxy[i][j][k]= 3.0*amplitude/2.0*
	    (sin(phase2)*sin(phase2)*cos(phase)*sin(phase));
	  Qyy[i][j][k]= amplitude*
	    (3.0/2.0*sin(phase2)*sin(phase2)*sin(phase)*sin(phase)-1.0/2.0);
	  Qxz[i][j][k]=
	    3.0*amplitude/2.0*(sin(phase2)*cos(phase2)*cos(phase));
	  Qyz[i][j][k]=
	    3.0*amplitude/2.0*(sin(phase2)*cos(phase2)*sin(phase));

	}



	for (l=0; l<15; l++) {
	  f[i][j][k][l]=density[i][j][k]/15.0;
	}
      }
    }
   }
}
//================================//
// end alternative configurations //
//================================//

}

void startDroplet(void)
{
  int i,j,k;
  double amplitude;
  double fracmin, fracmax;

  int ic, jc, kc;
  int ioff = 0, joff = 0, koff = 0;

  ioff = Lx*pe_cartesian_coordinates_[0]/pe_cartesian_size_[0];
  joff = Ly*pe_cartesian_coordinates_[1]/pe_cartesian_size_[1];
  koff = Lz*pe_cartesian_coordinates_[2]/pe_cartesian_size_[2];


  for (i=ix1; i<ix2; i++) {
    for (j=jy1; j<jy2; j++) {
      for (k=kz1; k<kz2; k++) {

	ic = i - ix1 + ioff;
	jc = j - jy1 + joff;
	kc = k - kz1 + koff;


      /**************/
      /* define ROI */
      /**************/

	fracmin = 0.5 - 1.0/numuc;
	fracmax = 0.5 + 1.0/numuc;

///*
// replace sites outside ROI

	if ( (ic < (fracmin*Lx) )||( ic > (fracmax*Lx))||
	     (jc < (fracmin*Ly) )||( jc > (fracmax*Ly))||
	     (kc < (fracmin*Lz) )||( kc > (fracmax*Lz)) ) { 
//*/

/*
// replace sites inside ROI

	if ( (ic > (fracmin*Lx) )&&( ic < (fracmax*Lx)) &&
	     (jc > (fracmin*Ly) )&&( jc < (fracmax*Ly)) &&
	     (kc > (fracmin*Lz) )&&( kc < (fracmax*Lz)) ) {
*/

	  amplitude=(0.546-0.2723/2.0);

	  // droplet in cholesteric environment
       /* 
	  Qxx[i][j][k]=0.2723/2.0+amplitude*cos(2.0*q0*jc);
	  Qxy[i][j][k]= 0.0;
	  Qyy[i][j][k]= -0.2723;
	  Qxz[i][j][k]= -amplitude*(sin(2.0*q0*jc));
	  Qyz[i][j][k]= 0.0;

	  // allow for different definition of the pitch in O8 and O8M
	  if(O8MSTRUCT == 1 || O8STRUCT == 1){

	    Qxx[i][j][k]=0.2723/2.0+amplitude*cos(2.0*q0/sqrt(2.0)*jc);
	    Qxy[i][j][k]= 0.0;
	    Qyy[i][j][k]= -0.2723;
	    Qxz[i][j][k]= -amplitude*(sin(2.0*q0/sqrt(2.0)*jc));
	    //sign change
	    //	 Qxz[i][j][k]= amplitude*(sin(2.0*q0/sqrt(2.0)*jc));
	    Qyz[i][j][k]= 0.0;


	  }
       */

	  //  droplet in isotropic environment

//	  /*
	        Qxx[i][j][k]= 1e-4/2.0;
	        Qxy[i][j][k]= 0.0;
	        Qyy[i][j][k]= -1e-4;
	        Qxz[i][j][k]= 0.0;
	        Qyz[i][j][k]= 0.0;
//	 */

// BP droplet 

/*
	if (O2STRUCT == 1) {

	  amplitude=0.3; 

	  Qxx[i][j][k]=amplitude*(cos(2.0*q0*kc)-cos(2.0*q0*jc));
	  Qxxinit[i][j][k]=Qxx[i][j][k];
	  Qxy[i][j][k]=amplitude*sin(2.0*q0*kc);
	  Qxyinit[i][j][k]=Qxy[i][j][k];
	  Qxz[i][j][k]=amplitude*sin(2.0*q0*jc);

	  Qyy[i][j][k]=amplitude*(cos(2.0*q0*ic)-cos(2.0*q0*kc));
	  Qyyinit[i][j][k]=Qyy[i][j][k];
	  Qyz[i][j][k]=amplitude*sin(2.0*q0*ic);
 	  Qyzinit[i][j][k]=Qyz[i][j][k];

	 }



	if ((O8STRUCT == 1) || (O8MSTRUCT == 1)) {

	  if (O8MSTRUCT == 1) amplitude=-0.2; 

	  Qxx[i][j][k]=amplitude*
		(-2.0*cos(sqrt(2.0)*q0*jc)*sin(sqrt(2.0)*q0*kc)+
		 sin(sqrt(2.0)*q0*ic)*cos(sqrt(2.0)*q0*kc)+
		 cos(sqrt(2.0)*q0*ic)*sin(sqrt(2.0)*q0*jc));
	  Qyy[i][j][k]=amplitude*
	    (-2.0*sin(sqrt(2.0)*q0*ic)*cos(sqrt(2.0)*q0*kc)+
	     sin(sqrt(2.0)*q0*jc)*cos(sqrt(2.0)*q0*ic)+
	     cos(sqrt(2.0)*q0*jc)*sin(sqrt(2.0)*q0*kc));
	  Qxy[i][j][k]=amplitude*
	    (sqrt(2.0)*cos(sqrt(2.0)*q0*jc)*cos(sqrt(2.0)*q0*kc)+
	     sqrt(2.0)*sin(sqrt(2.0)*q0*ic)*sin(sqrt(2.0)*q0*kc)-
	     sin(sqrt(2.0)*q0*ic)*cos(sqrt(2.0)*q0*jc));
	  Qxz[i][j][k]=amplitude*
	    (sqrt(2.0)*cos(sqrt(2.0)*q0*ic)*cos(sqrt(2.0)*q0*jc)+
	     sqrt(2.0)*sin(sqrt(2.0)*q0*kc)*sin(sqrt(2.0)*q0*jc)-
	     cos(sqrt(2.0)*q0*ic)*sin(sqrt(2.0)*q0*kc));
	  Qyz[i][j][k]=amplitude*
	    (sqrt(2.0)*cos(sqrt(2.0)*q0*kc)*cos(sqrt(2.0)*q0*ic)+
	     sqrt(2.0)*sin(sqrt(2.0)*q0*jc)*sin(sqrt(2.0)*q0*ic)-
	     sin(sqrt(2.0)*q0*jc)*cos(sqrt(2.0)*q0*kc));

	}

 */

	}
      }
    }
  }
}


void startSlab(void)
{
  int i,j,k;
  double fracmin, fracmax, amplitude;

  int ic, jc, kc;
  int ioff = 0, joff = 0, koff = 0;


  ioff = Lx*pe_cartesian_coordinates_[0]/pe_cartesian_size_[0];
  joff = Ly*pe_cartesian_coordinates_[1]/pe_cartesian_size_[1];
  koff = Lz*pe_cartesian_coordinates_[2]/pe_cartesian_size_[2];


  for (i=ix1; i<ix2; i++) {
    for (j=jy1; j<jy2; j++) {
      for (k=kz1; k<kz2; k++) {

	ic = i - ix1 + ioff;
	jc = j - jy1 + joff;
	kc = k - kz1 + koff;

	fracmin = 0.5;
	fracmax = 1.0;

	if ((jc > (fracmin*Ly)) && (jc < (fracmax*Ly))) {

/*
	  amplitude=(0.546-0.2723/2.0);

	  // slab in cholesteric environment
      
	  Qxx[i][j][k]=0.2723/2.0+amplitude*cos(2.0*q0*jc);
	  Qxy[i][j][k]= 0.0;
	  Qyy[i][j][k]= -0.2723;
	  Qxz[i][j][k]= -amplitude*(sin(2.0*q0*jc));
	  Qyz[i][j][k]= 0.0;

	  // allow for different definition of the pitch in O8 and O8M
	  if(O8MSTRUCT == 1 || O8STRUCT == 1){

	    Qxx[i][j][k]=0.2723/2.0+amplitude*cos(2.0*q0/sqrt(2.0)*jc);
	    Qxy[i][j][k]= 0.0;
	    Qyy[i][j][k]= -0.2723;
	    Qxz[i][j][k]= -amplitude*(sin(2.0*q0/sqrt(2.0)*jc));
	    //sign change
	    //	 Qxz[i][j][k]= amplitude*(sin(2.0*q0/sqrt(2.0)*jc));
	    Qyz[i][j][k]= 0.0;


	  }
*/

	  //  slab in isotropic environment

	  /*
	        Qxx[i][j][k]= 1e-4/2.0;
	        Qxy[i][j][k]= 0.0;
	        Qyy[i][j][k]= -1e-4;
	        Qxz[i][j][k]= 0.0;
	        Qyz[i][j][k]= 0.0;
	 */

// BP slab 

///*
	if (O2STRUCT == 1) {

	  amplitude=0.3; 

	  Qxx[i][j][k]=amplitude*(cos(2.0*q0*kc)-cos(2.0*q0*jc));
	  Qxxinit[i][j][k]=Qxx[i][j][k];
	  Qxy[i][j][k]=amplitude*sin(2.0*q0*kc);
	  Qxyinit[i][j][k]=Qxy[i][j][k];
	  Qxz[i][j][k]=amplitude*sin(2.0*q0*jc);

	  Qyy[i][j][k]=amplitude*(cos(2.0*q0*ic)-cos(2.0*q0*kc));
	  Qyyinit[i][j][k]=Qyy[i][j][k];
	  Qyz[i][j][k]=amplitude*sin(2.0*q0*ic);
 	  Qyzinit[i][j][k]=Qyz[i][j][k];

	 }



	if ((O8STRUCT == 1) || (O8MSTRUCT == 1)) {

	  if (O8MSTRUCT == 1) amplitude=-0.2; 

	  Qxx[i][j][k]=amplitude*
		(-2.0*cos(sqrt(2.0)*q0*jc)*sin(sqrt(2.0)*q0*kc)+
		 sin(sqrt(2.0)*q0*ic)*cos(sqrt(2.0)*q0*kc)+
		 cos(sqrt(2.0)*q0*ic)*sin(sqrt(2.0)*q0*jc));
	  Qyy[i][j][k]=amplitude*
	    (-2.0*sin(sqrt(2.0)*q0*ic)*cos(sqrt(2.0)*q0*kc)+
	     sin(sqrt(2.0)*q0*jc)*cos(sqrt(2.0)*q0*ic)+
	     cos(sqrt(2.0)*q0*jc)*sin(sqrt(2.0)*q0*kc));
	  Qxy[i][j][k]=amplitude*
	    (sqrt(2.0)*cos(sqrt(2.0)*q0*jc)*cos(sqrt(2.0)*q0*kc)+
	     sqrt(2.0)*sin(sqrt(2.0)*q0*ic)*sin(sqrt(2.0)*q0*kc)-
	     sin(sqrt(2.0)*q0*ic)*cos(sqrt(2.0)*q0*jc));
	  Qxz[i][j][k]=amplitude*
	    (sqrt(2.0)*cos(sqrt(2.0)*q0*ic)*cos(sqrt(2.0)*q0*jc)+
	     sqrt(2.0)*sin(sqrt(2.0)*q0*kc)*sin(sqrt(2.0)*q0*jc)-
	     cos(sqrt(2.0)*q0*ic)*sin(sqrt(2.0)*q0*kc));
	  Qyz[i][j][k]=amplitude*
	    (sqrt(2.0)*cos(sqrt(2.0)*q0*kc)*cos(sqrt(2.0)*q0*ic)+
	     sqrt(2.0)*sin(sqrt(2.0)*q0*jc)*sin(sqrt(2.0)*q0*ic)-
	     sin(sqrt(2.0)*q0*jc)*cos(sqrt(2.0)*q0*kc));

	}

// */



	}
      }
    }
  }
}


void reinit()
{
  gasdev(1);

  f=fa;
  fpr=fb;

  oneplusdtover2tau1=1.0+0.5*dt/tau1;
  oneplusdtover2tau2=1.0+0.5*dt/tau2;

  wallamp=0.03;

  if (gam > 2.7){
    wallamp=(0.25+0.75*sqrt((1.0-8.0/(3.0*gam))));
  }
 
  Qxxtop= wallamp*(sin(angztop/180.0*Pi)*sin(angztop/180.0*Pi)*
     cos(angxytop/180.0*Pi)*cos(angxytop/180.0*Pi)-1.0/3.0); 
  Qxytop= wallamp*sin(angztop/180.0*Pi)*sin(angztop/180.0*Pi)*
    cos(angxytop/180.0*Pi)*sin(angxytop/180.0*Pi);
  Qyytop= wallamp*(sin(angztop/180.0*Pi)*sin(angztop/180.0*Pi)*
     sin(angxytop/180.0*Pi)*sin(angxytop/180.0*Pi)-1.0/3.0);
  Qxztop= wallamp*sin(angztop/180.0*Pi)*cos(angztop/180.0*Pi)*
    cos(angxytop/180.0*Pi);
  Qyztop= wallamp*sin(angztop/180.0*Pi)*cos(angztop/180.0*Pi)*
    sin(angxytop/180.0*Pi);
	
  Qxxbot= wallamp*(sin(angzbot/180.0*Pi)*sin(angzbot/180.0*Pi)*
     cos(angxybot/180.0*Pi)*cos(angxybot/180.0*Pi)-1.0/3.0); 
  Qxybot= wallamp*sin(angzbot/180.0*Pi)*sin(angzbot/180.0*Pi)*
    cos(angxybot/180.0*Pi)*sin(angxybot/180.0*Pi);
  Qyybot= wallamp*(sin(angzbot/180.0*Pi)*sin(angzbot/180.0*Pi)*
     sin(angxybot/180.0*Pi)*sin(angxybot/180.0*Pi)-1.0/3.0);
  Qxzbot= wallamp*sin(angzbot/180.0*Pi)*cos(angzbot/180.0*Pi)*
    cos(angxybot/180.0*Pi);
  Qyzbot= wallamp*sin(angzbot/180.0*Pi)*cos(angzbot/180.0*Pi)*
    sin(angxybot/180.0*Pi);

  randomizeQ();

}

void initialize(void)
{
  int n;

#ifdef PARALLEL

  int periodic[3] = {1, 1, 1};
  int pe_cartesian_rank;
  int colour, key;
  int reorder=1;
 
  /* Basic run time check */

  n = pe_cartesian_size_[0]*pe_cartesian_size_[1]*pe_cartesian_size_[2];
  if (n != nbPE) {
    cout << "nbPE = " << nbPE << endl;
    cout << "x size: " << pe_cartesian_size_[0] << endl;
    cout << "y size: " << pe_cartesian_size_[1] << endl;
    cout << "z size: " << pe_cartesian_size_[2] << endl;
    cout << "Incorrect decomposition " << endl;
    MPI_Abort(MPI_COMM_WORLD, 0);
  }

  MPI_Cart_create(MPI_COMM_WORLD, 3, pe_cartesian_size_, periodic, reorder,
		  &cartesian_communicator_);

  MPI_Comm_rank(cartesian_communicator_, &pe_cartesian_rank);

  MPI_Cart_coords(cartesian_communicator_, pe_cartesian_rank, 3,
		  pe_cartesian_coordinates_);

  /* Add 2 to the lattice size in each direction to accomodate halo regions,
   * which are located at L=0 and L=L-1 */

  Lx2 = 2 + Lx/pe_cartesian_size_[0];
  ix1=1;
  ix2=Lx2-1;

  Ly2 = 2 + Ly/pe_cartesian_size_[1];
  jy1=1;
  jy2=Ly2-1;

  Lz2 = 2 + Lz/pe_cartesian_size_[2];
  kz1=1;
  kz2=Lz2-1;

  /* We use reorder as a temporary here, as MPI_Cart_shift can
   * change the actual argument for the rank of the source of
   * the recieve, but we only want destination of send. */

  for (n = 0; n < 3; n++) {
    reorder = pe_cartesian_coordinates_[n];
    MPI_Cart_shift(cartesian_communicator_, n, -1, &reorder,
		   &pe_cartesian_neighbour_[0][n]);
    reorder = pe_cartesian_coordinates_[n];
    MPI_Cart_shift(cartesian_communicator_, n, +1, &reorder,
		   &pe_cartesian_neighbour_[1][n]);
  }

  /* I/O communciation */

  io_group_size_ = nbPE / io_ngroups_;

  colour = pe_cartesian_rank / io_group_size_;
  key = pe_cartesian_rank;
  io_group_id_ = colour;

  MPI_Comm_split(cartesian_communicator_, colour, key, &io_communicator_);
  MPI_Comm_rank(io_communicator_, &io_rank_);

  cout << "World rank " << myPE << " Cart rank " << pe_cartesian_rank <<
    " io_group " << io_group_id_ << " io rank " << io_rank_ << endl;

  MPI_Barrier(MPI_COMM_WORLD);

#else

  // Serial version
  // Check the input was appropriate
  n = pe_cartesian_size_[0]*pe_cartesian_size_[1]*pe_cartesian_size_[2];
  if (n != 1) {
    cout << " Check input decomposition is {1, 1, 1}!" << endl;
    cout << " Setting correct values for serial run..." << endl;
  }
  // Whatever was in the input, we must have...
  pe_cartesian_coordinates_[0] = 0;
  pe_cartesian_coordinates_[1] = 0;
  pe_cartesian_coordinates_[2] = 0;
  pe_cartesian_size_[0] = 1;
  pe_cartesian_size_[1] = 1;
  pe_cartesian_size_[2] = 1;
  io_ngroups_ = 1;
  io_group_id_ = 0;

  Lx2=Lx+2;
  ix1=1;
  ix2=Lx2-1;
  Ly2=Ly+2;
  jy1=1;
  jy2=Ly2-1;
  Lz2=Lz+2;
  kz1=1;
  kz2=Lz2-1;
#endif

  int ix,iy,iz;

  fa=new double***[Lx2];
  for (ix=0;ix<Lx2;ix++) {
    fa[ix]=new double**[Ly2];
    for (iy=0;iy<Ly2;iy++) {
      fa[ix][iy]=new double*[Lz2];
      for (iz=0;iz<Lz2;iz++)
	fa[ix][iy][iz]=new double[15];
    }
  }


  fb=new double***[Lx2];
  for (ix=0;ix<Lx2;ix++) {
    fb[ix]=new double**[Ly2];
    for (iy=0;iy<Ly2;iy++) {
      fb[ix][iy]=new double*[Lz2];
      for (iz=0;iz<Lz2;iz++)
	fb[ix][iy][iz]=new double[15];
    }
  }


  Fc=new double***[Lx2];
  for (ix=0;ix<Lx2;ix++) {
    Fc[ix]=new double**[Ly2];
    for (iy=0;iy<Ly2;iy++) {
      Fc[ix][iy]=new double*[Lz2];
      for (iz=0;iz<Lz2;iz++)
	Fc[ix][iy][iz]=new double[15];
    }
  }


  feq=new double***[Lx2];
  for (ix=0;ix<Lx2;ix++) {
    feq[ix]=new double**[Ly2];
    for (iy=0;iy<Ly2;iy++) {
      feq[ix][iy]=new double*[Lz2];
      for (iz=0;iz<Lz2;iz++)
	feq[ix][iy][iz]=new double[15];
    }
  }

  u=new double***[Lx2];
  for (ix=0;ix<Lx2;ix++) {
    u[ix]=new double**[Ly2];
    for (iy=0;iy<Ly2;iy++) {
      u[ix][iy]=new double*[Lz2];
      for (iz=0;iz<Lz2;iz++)
	u[ix][iy][iz]=new double[3];
    }
  }

  Fh=new double***[Lx2];
  for (ix=0;ix<Lx2;ix++) {
    Fh[ix]=new double**[Ly2];
    for (iy=0;iy<Ly2;iy++) {
      Fh[ix][iy]=new double*[Lz2];
      for (iz=0;iz<Lz2;iz++)
	Fh[ix][iy][iz]=new double[3];
    }
  }



  density=new double**[Lx2];
  for (ix=0;ix<Lx2;ix++) {
    density[ix]=new double*[Ly2];
    for (iy=0;iy<Ly2;iy++)
      density[ix][iy]=new double[Lz2];
  }
  Qxx=new double**[Lx2];
  for (ix=0;ix<Lx2;ix++) {
    Qxx[ix]=new double*[Ly2];
    for (iy=0;iy<Ly2;iy++)
      Qxx[ix][iy]=new double[Lz2];
  }
  Qxy=new double**[Lx2];
  for (ix=0;ix<Lx2;ix++) {
    Qxy[ix]=new double*[Ly2];
    for (iy=0;iy<Ly2;iy++)
      Qxy[ix][iy]=new double[Lz2];
  }
  Qyy=new double**[Lx2];
  for (ix=0;ix<Lx2;ix++) {
    Qyy[ix]=new double*[Ly2];
    for (iy=0;iy<Ly2;iy++)
      Qyy[ix][iy]=new double[Lz2];
  }
  Qxz=new double**[Lx2];
  for (ix=0;ix<Lx2;ix++) {
    Qxz[ix]=new double*[Ly2];
    for (iy=0;iy<Ly2;iy++)
      Qxz[ix][iy]=new double[Lz2];
  }
  Qyz=new double**[Lx2];
  for (ix=0;ix<Lx2;ix++) {
    Qyz[ix]=new double*[Ly2];
    for (iy=0;iy<Ly2;iy++)
      Qyz[ix][iy]=new double[Lz2];
  }
  Qxxold=new double**[Lx2];
  for (ix=0;ix<Lx2;ix++) {
    Qxxold[ix]=new double*[Ly2];
    for (iy=0;iy<Ly2;iy++)
      Qxxold[ix][iy]=new double[Lz2];
  }
  Qxyold=new double**[Lx2];
  for (ix=0;ix<Lx2;ix++) {
    Qxyold[ix]=new double*[Ly2];
    for (iy=0;iy<Ly2;iy++)
      Qxyold[ix][iy]=new double[Lz2];
  }
  Qyyold=new double**[Lx2];
  for (ix=0;ix<Lx2;ix++) {
    Qyyold[ix]=new double*[Ly2];
    for (iy=0;iy<Ly2;iy++)
      Qyyold[ix][iy]=new double[Lz2];
  }
  Qxzold=new double**[Lx2];
  for (ix=0;ix<Lx2;ix++) {
    Qxzold[ix]=new double*[Ly2];
    for (iy=0;iy<Ly2;iy++)
      Qxzold[ix][iy]=new double[Lz2];
  }
  Qyzold=new double**[Lx2];
  for (ix=0;ix<Lx2;ix++) {
    Qyzold[ix]=new double*[Ly2];
    for (iy=0;iy<Ly2;iy++)
      Qyzold[ix][iy]=new double[Lz2];
  }
  Qxxnew=new double**[Lx2];
  for (ix=0;ix<Lx2;ix++) {
    Qxxnew[ix]=new double*[Ly2];
    for (iy=0;iy<Ly2;iy++)
      Qxxnew[ix][iy]=new double[Lz2];
  }
  Qxynew=new double**[Lx2];
  for (ix=0;ix<Lx2;ix++) {
    Qxynew[ix]=new double*[Ly2];
    for (iy=0;iy<Ly2;iy++)
      Qxynew[ix][iy]=new double[Lz2];
  }
  Qyynew=new double**[Lx2];
  for (ix=0;ix<Lx2;ix++) {
    Qyynew[ix]=new double*[Ly2];
    for (iy=0;iy<Ly2;iy++)
      Qyynew[ix][iy]=new double[Lz2];
  }
  Qxznew=new double**[Lx2];
  for (ix=0;ix<Lx2;ix++) {
    Qxznew[ix]=new double*[Ly2];
    for (iy=0;iy<Ly2;iy++)
      Qxznew[ix][iy]=new double[Lz2];
  }
  Qyznew=new double**[Lx2];
  for (ix=0;ix<Lx2;ix++) {
    Qyznew[ix]=new double*[Ly2];
    for (iy=0;iy<Ly2;iy++)
      Qyznew[ix][iy]=new double[Lz2];
  }
  Qxxinit=new double**[Lx2];
  for (ix=0;ix<Lx2;ix++) {
    Qxxinit[ix]=new double*[Ly2];
    for (iy=0;iy<Ly2;iy++)
      Qxxinit[ix][iy]=new double[Lz2];
  }
  Qxyinit=new double**[Lx2];
  for (ix=0;ix<Lx2;ix++) {
    Qxyinit[ix]=new double*[Ly2];
    for (iy=0;iy<Ly2;iy++)
      Qxyinit[ix][iy]=new double[Lz2];
  }
  Qyyinit=new double**[Lx2];
  for (ix=0;ix<Lx2;ix++) {
    Qyyinit[ix]=new double*[Ly2];
    for (iy=0;iy<Ly2;iy++)
      Qyyinit[ix][iy]=new double[Lz2];
  }
  Qxzinit=new double**[Lx2];
  for (ix=0;ix<Lx2;ix++) {
    Qxzinit[ix]=new double*[Ly2];
    for (iy=0;iy<Ly2;iy++)
      Qxzinit[ix][iy]=new double[Lz2];
  }
  Qyzinit=new double**[Lx2];
  for (ix=0;ix<Lx2;ix++) {
    Qyzinit[ix]=new double*[Ly2];
    for (iy=0;iy<Ly2;iy++)
      Qyzinit[ix][iy]=new double[Lz2];
  }
  DEHxx=new double**[Lx2];
  for (ix=0;ix<Lx2;ix++) {
    DEHxx[ix]=new double*[Ly2];
    for (iy=0;iy<Ly2;iy++)
      DEHxx[ix][iy]=new double[Lz2];
  }
  DEHxy=new double**[Lx2];
  for (ix=0;ix<Lx2;ix++) {
    DEHxy[ix]=new double*[Ly2];
    for (iy=0;iy<Ly2;iy++)
      DEHxy[ix][iy]=new double[Lz2];
  }
  DEHyy=new double**[Lx2];
  for (ix=0;ix<Lx2;ix++) {
    DEHyy[ix]=new double*[Ly2];
    for (iy=0;iy<Ly2;iy++)
      DEHyy[ix][iy]=new double[Lz2];
  }
  DEHxz=new double**[Lx2];
  for (ix=0;ix<Lx2;ix++) {
    DEHxz[ix]=new double*[Ly2];
    for (iy=0;iy<Ly2;iy++)
      DEHxz[ix][iy]=new double[Lz2];
  }
  DEHyz=new double**[Lx2];
  for (ix=0;ix<Lx2;ix++) {
    DEHyz[ix]=new double*[Ly2];
    for (iy=0;iy<Ly2;iy++)
      DEHyz[ix][iy]=new double[Lz2];
  }
  DEHxxold=new double**[Lx2];
  for (ix=0;ix<Lx2;ix++) {
    DEHxxold[ix]=new double*[Ly2];
    for (iy=0;iy<Ly2;iy++)
      DEHxxold[ix][iy]=new double[Lz2];
  }
  DEHxyold=new double**[Lx2];
  for (ix=0;ix<Lx2;ix++) {
    DEHxyold[ix]=new double*[Ly2];
    for (iy=0;iy<Ly2;iy++)
      DEHxyold[ix][iy]=new double[Lz2];
  }
  DEHyyold=new double**[Lx2];
  for (ix=0;ix<Lx2;ix++) {
    DEHyyold[ix]=new double*[Ly2];
    for (iy=0;iy<Ly2;iy++)
      DEHyyold[ix][iy]=new double[Lz2];
  }
  DEHxzold=new double**[Lx2];
  for (ix=0;ix<Lx2;ix++) {
    DEHxzold[ix]=new double*[Ly2];
    for (iy=0;iy<Ly2;iy++)
      DEHxzold[ix][iy]=new double[Lz2];
  }
  DEHyzold=new double**[Lx2];
  for (ix=0;ix<Lx2;ix++) {
    DEHyzold[ix]=new double*[Ly2];
    for (iy=0;iy<Ly2;iy++)
      DEHyzold[ix][iy]=new double[Lz2];
  }

  DEH1xx=new double**[Lx2];
  for (ix=0;ix<Lx2;ix++) {
    DEH1xx[ix]=new double*[Ly2];
    for (iy=0;iy<Ly2;iy++)
      DEH1xx[ix][iy]=new double[Lz2];
  }
  DEH1xy=new double**[Lx2];
  for (ix=0;ix<Lx2;ix++) {
    DEH1xy[ix]=new double*[Ly2];
    for (iy=0;iy<Ly2;iy++)
      DEH1xy[ix][iy]=new double[Lz2];
  }
  DEH1yy=new double**[Lx2];
  for (ix=0;ix<Lx2;ix++) {
    DEH1yy[ix]=new double*[Ly2];
    for (iy=0;iy<Ly2;iy++)
      DEH1yy[ix][iy]=new double[Lz2];
  }
  DEH1xz=new double**[Lx2];
  for (ix=0;ix<Lx2;ix++) {
    DEH1xz[ix]=new double*[Ly2];
    for (iy=0;iy<Ly2;iy++)
      DEH1xz[ix][iy]=new double[Lz2];
  }
  DEH1yz=new double**[Lx2];
  for (ix=0;ix<Lx2;ix++) {
    DEH1yz[ix]=new double*[Ly2];
    for (iy=0;iy<Ly2;iy++)
      DEH1yz[ix][iy]=new double[Lz2];
  }

  DEH3xx=new double**[Lx2];
  for (ix=0;ix<Lx2;ix++) {
    DEH3xx[ix]=new double*[Ly2];
    for (iy=0;iy<Ly2;iy++)
      DEH3xx[ix][iy]=new double[Lz2];
  }
  DEH3xy=new double**[Lx2];
  for (ix=0;ix<Lx2;ix++) {
    DEH3xy[ix]=new double*[Ly2];
    for (iy=0;iy<Ly2;iy++)
      DEH3xy[ix][iy]=new double[Lz2];
  }
  DEH3yy=new double**[Lx2];
  for (ix=0;ix<Lx2;ix++) {
    DEH3yy[ix]=new double*[Ly2];
    for (iy=0;iy<Ly2;iy++)
      DEH3yy[ix][iy]=new double[Lz2];
  }
  DEH3xz=new double**[Lx2];
  for (ix=0;ix<Lx2;ix++) {
    DEH3xz[ix]=new double*[Ly2];
    for (iy=0;iy<Ly2;iy++)
      DEH3xz[ix][iy]=new double[Lz2];
  }
  DEH3yz=new double**[Lx2];
  for (ix=0;ix<Lx2;ix++) {
    DEH3yz[ix]=new double*[Ly2];
    for (iy=0;iy<Ly2;iy++)
      DEH3yz[ix][iy]=new double[Lz2];
  }

  molfieldxx=new double**[Lx2];
  for (ix=0;ix<Lx2;ix++) {
    molfieldxx[ix]=new double*[Ly2];
    for (iy=0;iy<Ly2;iy++)
      molfieldxx[ix][iy]=new double[Lz2];
  }
  molfieldxy=new double**[Lx2];
  for (ix=0;ix<Lx2;ix++) {
    molfieldxy[ix]=new double*[Ly2];
    for (iy=0;iy<Ly2;iy++)
      molfieldxy[ix][iy]=new double[Lz2];
  }
  molfieldyy=new double**[Lx2];
  for (ix=0;ix<Lx2;ix++) {
    molfieldyy[ix]=new double*[Ly2];
    for (iy=0;iy<Ly2;iy++)
      molfieldyy[ix][iy]=new double[Lz2];
  }
  molfieldxz=new double**[Lx2];
  for (ix=0;ix<Lx2;ix++) {
    molfieldxz[ix]=new double*[Ly2];
    for (iy=0;iy<Ly2;iy++)
      molfieldxz[ix][iy]=new double[Lz2];
  }
  molfieldyz=new double**[Lx2];
  for (ix=0;ix<Lx2;ix++) {
    molfieldyz[ix]=new double*[Ly2];
    for (iy=0;iy<Ly2;iy++)
      molfieldyz[ix][iy]=new double[Lz2];
  }

  DG2xx=new double**[Lx2];
  for (ix=0;ix<Lx2;ix++) {
    DG2xx[ix]=new double*[Ly2];
    for (iy=0;iy<Ly2;iy++)
      DG2xx[ix][iy]=new double[Lz2];
  }
  DG2xy=new double**[Lx2];
  for (ix=0;ix<Lx2;ix++) {
    DG2xy[ix]=new double*[Ly2];
    for (iy=0;iy<Ly2;iy++)
      DG2xy[ix][iy]=new double[Lz2];
  }
  DG2yy=new double**[Lx2];
  for (ix=0;ix<Lx2;ix++) {
    DG2yy[ix]=new double*[Ly2];
    for (iy=0;iy<Ly2;iy++)
      DG2yy[ix][iy]=new double[Lz2];
  }
  DG2xz=new double**[Lx2];
  for (ix=0;ix<Lx2;ix++) {
    DG2xz[ix]=new double*[Ly2];
    for (iy=0;iy<Ly2;iy++)
      DG2xz[ix][iy]=new double[Lz2];
  }
  DG2yz=new double**[Lx2];
  for (ix=0;ix<Lx2;ix++) {
    DG2yz[ix]=new double*[Ly2];
    for (iy=0;iy<Ly2;iy++)
      DG2yz[ix][iy]=new double[Lz2];
  }
  DG2zz=new double**[Lx2];
  for (ix=0;ix<Lx2;ix++) {
    DG2zz[ix]=new double*[Ly2];
    for (iy=0;iy<Ly2;iy++)
      DG2zz[ix][iy]=new double[Lz2];
  }
  DG2yx=new double**[Lx2];
  for (ix=0;ix<Lx2;ix++) {
    DG2yx[ix]=new double*[Ly2];
    for (iy=0;iy<Ly2;iy++)
      DG2yx[ix][iy]=new double[Lz2];
  }
  DG2zy=new double**[Lx2];
  for (ix=0;ix<Lx2;ix++) {
    DG2zy[ix]=new double*[Ly2];
    for (iy=0;iy<Ly2;iy++)
      DG2zy[ix][iy]=new double[Lz2];
  }
  DG2zx=new double**[Lx2];
  for (ix=0;ix<Lx2;ix++) {
    DG2zx[ix]=new double*[Ly2];
    for (iy=0;iy<Ly2;iy++)
      DG2zx[ix][iy]=new double[Lz2];
  }
  tauxy=new double**[Lx2];
  for (ix=0;ix<Lx2;ix++) {
    tauxy[ix]=new double*[Ly2];
    for (iy=0;iy<Ly2;iy++)
      tauxy[ix][iy]=new double[Lz2];
  }
  tauxz=new double**[Lx2];
  for (ix=0;ix<Lx2;ix++) {
    tauxz[ix]=new double*[Ly2];
    for (iy=0;iy<Ly2;iy++)
      tauxz[ix][iy]=new double[Lz2];
  }
  tauyz=new double**[Lx2];
  for (ix=0;ix<Lx2;ix++) {
    tauyz[ix]=new double*[Ly2];
    for (iy=0;iy<Ly2;iy++)
      tauyz[ix][iy]=new double[Lz2];
  }
  Stressxx=new double**[Lx2];
  for (ix=0;ix<Lx2;ix++) {
    Stressxx[ix]=new double*[Ly2];
    for (iy=0;iy<Ly2;iy++)
      Stressxx[ix][iy]=new double[Lz2];
  }
  Stressxy=new double**[Lx2];
  for (ix=0;ix<Lx2;ix++) {
    Stressxy[ix]=new double*[Ly2];
    for (iy=0;iy<Ly2;iy++)
      Stressxy[ix][iy]=new double[Lz2];
  }
  Stressyy=new double**[Lx2];
  for (ix=0;ix<Lx2;ix++) {
    Stressyy[ix]=new double*[Ly2];
    for (iy=0;iy<Ly2;iy++)
      Stressyy[ix][iy]=new double[Lz2];
  }
  Stressxz=new double**[Lx2];
  for (ix=0;ix<Lx2;ix++) {
    Stressxz[ix]=new double*[Ly2];
    for (iy=0;iy<Ly2;iy++)
      Stressxz[ix][iy]=new double[Lz2];
  }
  Stressyz=new double**[Lx2];
  for (ix=0;ix<Lx2;ix++) {
    Stressyz[ix]=new double*[Ly2];
    for (iy=0;iy<Ly2;iy++)
      Stressyz[ix][iy]=new double[Lz2];
  }
  Stresszz=new double**[Lx2];
  for (ix=0;ix<Lx2;ix++) {
    Stresszz[ix]=new double*[Ly2];
    for (iy=0;iy<Ly2;iy++)
      Stresszz[ix][iy]=new double[Lz2];
  }
  Stressyx=new double**[Lx2];
  for (ix=0;ix<Lx2;ix++) {
    Stressyx[ix]=new double*[Ly2];
    for (iy=0;iy<Ly2;iy++)
      Stressyx[ix][iy]=new double[Lz2];
  }
  Stresszx=new double**[Lx2];
  for (ix=0;ix<Lx2;ix++) {
    Stresszx[ix]=new double*[Ly2];
    for (iy=0;iy<Ly2;iy++)
      Stresszx[ix][iy]=new double[Lz2];
  }
  Stresszy=new double**[Lx2];
  for (ix=0;ix<Lx2;ix++) {
    Stresszy[ix]=new double*[Ly2];
    for (iy=0;iy<Ly2;iy++)
      Stresszy[ix][iy]=new double[Lz2];
  }
  Ex=new double**[Lx2];
  for (ix=0;ix<Lx2;ix++) {
    Ex[ix]=new double*[Ly2];
    for (iy=0;iy<Ly2;iy++)
      Ex[ix][iy]=new double[Lz2];
  }
  Ey=new double**[Lx2];
  for (ix=0;ix<Lx2;ix++) {
    Ey[ix]=new double*[Ly2];
    for (iy=0;iy<Ly2;iy++)
      Ey[ix][iy]=new double[Lz2];
  }
  Ez=new double**[Lx2];
  for (ix=0;ix<Lx2;ix++) {
    Ez[ix]=new double*[Ly2];
    for (iy=0;iy<Ly2;iy++)
      Ez[ix][iy]=new double[Lz2];
  }
  Pdx=new double**[Lx2];
  for (ix=0;ix<Lx2;ix++) {
    Pdx[ix]=new double*[Ly2];
    for (iy=0;iy<Ly2;iy++)
      Pdx[ix][iy]=new double[Lz2];
  }
  Pdy=new double**[Lx2];
  for (ix=0;ix<Lx2;ix++) {
    Pdy[ix]=new double*[Ly2];
    for (iy=0;iy<Ly2;iy++)
      Pdy[ix][iy]=new double[Lz2];
  }
  Pdz=new double**[Lx2];
  for (ix=0;ix<Lx2;ix++) {
    Pdz[ix]=new double*[Ly2];
    for (iy=0;iy<Ly2;iy++)
      Pdz[ix][iy]=new double[Lz2];
  }

  e[0][0]= 0;
  e[0][1]= 0;
  e[0][2]= 0;

  e[1][0]= 1;
  e[1][1]= 0;
  e[1][2]= 0;

  e[2][0]= 0;
  e[2][1]= 1;
  e[2][2]= 0;

  e[3][0]= -1;
  e[3][1]= 0;
  e[3][2]= 0;

  e[4][0]= 0;
  e[4][1]= -1;
  e[4][2]= 0;

  e[5][0]= 0;
  e[5][1]= 0;
  e[5][2]= 1;

  e[6][0]= 0;
  e[6][1]= 0;
  e[6][2]= -1;

  e[7][0]= 1;
  e[7][1]= 1;
  e[7][2]= 1;

  e[8][0]= -1;
  e[8][1]= 1;
  e[8][2]= 1;

  e[9][0]= -1;
  e[9][1]= -1;
  e[9][2]= 1;

  e[10][0]= 1;
  e[10][1]= -1;
  e[10][2]= 1;

  e[11][0]= 1;
  e[11][1]= 1;
  e[11][2]= -1;

  e[12][0]= -1;
  e[12][1]= 1;
  e[12][2]= -1;

  e[13][0]= -1;
  e[13][1]= -1;
  e[13][2]= -1;

  e[14][0]= 1;
  e[14][1]= -1;
  e[14][2]= -1;

}


