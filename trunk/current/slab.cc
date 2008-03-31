void startSlab(void)
{
  int i,j,k,l;
  double phase,phase2,amplitude;

  for (i=ix1; i<ix2; i++) {
    for (j=0; j<Ly; j++) {
      for (k=0; k< Lz; k++) {
	if (O2STRUCT == 1) {

	      int iactual;
	      iactual=i;

#ifdef PARALLEL

	      iactual=(i-1)+Lx/nbPE*myPE;

#endif

	      if( pow(j-Ly/2.0,2.0)+pow(iactual-Lx/2.0,2.0)>pow(22.0,2.0)) {

		  amplitude=0.0001;
      
		  Qxx[i][j][k]= amplitude;
		  Qxy[i][j][k]= amplitude;
		  Qyy[i][j][k]= amplitude;
		  Qxz[i][j][k]= amplitude;
		  Qyz[i][j][k]= amplitude;

/*      amplitude=(0.546-0.2723/2.0);
      
      Qxx[i][j][k]=0.2723/2.0+amplitude*cos(2.0*q0*j);
      Qxy[i][j][k]= 0.0;
      Qyy[i][j][k]= -0.2723;
      Qxz[i][j][k]= -amplitude*(sin(2.0*q0*j));
      Qyz[i][j][k]= 0.0;*/


                 }
	}
      }
    }
  }
}
