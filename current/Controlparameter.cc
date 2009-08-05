

/***************************/
/* cut droplets or slabs   */
/* ROI and phase specified */
/* in Initialization.cc    */
/***************************/

//if(n==2000) startDroplet();
//if(n==1500) startSlab();

/********************************/
/* change parameters during run */
/********************************/

// change redshift

/*
if((REDSHIFT==1) && (n<150000)){
   if(O2STRUCT) rr=0.91;
   if(O5STRUCT) rr=0.97;
   if(O8STRUCT) rr=0.82;
   if(O8MSTRUCT) rr=0.83;
}
*/

// temperature-chirality quench 

/*


if(n==20000){

   gam=3.0;
   Abulk=0.0069395656;


   if(myPE==0){
      cout << "timestep " << n << ": changing parameter" << endl;
      cout << "Abulk = " << Abulk << endl;
      cout << "gamma = " << gam << endl;
     }
}


   kappa = sqrt(L1*27.0*q0*q0/(Abulk*gam));
   caz = (1.0+4./3.*kappa*kappa);
   tauc = 1.0/8.0*(1-4.0*kappa*kappa+pow(caz,1.5));

*/

/*

// change noise strength

if(n==450000){noise_strength=1e-05;if(myPE==0){cout << "timestep " << n << ": changing parameter; noise_strength = " << noise_strength << endl;}}
if(n==500000){noise_strength=5e-05;if(myPE==0){cout << "timestep " << n << ": changing parameter; noise_strength = " << noise_strength << endl;}}

*/

/*

// change external E-field

   if(n==410000){delVz= 0.55*5.495388; if(myPE==0){cout << "timestep " << n << ": changing parameter; delVz = " << delVz << endl;}}
   if(n==460000){delVz= 0.5*5.495388; if(myPE==0){cout << "timestep " << n << ": changing parameter; delVz = " << delVz << endl;}}
   if(n==510000){delVz= 0.55*5.495388; if(myPE==0){cout << "timestep " << n << ": changing parameter; delVz = " << delVz << endl;}}
   if(n==560000){delVz= 0.6*5.495388; if(myPE==0){cout << "timestep " << n << ": changing parameter; delVz = " << delVz << endl;}}
   if(n==610000){delVz= 0.65*5.495388; if(myPE==0){cout << "timestep " << n << ": changing parameter; delVz = " << delVz << endl;}}

*/

