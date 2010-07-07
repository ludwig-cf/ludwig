/*****************************************************************************
 *
 *  ewald.c
 *
 *  The Ewald summation for magnetic dipoles. Currently assumes
 *  all dipole strengths (mu) are the same.
 *
 *  See, for example, Allen and Tildesley, Computer Simulation of Liquids.
 *
 *  $Id: ewald.c,v 1.3.16.4 2010-07-07 09:05:14 kevin Exp $
 *
 *  Edinburgh Soft Matter and Statistical Physics Group and
 *  Edinburgh Parallel Computing Centre
 *
 *  Kevin Stratford (kevin@epcc.ed.ac.uk)
 *  (c) 2007 The University of Edinburgh.
 *
 *****************************************************************************/

#include <assert.h>
#include <math.h>
#include <stdlib.h>

#include "pe.h"
#include "coords.h"
#include "colloids.h"
#include "ewald.h"
#include "timer.h"
#include "util.h"

static int ewald_on_ = 0;
static int nk_[3];
static int nkmax_;
static int nktot_;
static double ewald_rc_ = 0.0;
static double kappa_ = 0.0;
static double kmax_;
static double rpi_;
static double mu_;

static double ereal_ = 0.0;
static double efourier_ = 0.0;

static double * sinx_;   /* The term S(k) for each k */
static double * cosx_;   /* The term C(k) for each k */
static double * sinkr_;  /* Table for sin(kr) values */
static double * coskr_;  /* Table for cos(kr) values */

static void ewald_sum_sin_cos_terms(void);
static int  ewald_get_number_fourier_terms(void);
static void ewald_set_kr_table(double []);

/*****************************************************************************
 *
 *  ewald_init
 *
 *  We always have metalic (conducting) boundary conditions at infinity.
 *  The system is assumed to be a cube.
 *
 *  The dipole strength is mu_input.
 *  The real space cut off is rc_input.
 *
 *****************************************************************************/

void ewald_init(double mu_input, double rc_input) {

  int nk;

  /* Set constants */

  rpi_      = 1.0/sqrt(pi_);
  mu_       = mu_input;
  ewald_rc_ = rc_input;
  ewald_on_ = 1;
  kappa_    = 5.0/(2.0*ewald_rc_);

  nk = ceil(kappa_*kappa_*ewald_rc_*L(X)/pi_);

  info("\nThe Ewald sum:\n");
  info("Real space cut off is     %f\n", ewald_rc_);
  info("Ewald parameter kappa is  %f\n", kappa_);
  info("Dipole strength mu is     %f\n", mu_);
  info("Self energy (constant)    %f\n", ewald_self_energy());
  info("Max. term retained in Fourier space sum is %d\n", nk);

  nk_[X] = nk;
  nk_[Y] = nk;
  nk_[Z] = nk;
  kmax_ = pow(2.0*pi_*nk/L(X), 2);
  nkmax_ = nk + 1;
  nktot_ = ewald_get_number_fourier_terms();

  info("maximum square wavevector is %g\n", kmax_);
  info("Total terms retained in Fourier space sum is %d\n\n", nktot_);

  sinx_ = (double *) malloc(nktot_*sizeof(double));
  cosx_ = (double *) malloc(nktot_*sizeof(double));

  if (sinx_ == NULL) fatal("Ewald sum malloc(sinx_) failed\n");
  if (cosx_ == NULL) fatal("Ewald sum malloc(cosx_) failed\n");

  sinkr_ = (double *) malloc(3*nkmax_*sizeof(double));
  coskr_ = (double *) malloc(3*nkmax_*sizeof(double));

  if (sinkr_ == NULL) fatal("Ewald sum malloc(sinx_) failed\n");
  if (coskr_ == NULL) fatal("Ewald sum malloc(cosx_) failed\n");

  return;
}

/*****************************************************************************
 *
 *  ewald_kappa
 *
 *  Return the value of the Ewald parameter kappa.
 *
 *****************************************************************************/

double ewald_kappa(void) {

  assert(ewald_on_);
  return kappa_;
}

/*****************************************************************************
 *
 *  ewald_finish
 *
 *****************************************************************************/

void ewald_finish() {

  if (ewald_on_) {
    free(sinx_);
    free(cosx_);
    free(sinkr_);
    free(coskr_);
  }

  ewald_on_ = 0;

  return;
}

/*****************************************************************************
 *
 *  ewald_sum
 *
 *  A self-contained routine to do the Ewald sum, i.e., accumulate
 *  forces and torques on all the particles. This also computes the
 *  energy, as it's little extra overhead.
 *
 *****************************************************************************/

void ewald_sum() {

  if (ewald_on_) {
    TIMER_start(TIMER_EWALD_TOTAL);

    ewald_fourier_space_sum();
    ewald_real_space_sum();

    TIMER_stop(TIMER_EWALD_TOTAL);
  }

  return;
}

/*****************************************************************************
 *
 *  ewald_real_space_energy
 *
 *  Compute contribution to the energy from single interaction.
 *
 *****************************************************************************/

double ewald_real_space_energy(const double u1[3], const double u2[3],
			       const double r12[3]) {

  double e = 0.0;
  double r;
  double erfc(double); /* ANSI C does not define erfc() in math.h. */

  r = sqrt(r12[X]*r12[X] + r12[Y]*r12[Y] + r12[Z]*r12[Z]);

  if (r < ewald_rc_) {
    double rr = 1.0/r;
    double b, b1, b2, c;

    b1 = mu_*mu_*erfc(kappa_*r)*(rr*rr*rr);
    b2 = mu_*mu_*(2.0*kappa_*rpi_)*exp(-kappa_*kappa_*r*r)*(rr*rr);

    b = b1 + b2;
    c = 3.0*b1*rr*rr + (2.0*kappa_*kappa_ + 3.0*rr*rr)*b2;

    e = dot_product(u1,u2)*b - dot_product(u1,r12)*dot_product(u2,r12)*c;
  }

  return e;
}

/*****************************************************************************
 *
 *  ewald_fourier_space_energy
 *
 *  Fourier-space part of the Ewald summation for the energy.
 *
 *****************************************************************************/

double ewald_fourier_space_energy() {

  double e = 0.0;
  double k[3], ksq;
  double fkx, fky, fkz;
  double b0, b;
  double r4kappa_sq;
  int kx, ky, kz, kn = 0;

  ewald_sum_sin_cos_terms();

  fkx = 2.0*pi_/L(X);
  fky = 2.0*pi_/L(Y);
  fkz = 2.0*pi_/L(Z);
  b0 = (4.0*pi_/(L(X)*L(Y)*L(Z)))*mu_*mu_;
  r4kappa_sq = 1.0/(4.0*kappa_*kappa_);

  /* Sum over k to get the energy. */

  for (kz = 0; kz <= nk_[Z]; kz++) {
    for (ky = -nk_[Y]; ky <= nk_[Y]; ky++) {
      for (kx = -nk_[X]; kx <= nk_[X]; kx++) {

        k[X] = fkx*kx;
        k[Y] = fky*ky;
        k[Z] = fkz*kz;
        ksq = k[X]*k[X] + k[Y]*k[Y] + k[Z]*k[Z];

        if (ksq <= 0.0 || ksq > kmax_) continue;

        b = b0*exp(-r4kappa_sq*ksq)/ksq;
	if (kz == 0) {
	  e += 0.5*b*(sinx_[kn]*sinx_[kn] + cosx_[kn]*cosx_[kn]);
	}
	else {
	  e +=     b*(sinx_[kn]*sinx_[kn] + cosx_[kn]*cosx_[kn]);
	}
	kn++;
      }
    }
  }

  return e;
}

/*****************************************************************************
 *
 *  ewald_sum_sin_cos_terms
 *
 *  For each k, for the Fourier space sum, we need
 *      sinx_ = \sum_i u_i.k sin(k.r_i)    i.e., S(k)
 *      cosx_ = \sum_i u_i.k cos(k.r_i)    i.e., C(k)
 *
 *****************************************************************************/

static void ewald_sum_sin_cos_terms() {

  double k[3], ksq;
  double fkx, fky, fkz;
  int kx, ky, kz, kn = 0;
  int ic, jc, kc;
  int ncell[3];

  double * subsin;
  double * subcos;

  fkx = 2.0*pi_/L(X);
  fky = 2.0*pi_/L(Y);
  fkz = 2.0*pi_/L(Z);

  ncell[X] = Ncell(X);
  ncell[Y] = Ncell(Y);
  ncell[Z] = Ncell(Z);

  /* Comupte S(k) and C(k) from sum over particles */

  for (kn = 0; kn < nktot_; kn++) {
    sinx_[kn] = 0.0;
    cosx_[kn] = 0.0;
  }

  for (ic = 1; ic <= ncell[X]; ic++) {
    for (jc = 1; jc <= ncell[Y]; jc++) {
      for (kc = 1; kc <= ncell[Z]; kc++) {

	Colloid * p_colloid;

	p_colloid = colloids_cell_list(ic, jc, kc);

	while (p_colloid != NULL) {

	  kn = 0;

	  ewald_set_kr_table(p_colloid->s.r);

	  for (kz = 0; kz <= nk_[Z]; kz++) {
	    for (ky = -nk_[Y]; ky <= nk_[Y]; ky++) {
	      for (kx = -nk_[X]; kx <= nk_[X]; kx++) {
		double udotk, kdotr;
		double skr[3], ckr[3];

		k[X] = fkx*kx;
		k[Y] = fky*ky;
		k[Z] = fkz*kz;
		ksq = k[X]*k[X] + k[Y]*k[Y] + k[Z]*k[Z];

		if (ksq <= 0.0 || ksq > kmax_) continue;

		skr[X] = sinkr_[3*abs(kx) + X];
		skr[Y] = sinkr_[3*abs(ky) + Y];
		skr[Z] = sinkr_[3*kz      + Z];
		ckr[X] = coskr_[3*abs(kx) + X];
		ckr[Y] = coskr_[3*abs(ky) + Y];
		ckr[Z] = coskr_[3*kz      + Z];

		if (kx < 0) skr[X] = -skr[X];
		if (ky < 0) skr[Y] = -skr[Y];

		udotk = dot_product(p_colloid->s.s, k);

		/*
		sinx_[kn] += udotk*sin(kdotr);
		cosx_[kn] += udotk*cos(kdotr);
		*/
		kdotr = skr[X]*ckr[Y]*ckr[Z] + ckr[X]*skr[Y]*ckr[Z]
		  + ckr[X]*ckr[Y]*skr[Z] - skr[X]*skr[Y]*skr[Z];
		sinx_[kn] += udotk*kdotr;

		kdotr = ckr[X]*ckr[Y]*ckr[Z] - ckr[X]*skr[Y]*skr[Z]
		  - skr[X]*ckr[Y]*skr[Z] - skr[X]*skr[Y]*ckr[Z];
		cosx_[kn] += udotk*kdotr;

		kn++;
	      }
	    }
	  }
	  p_colloid = p_colloid->next;
	}
	/* Next cell */
      }
    }
  }

  subsin = (double *) calloc(nktot_, sizeof(double));
  subcos = (double *) calloc(nktot_, sizeof(double));
  if (subsin == NULL) fatal("calloc(subsin) failed\n");
  if (subcos == NULL) fatal("calloc(subcos) failed\n");

  for (kn = 0; kn < nktot_; kn++) {
    subsin[kn] = sinx_[kn];
    subcos[kn] = cosx_[kn];
  }

  MPI_Allreduce(subsin, sinx_, nktot_, MPI_DOUBLE, MPI_SUM, cart_comm());
  MPI_Allreduce(subcos, cosx_, nktot_, MPI_DOUBLE, MPI_SUM, cart_comm());

  free(subsin);
  free(subcos);

  return ;
}

/*****************************************************************************
 *
 *  ewald_self_energy
 *
 *  Return the value of the self energy term.
 *
 *****************************************************************************/

double ewald_self_energy() {

  double eself;

  eself = -2.0*mu_*mu_*(kappa_*kappa_*kappa_/(3.0*sqrt(pi_)))*colloid_ntotal();

  return eself;
}

/*****************************************************************************
 *
 *  ewald_total_energy
 *
 *  Return the contributions to the energy.
 *
 *****************************************************************************/

void ewald_total_energy(double * ereal, double * efour, double * eself) {

  if (ewald_on_) {
    *ereal = ereal_;
    *efour = efourier_;
    *eself = ewald_self_energy();
  }
  else {
    *ereal = 0.0;
    *efour = 0.0;
    *eself = 0.0;
  }

  return;
}

/*****************************************************************************
 *
 *  ewald_real_space_sum
 *
 *  Look for interactions in real space and accumulate the force
 *  and torque on each particle involved.
 *
 *****************************************************************************/

void ewald_real_space_sum() {

  Colloid * p_c1;
  Colloid * p_c2;

  int    ic, jc, kc, id, jd, kd, dx, dy, dz;

  double r12[3];

  double erfc(double);

  TIMER_start(TIMER_EWALD_REAL_SPACE);

  ereal_ = 0.0;

  for (ic = 1; ic <= Ncell(X); ic++) {
    for (jc = 1; jc <= Ncell(Y); jc++) {
      for (kc = 1; kc <= Ncell(Z); kc++) {

	p_c1 = colloids_cell_list(ic, jc, kc);

	while (p_c1) {

	  for (dx = -1; dx <= +1; dx++) {
	    for (dy = -1; dy <= +1; dy++) {
	      for (dz = -1; dz <= +1; dz++) {

		id = ic + dx;
		jd = jc + dy;
		kd = kc + dz;

		p_c2 = colloids_cell_list(id, jd, kd);

		while (p_c2) {
		  if (p_c1->s.index < p_c2->s.index) {
		    double r;

		    /* Here we need r2-r1 */

		    coords_minimum_distance(p_c2->s.r, p_c1->s.r, r12);
		    r = sqrt(r12[X]*r12[X] + r12[Y]*r12[Y] + r12[Z]*r12[Z]);

		    if (r < ewald_rc_) {
		      double rr = 1.0/r;
		      double b, b1, b2, c, d;
		      double udotu, u1dotr, u2dotr;
		      double f[3], g[3];
		      int i;

		      /* Energy */
		      b1 = mu_*mu_*erfc(kappa_*r)*(rr*rr*rr);
		      b2 = mu_*mu_*(2.0*kappa_*rpi_)
			*exp(-kappa_*kappa_*r*r)*(rr*rr);

		      b = b1 + b2;
		      c = 3.0*b1*rr*rr + (2.0*kappa_*kappa_ + 3.0*rr*rr)*b2;
		      d = 5.0*c/(r*r)
			+ 4.0*kappa_*kappa_*kappa_*kappa_*b2;

		      udotu  = dot_product(p_c1->s.s, p_c2->s.s);
		      u1dotr = dot_product(p_c1->s.s, r12);
		      u2dotr = dot_product(p_c2->s.s, r12);

		      ereal_ += udotu*b - u1dotr*u2dotr*c;

		      /* Force */

		      for (i = 0; i < 3; i++) {
			f[i] = (udotu*c - u1dotr*u2dotr*d)*r12[i]
			  + c*(u2dotr*p_c1->s.s[i] + u1dotr*p_c2->s.s[i]);
		      }

		      for (i = 0; i < 3; i++) {
			p_c1->force[i] += f[i];
			p_c2->force[i] -= f[i];
		      }

		      /* Torque on particle 1 */

		      g[X] = b*p_c2->s.s[X] - c*u2dotr*r12[X];
		      g[Y] = b*p_c2->s.s[Y] - c*u2dotr*r12[Y];
		      g[Z] = b*p_c2->s.s[Z] - c*u2dotr*r12[Z];

		      p_c1->torque[X] += -(p_c1->s.s[Y]*g[Z] - p_c1->s.s[Z]*g[Y]);
		      p_c1->torque[Y] += -(p_c1->s.s[Z]*g[X] - p_c1->s.s[X]*g[Z]);
		      p_c1->torque[Z] += -(p_c1->s.s[X]*g[Y] - p_c1->s.s[Y]*g[X]);

		      /* Torque on particle 2 */

		      g[X] = b*p_c1->s.s[X] - c*u1dotr*r12[X];
		      g[Y] = b*p_c1->s.s[Y] - c*u1dotr*r12[Y];
		      g[Z] = b*p_c1->s.s[Z] - c*u1dotr*r12[Z];

		      p_c2->torque[X] += -(p_c2->s.s[Y]*g[Z] - p_c2->s.s[Z]*g[Y]);
		      p_c2->torque[Y] += -(p_c2->s.s[Z]*g[X] - p_c2->s.s[X]*g[Z]);
		      p_c2->torque[Z] += -(p_c2->s.s[X]*g[Y] - p_c2->s.s[Y]*g[X]);
		    }
 
		  }

		  p_c2 = p_c2->next;
		}

		/* Next cell */
	      }
	    }
	  }

	  p_c1 = p_c1->next;
	}

	/* Next cell */
      }
    }
  }

  TIMER_stop(TIMER_EWALD_REAL_SPACE);

  return;
}

/*****************************************************************************
 *
 *  ewald_fourier_space_sum
 *
 *  Accumulate the force and torque on each particle arising from
 *  the Fourier space part of the Ewald sum.
 *
 *****************************************************************************/

void ewald_fourier_space_sum() {

  double k[3], ksq;
  double b0, b;
  double fkx, fky, fkz;
  double r4kappa_sq;
  int ic, jc, kc;
  int kx, ky, kz, kn = 0;
  int ncell[3];

  TIMER_start(TIMER_EWALD_FOURIER_SPACE);

  ewald_sum_sin_cos_terms();

  fkx = 2.0*pi_/L(X);
  fky = 2.0*pi_/L(Y);
  fkz = 2.0*pi_/L(Z);
  r4kappa_sq = 1.0/(4.0*kappa_*kappa_);
  b0 = (4.0*pi_/(L(X)*L(Y)*L(Z)))*mu_*mu_;

  ncell[X] = Ncell(X);
  ncell[Y] = Ncell(Y);
  ncell[Z] = Ncell(Z);

  for (ic = 1; ic <= ncell[X]; ic++) {
    for (jc = 1; jc <= ncell[Y]; jc++) {
      for (kc = 1; kc <= ncell[Z]; kc++) {

	Colloid * p_colloid;

	p_colloid = colloids_cell_list(ic, jc, kc);

	while (p_colloid != NULL) {

	  /* Sum over k to get the force/torque. */

	  double f[3], t[3];
	  int i;

	  ewald_set_kr_table(p_colloid->s.r);

	  for (i = 0; i < 3; i++) {
	    f[i] = 0.0;
	    t[i] = 0.0;
	  }

	  efourier_ = 0.0; /* Count only once! */

	  kn = 0;
	  for (kz = 0; kz <= nk_[Z]; kz++) {
	    for (ky = -nk_[Y]; ky <= nk_[Y]; ky++) {
	      for (kx = -nk_[X]; kx <= nk_[X]; kx++) {

		double udotk, g[3];
		double coskr, sinkr, ckr[3], skr[3];

		k[X] = fkx*kx;
		k[Y] = fky*ky;
		k[Z] = fkz*kz;
		ksq = k[X]*k[X] + k[Y]*k[Y] + k[Z]*k[Z];

		if (ksq <= 0.0 || ksq > kmax_) continue;		
		b = b0*exp(-r4kappa_sq*ksq)/ksq;

		/* Energy */ 

		if (kz > 0) b *= 2.0; 
		efourier_ += 0.5*b*(sinx_[kn]*sinx_[kn] + cosx_[kn]*cosx_[kn]);

		skr[X] = sinkr_[3*abs(kx) + X];
		skr[Y] = sinkr_[3*abs(ky) + Y];
		skr[Z] = sinkr_[3*kz      + Z];
		ckr[X] = coskr_[3*abs(kx) + X];
		ckr[Y] = coskr_[3*abs(ky) + Y];
		ckr[Z] = coskr_[3*kz      + Z];

		if (kx < 0) skr[X] = -skr[X];
		if (ky < 0) skr[Y] = -skr[Y];

		sinkr = skr[X]*ckr[Y]*ckr[Z] + ckr[X]*skr[Y]*ckr[Z]
		  + ckr[X]*ckr[Y]*skr[Z] - skr[X]*skr[Y]*skr[Z];

		coskr = ckr[X]*ckr[Y]*ckr[Z] - ckr[X]*skr[Y]*skr[Z]
		  - skr[X]*ckr[Y]*skr[Z] - skr[X]*skr[Y]*ckr[Z];

		/* Force and torque */

		udotk = dot_product(p_colloid->s.s, k);

		for (i = 0; i < 3; i++) {
		  f[i] += b*k[i]*udotk*(cosx_[kn]*sinkr - sinx_[kn]*coskr);
		  g[i] =  b*k[i]*(cosx_[kn]*coskr + sinx_[kn]*sinkr);
		}

		t[X] += -(p_colloid->s.s[Y]*g[Z] - p_colloid->s.s[Z]*g[Y]);
		t[Y] += -(p_colloid->s.s[Z]*g[X] - p_colloid->s.s[X]*g[Z]);
		t[Z] += -(p_colloid->s.s[X]*g[Y] - p_colloid->s.s[Y]*g[X]);

		kn++;
	      }
	    }
	  }

	  /* Accululate force/torque */

	  for (i = 0; i < 3; i++) {
	    p_colloid->force[i] += f[i];
	    p_colloid->torque[i] += t[i];
	  }

	  p_colloid = p_colloid->next;
	}

	/* Next cell */
      }
    }
  }

  TIMER_stop(TIMER_EWALD_FOURIER_SPACE);

  return;
}

/*****************************************************************************
 *
 *  ewald_get_number_fourier_terms
 *
 *  This works out the number of terms that will be required in the
 *  Fourier space sum via a trial run.
 *
 *****************************************************************************/

static int ewald_get_number_fourier_terms() {

  double k[3], ksq;
  double fkx, fky, fkz;
  int kx, ky, kz, kn = 0;

  fkx = 2.0*pi_/L(X);
  fky = 2.0*pi_/L(Y);
  fkz = 2.0*pi_/L(Z);

  for (kz = 0; kz <= nk_[Z]; kz++) {
    for (ky = -nk_[Y]; ky <= nk_[Y]; ky++) {
      for (kx = -nk_[X]; kx <= nk_[X]; kx++) {

        k[0] = fkx*kx;
        k[1] = fky*ky;
        k[2] = fkz*kz;
        ksq = k[0]*k[0] + k[1]*k[1] + k[2]*k[2];

        if (ksq <= 0.0 || ksq > kmax_) continue;
	kn++;
      }
    }
  }

  return kn;
}

/*****************************************************************************
 *
 *  ewald_set_kr_table
 *
 *  For a given particle position r, set the tables of sin(kr) and
 *  cos(kr) for different values of k required via recurrence relation.
 *
 *****************************************************************************/

static void ewald_set_kr_table(double r[3]) {

  int i, k;
  double c2[3];

  for (i = 0; i < 3; i++) {
    sinkr_[3*0 + i] = 0.0;
    coskr_[3*0 + i] = 1.0;
    sinkr_[3*1 + i] = sin(2.0*pi_*r[i]/L(i));
    coskr_[3*1 + i] = cos(2.0*pi_*r[i]/L(i));
    c2[i] = 2.0*coskr_[3*1 + i];
  }

  for (k = 2; k < nkmax_; k++) {
    for (i = 0; i < 3; i++) {
      sinkr_[3*k + i] = c2[i]*sinkr_[3*(k-1) + i] - sinkr_[3*(k-2) + i];
      coskr_[3*k + i] = c2[i]*coskr_[3*(k-1) + i] - coskr_[3*(k-2) + i]; 
    }
  }

  return;
}
