/*****************************************************************************
 *
 *  ewald.c
 *
 *  The Ewald summation for magnetic dipoles. Currently assumes
 *  all dipole strengths (mu) are the same.
 *
 *  See, for example, Allen and Tildesley, Computer Simulation of Liquids.
 *
 *  $Id$
 *
 *  Edinburgh Soft Matter and Statistical Physics Group and
 *  Edinburgh Parallel Computing Centre
 *
 *  (c) 2007-2017 The University of Edinburgh.
 *
 *  Contributing authors:
 *  Grace Kim
 *  Kevin Stratford (kevin@epcc.ed.ac.uk)
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

struct ewald_s {
  pe_t * pe;                 /* Parallel environment */
  cs_t * cs;                 /* Coordinate system */
  colloids_info_t * cinfo;   /* Retain a reference to colloids_info_t */
};

static int ewald_sum_sin_cos_terms(ewald_t * ewald);
static int ewald_get_number_fourier_terms(ewald_t * ewald);
static int ewald_set_kr_table(ewald_t * ewlad, double []);

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

int ewald_create(pe_t * pe, cs_t * cs, double mu_input, double rc_input,
		 colloids_info_t * cinfo, ewald_t ** pewald) {
  int nk;
  double ltot[3];
  PI_DOUBLE(pi);
  ewald_t * ewald = NULL;

  assert(pe);
  assert(cs);
  assert(pewald);
  assert(cinfo);

  ewald = (ewald_t *) calloc(1, sizeof(ewald_t));
  if (ewald == NULL) pe_fatal(pe, "calloc(ewald) failed");

  ewald->pe = pe;
  ewald->cs = cs;
  ewald->cinfo = cinfo;

  cs_ltot(cs, ltot);

  /* Set constants */

  rpi_      = 1.0/sqrt(pi);
  mu_       = mu_input;
  ewald_rc_ = rc_input;
  ewald_on_ = 1;
  kappa_    = 5.0/(2.0*ewald_rc_);

  nk = ceil(kappa_*kappa_*ewald_rc_*ltot[X]/pi);

  nk_[X] = nk;
  nk_[Y] = nk;
  nk_[Z] = nk;
  kmax_ = pow(2.0*pi*nk/ltot[X], 2);
  nkmax_ = nk + 1;
  nktot_ = ewald_get_number_fourier_terms(ewald);

  sinx_ = (double *) malloc(nktot_*sizeof(double));
  cosx_ = (double *) malloc(nktot_*sizeof(double));

  if (sinx_ == NULL) pe_fatal(pe, "Ewald sum malloc(sinx_) failed\n");
  if (cosx_ == NULL) pe_fatal(pe, "Ewald sum malloc(cosx_) failed\n");

  sinkr_ = (double *) malloc(3*nkmax_*sizeof(double));
  coskr_ = (double *) malloc(3*nkmax_*sizeof(double));

  if (sinkr_ == NULL) pe_fatal(pe, "Ewald sum malloc(sinx_) failed\n");
  if (coskr_ == NULL) pe_fatal(pe, "Ewald sum malloc(cosx_) failed\n");

  *pewald = ewald;

  return 0;
}

/*****************************************************************************
 *
 *  ewald_free
 *
 *****************************************************************************/

int ewald_free(ewald_t * ewald) {

  assert(ewald);
  free(ewald);

  return 0;
}

/*****************************************************************************
 *
 *  ewald_info
 *
 *****************************************************************************/

int ewald_info(ewald_t * ewald) {

  int ncolloid;
  double eself;

  assert(ewald);

  colloids_info_ntotal(ewald->cinfo, &ncolloid);
  ewald_self_energy(ewald, &eself);

  pe_info(ewald->pe, "\n");
  pe_info(ewald->pe, "Ewald sum\n");
  pe_info(ewald->pe, "---------\n");
  pe_info(ewald->pe, "Number of particles:                      %d\n", ncolloid);
  pe_info(ewald->pe, "Real space cut off:                      %14.7e\n", ewald_rc_);
  pe_info(ewald->pe, "Dipole strength mu:                      %14.7e\n", mu_);
  pe_info(ewald->pe, "Ewald parameter kappa:                   %14.7e\n", kappa_);
  pe_info(ewald->pe, "Self energy (constant):                  %14.7e\n", eself);
  pe_info(ewald->pe, "Maximum square wavevector:               %14.7e\n", kmax_);
  pe_info(ewald->pe, "Max. term retained in Fourier space sum:  %d\n", nkmax_);
  pe_info(ewald->pe, "Total terms kept in Fourier space sum:    %d\n\n", nktot_);

  return 0;
}

/*****************************************************************************
 *
 *  ewald_kappa
 *
 *  Return the value of the Ewald parameter kappa.
 *
 *****************************************************************************/

int ewald_kappa(ewald_t * ewald, double * kappa) {

  assert(ewald);
  assert(kappa);

  *kappa = kappa_;

  return 0;
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
  assert(0); /* TODO clear me up */
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

int ewald_sum(ewald_t * ewald) {

  if (ewald == NULL) return 0;

  TIMER_start(TIMER_EWALD_TOTAL);

  ewald_fourier_space_sum(ewald);
  ewald_real_space_sum(ewald);

  TIMER_stop(TIMER_EWALD_TOTAL);

  return 0;
}

/*****************************************************************************
 *
 *  ewald_real_space_energy
 *
 *  Compute contribution to the energy from single interaction.
 *
 *****************************************************************************/

int ewald_real_space_energy(ewald_t * ewald, const double u1[3],
			    const double u2[3], const double r12[3],
			    double * ereal) {
  double r;

  assert(ewald);
  assert(ereal);

  *ereal = 0.0;

  r = sqrt(r12[X]*r12[X] + r12[Y]*r12[Y] + r12[Z]*r12[Z]);

  if (r < ewald_rc_) {
    double rr = 1.0/r;
    double b, b1, b2, c;

    b1 = mu_*mu_*erfc(kappa_*r)*(rr*rr*rr);
    b2 = mu_*mu_*(2.0*kappa_*rpi_)*exp(-kappa_*kappa_*r*r)*(rr*rr);

    b = b1 + b2;
    c = 3.0*b1*rr*rr + (2.0*kappa_*kappa_ + 3.0*rr*rr)*b2;

    *ereal = dot_product(u1,u2)*b - dot_product(u1,r12)*dot_product(u2,r12)*c;
  }

  return 0;
}

/*****************************************************************************
 *
 *  ewald_fourier_space_energy
 *
 *  Fourier-space part of the Ewald summation for the energy.
 *
 *****************************************************************************/

int ewald_fourier_space_energy(ewald_t * ewald, double * ef) {

  double e = 0.0;
  double k[3], ksq;
  double fkx, fky, fkz;
  double b0, b;
  double r4kappa_sq;
  double ltot[3];
  int kx, ky, kz, kn = 0;
  PI_DOUBLE(pi);

  assert(ewald);

  cs_ltot(ewald->cs, ltot);
  ewald_sum_sin_cos_terms(ewald);

  fkx = 2.0*pi/ltot[X];
  fky = 2.0*pi/ltot[Y];
  fkz = 2.0*pi/ltot[Z];
  b0 = (4.0*pi/(ltot[X]*ltot[Y]*ltot[Z]))*mu_*mu_;
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

  *ef = e;

  return 0;
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

static int ewald_sum_sin_cos_terms(ewald_t * ewald) {

  double k[3], ksq;
  double fkx, fky, fkz;
  int kx, ky, kz, kn = 0;
  int ic, jc, kc;
  int ncell[3];

  double ltot[3];
  double * subsin;
  double * subcos;
  PI_DOUBLE(pi);
  MPI_Comm comm;

  assert(ewald);

  cs_ltot(ewald->cs, ltot);
  colloids_info_ncell(ewald->cinfo, ncell);

  fkx = 2.0*pi/ltot[X];
  fky = 2.0*pi/ltot[Y];
  fkz = 2.0*pi/ltot[Z];

  /* Comupte S(k) and C(k) from sum over particles */

  for (kn = 0; kn < nktot_; kn++) {
    sinx_[kn] = 0.0;
    cosx_[kn] = 0.0;
  }

  for (ic = 1; ic <= ncell[X]; ic++) {
    for (jc = 1; jc <= ncell[Y]; jc++) {
      for (kc = 1; kc <= ncell[Z]; kc++) {

	colloid_t * p_colloid;

	colloids_info_cell_list_head(ewald->cinfo, ic, jc, kc, &p_colloid);

	while (p_colloid != NULL) {

	  kn = 0;

	  ewald_set_kr_table(ewald, p_colloid->s.r);

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
  if (subsin == NULL) pe_fatal(ewald->pe, "calloc(subsin) failed\n");
  if (subcos == NULL) pe_fatal(ewald->pe, "calloc(subcos) failed\n");

  for (kn = 0; kn < nktot_; kn++) {
    subsin[kn] = sinx_[kn];
    subcos[kn] = cosx_[kn];
  }

  cs_cart_comm(ewald->cs, &comm);
  MPI_Allreduce(subsin, sinx_, nktot_, MPI_DOUBLE, MPI_SUM, comm);
  MPI_Allreduce(subcos, cosx_, nktot_, MPI_DOUBLE, MPI_SUM, comm);

  free(subsin);
  free(subcos);

  return 0;
}

/*****************************************************************************
 *
 *  ewald_self_energy
 *
 *  Return the value of the self energy term.
 *
 *****************************************************************************/

int ewald_self_energy(ewald_t * ewald, double * eself) {

  int ntotal;
  PI_DOUBLE(pi);

  assert(ewald);
  colloids_info_ntotal(ewald->cinfo, &ntotal);

  *eself = -2.0*mu_*mu_*(kappa_*kappa_*kappa_/(3.0*sqrt(pi)))*ntotal;

  return 0;
}

/*****************************************************************************
 *
 *  ewald_total_energy
 *
 *  Return the contributions to the energy.
 *
 *****************************************************************************/

int ewald_total_energy(ewald_t * ewald, double * ereal, double * efour,
		       double * eself) {

  if (ewald) {
    *ereal = ereal_;
    *efour = efourier_;
    ewald_self_energy(ewald, eself);
  }
  else {
    *ereal = 0.0;
    *efour = 0.0;
    *eself = 0.0;
  }

  return 0;
}

/*****************************************************************************
 *
 *  ewald_real_space_sum
 *
 *  Look for interactions in real space and accumulate the force
 *  and torque on each particle involved.
 *
 *****************************************************************************/

int ewald_real_space_sum(ewald_t * ewald) {

  colloid_t * p_c1;
  colloid_t * p_c2;

  int ic, jc, kc, id, jd, kd, dx, dy, dz;
  int ncell[3];

  double r12[3];

  TIMER_start(TIMER_EWALD_REAL_SPACE);

  assert(ewald);
  colloids_info_ncell(ewald->cinfo, ncell);

  ereal_ = 0.0;

  for (ic = 1; ic <= ncell[X]; ic++) {
    for (jc = 1; jc <= ncell[Y]; jc++) {
      for (kc = 1; kc <= ncell[Z]; kc++) {

	colloids_info_cell_list_head(ewald->cinfo, ic, jc, kc, &p_c1);

	while (p_c1) {

	  for (dx = -1; dx <= +1; dx++) {
	    for (dy = -1; dy <= +1; dy++) {
	      for (dz = -1; dz <= +1; dz++) {

		id = ic + dx;
		jd = jc + dy;
		kd = kc + dz;

		colloids_info_cell_list_head(ewald->cinfo, id, jd, kd, &p_c2);

		while (p_c2) {
		  if (p_c1->s.index < p_c2->s.index) {
		    double r;

		    /* Here we need r2-r1 */

		    cs_minimum_distance(ewald->cs, p_c2->s.r, p_c1->s.r, r12);
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

  return 0;
}

/*****************************************************************************
 *
 *  ewald_fourier_space_sum
 *
 *  Accumulate the force and torque on each particle arising from
 *  the Fourier space part of the Ewald sum.
 *
 *****************************************************************************/

int ewald_fourier_space_sum(ewald_t * ewald) {

  double k[3], ksq;
  double b0, b;
  double fkx, fky, fkz;
  double r4kappa_sq;
  double ltot[3];

  int ic, jc, kc;
  int kx, ky, kz, kn = 0;
  int ncell[3];
  PI_DOUBLE(pi);

  TIMER_start(TIMER_EWALD_FOURIER_SPACE);

  assert(ewald);

  cs_ltot(ewald->cs, ltot);
  ewald_sum_sin_cos_terms(ewald);

  fkx = 2.0*pi/ltot[X];
  fky = 2.0*pi/ltot[Y];
  fkz = 2.0*pi/ltot[Z];
  r4kappa_sq = 1.0/(4.0*kappa_*kappa_);
  b0 = (4.0*pi/(ltot[X]*ltot[Y]*ltot[Z]))*mu_*mu_;

  colloids_info_ncell(ewald->cinfo, ncell);

  for (ic = 1; ic <= ncell[X]; ic++) {
    for (jc = 1; jc <= ncell[Y]; jc++) {
      for (kc = 1; kc <= ncell[Z]; kc++) {

	colloid_t * p_colloid;

	colloids_info_cell_list_head(ewald->cinfo, ic, jc, kc, &p_colloid);

	while (p_colloid != NULL) {

	  /* Sum over k to get the force/torque. */

	  double f[3], t[3];
	  int i;

	  ewald_set_kr_table(ewald, p_colloid->s.r);

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

  return 0;
}

/*****************************************************************************
 *
 *  ewald_get_number_fourier_terms
 *
 *  This works out the number of terms that will be required in the
 *  Fourier space sum via a trial run.
 *
 *****************************************************************************/

static int ewald_get_number_fourier_terms(ewald_t * ewald) {

  int kx, ky, kz, kn = 0;

  double k[3], ksq;
  double fkx, fky, fkz;
  double ltot[3];
  PI_DOUBLE(pi);

  assert(ewald);

  cs_ltot(ewald->cs, ltot);

  fkx = 2.0*pi/ltot[X];
  fky = 2.0*pi/ltot[Y];
  fkz = 2.0*pi/ltot[Z];

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

static int ewald_set_kr_table(ewald_t * ewald, double r[3]) {

  int i, k;
  double c2[3];
  double ltot[3];
  PI_DOUBLE(pi);

  assert(ewald);

  cs_ltot(ewald->cs, ltot);

  for (i = 0; i < 3; i++) {
    sinkr_[3*0 + i] = 0.0;
    coskr_[3*0 + i] = 1.0;
    sinkr_[3*1 + i] = sin(2.0*pi*r[i]/ltot[i]);
    coskr_[3*1 + i] = cos(2.0*pi*r[i]/ltot[i]);
    c2[i] = 2.0*coskr_[3*1 + i];
  }

  for (k = 2; k < nkmax_; k++) {
    for (i = 0; i < 3; i++) {
      sinkr_[3*k + i] = c2[i]*sinkr_[3*(k-1) + i] - sinkr_[3*(k-2) + i];
      coskr_[3*k + i] = c2[i]*coskr_[3*(k-1) + i] - coskr_[3*(k-2) + i]; 
    }
  }

  return 0;
}
