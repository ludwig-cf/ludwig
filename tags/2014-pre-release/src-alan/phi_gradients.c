/*****************************************************************************
 *
 *  phi_gradients.c
 *
 *  Compute various gradients in the order parameter.
 *
 *  $Id: phi_gradients.c,v 1.12 2010-10-15 12:40:03 kevin Exp $
 *
 *  Edinburgh Soft Matter and Statistical Physics Group and
 *  Edinburgh Parallel Computing Centre
 *
 *  Kevin Stratford (kevin@epcc.ed.ac.uk)
 *  (c) 2007 The University of Edinburgh
 *
 *****************************************************************************/

#include <assert.h>
#include <stdlib.h>

#include "pe.h"
#include "coords.h"
#include "leesedwards.h"
#include "phi.h"
#include "gradient.h"
#include "phi_gradients.h"

static int      level_required_ = 2;
static int      dyadic_required_ = 0;


/* static double * phi_delsq_; */
/* static double * phi_grad_; */
/* static double * phi_delsq_delsq_; */
/* static double * phi_grad_delsq_; */
/* static double * phi_delsq_pp_; */
/* static double * phi_dpp_; */

//need to expose these for targetDP
 double * phi_delsq_;
 double * phi_grad_;
 double * phi_delsq_delsq_;
 double * phi_grad_delsq_;
 double * phi_delsq_pp_;
 double * phi_dpp_;


/****************************************************************************
 *
 *  phi_gradients_init
 *
 ****************************************************************************/

void phi_gradients_init(void) {

  int nop;
  int nhalo;
  int nsites;
  int nlocal[3];

  nop = phi_nop();
  nhalo = coords_nhalo();
  coords_nlocal(nlocal);

  nsites = (nlocal[X] + 2*nhalo + le_get_nxbuffer())*
    (nlocal[Y] + 2*nhalo)*(nlocal[Z] + 2*nhalo);

  if (level_required_ >= 2) {
    phi_delsq_ = (double *) malloc(nop*nsites*sizeof(double));
    phi_grad_ = (double *) malloc(3*nop*nsites*sizeof(double));

    if (phi_delsq_ == NULL) fatal("malloc(phi_delsq_) failed\n");
    if (phi_grad_ == NULL) fatal("malloc(phi_grad_) failed\n");
  }

  if (level_required_ >= 4) {
    assert(phi_nop() == 1); /* Brazovskii only */
    phi_grad_delsq_ = (double *) malloc(3*nsites*sizeof(double));
    phi_delsq_delsq_ = (double *) malloc(nsites*sizeof(double));

    if (phi_grad_delsq_ == NULL) fatal("malloc(phi_grad_delsq_) failed\n");
    if (phi_delsq_delsq_ == NULL) fatal("malloc(phi_delsq_delsq_) failed\n");
  }

  if (dyadic_required_) {
    assert(nop == 3);
    phi_delsq_pp_ = (double *) malloc(18*nop*nsites*sizeof(double));
    phi_dpp_ = (double *) malloc(6*nop*nsites*sizeof(double));

    if (phi_delsq_pp_ == NULL) fatal("malloc(phi_delsq_pp_) failed\n");
    if (phi_dpp_ == NULL) fatal("malloc(phi_dpp_) failed\n");
  }

  return;
}

/****************************************************************************
 *
 *  phi_gradients_finish
 *
 ****************************************************************************/

void phi_gradients_finish(void) {

  if (phi_delsq_) free(phi_delsq_);
  if (phi_grad_) free(phi_grad_);
  if (phi_grad_delsq_) free(phi_grad_delsq_);
  if (phi_delsq_delsq_) free(phi_delsq_delsq_);
  if (phi_delsq_pp_) free(phi_delsq_pp_);
  if (phi_dpp_) free(phi_dpp_);

  return;
}

/****************************************************************************
 *
 *  phi_gradients_level_set
 *
 ****************************************************************************/

void phi_gradients_level_set(const int level) {

  level_required_ = level;
  return;
}

/****************************************************************************
 *
 *  phi_gradients_dyadic_set
 *
 ****************************************************************************/

void phi_gradients_dyadic_set(const int level) {

  dyadic_required_ = level;
  return;
}

/****************************************************************************
 *
 *  phi_gradients_compute
 *
 *  To compute the gradients, the order parameter must be translated
 *  to take account of the Lees Edwards sliding blocks. The gradients
 *  can then be comupted as required.
 *
 ****************************************************************************/

void phi_gradients_compute() {

  int nop;
  extern double * phi_site;

  nop = phi_nop();

  phi_leesedwards_transformation();
  gradient_d2(nop, phi_site, phi_grad_, phi_delsq_);

  if (level_required_ > 2) {
    gradient_d4(nop, phi_delsq_, phi_grad_delsq_, phi_delsq_delsq_);
  }

  if (dyadic_required_) {
    gradient_d2_dyadic(nop, phi_site, phi_dpp_, phi_delsq_pp_);
  }

  return;
}


/*****************************************************************************
 *
 *  phi_gradients_grad_dyadic
 *
 *  Return d_c q_a q_b for vector order parameter.
 *
 *****************************************************************************/

void phi_gradients_grad_dyadic(const int index, double dqq[3][3][3]) {

  int ia;

  assert(phi_dpp_);
  assert(phi_nop() == 3);

  for (ia = 0; ia < 3; ia++) {
    dqq[ia][X][X] = phi_dpp_[18*index + 6*ia + XX];
    dqq[ia][X][Y] = phi_dpp_[18*index + 6*ia + XY];
    dqq[ia][X][Z] = phi_dpp_[18*index + 6*ia + XZ];
    dqq[ia][Y][X] = dqq[X][X][Y];
    dqq[ia][Y][Y] = phi_dpp_[18*index + 6*ia + YY];
    dqq[ia][Y][Z] = phi_dpp_[18*index + 6*ia + YZ];
    dqq[ia][Z][X] = dqq[X][X][Z];
    dqq[ia][Z][Y] = dqq[X][Y][Z];
    dqq[ia][Z][Z] = phi_dpp_[18*index + 6*ia + ZZ];
  }

  return;
}

/*****************************************************************************
 *
 *  phi_gradients_delsq_dyadic
 *
 *  Return nabla^2 q_a q_b for vector order parameter q_a
 *
 *****************************************************************************/

void phi_gradients_delsq_dyadic(const int index, double delsq[3][3]) {

  assert(phi_delsq_pp_);
  assert(phi_nop() == 3);

  delsq[X][X] = phi_delsq_pp_[6*index + XX];
  delsq[X][Y] = phi_delsq_pp_[6*index + XY];
  delsq[X][Z] = phi_delsq_pp_[6*index + XZ];
  delsq[Y][X] = delsq[X][Y];
  delsq[Y][Y] = phi_delsq_pp_[6*index + YY];
  delsq[Y][Z] = phi_delsq_pp_[6*index + YZ];
  delsq[Z][X] = delsq[X][Z];
  delsq[Z][Y] = delsq[Y][Z];
  delsq[Z][Z] = phi_delsq_pp_[6*index + ZZ];

  return;
}

/*****************************************************************************
 *
 *  phi_gradients_grad_delsq
 *
 *****************************************************************************/

void phi_gradients_grad_delsq(const int index, double grad[3]) {

  int ia;

  assert(phi_grad_delsq_);
  assert(phi_nop() == 1);

  for (ia = 0; ia < 3; ia++) {
    grad[ia] = phi_grad_delsq_[3*index + ia];
  }

  return;
}

/*****************************************************************************
 *
 *  phi_gradients_delsq_delsq
 *
 *****************************************************************************/

double phi_gradients_delsq_delsq(const int index) {

  assert(phi_delsq_delsq_);
  assert(phi_nop() == 1);

  return phi_delsq_delsq_[index];
}

/*****************************************************************************
 *
 *  phi_gradients_delsq_n
 *
 *****************************************************************************/

double phi_gradients_delsq_n(const int index, const int nop) {

  assert(phi_delsq_);
  assert(nop < phi_nop());

  return phi_delsq_[phi_nop()*index + nop];
}

/*****************************************************************************
 *
 *  phi_gradients_grad_n
 *
 *****************************************************************************/

void phi_gradients_grad_n(const int index, const int nop, double grad[3]) {

  int ia;

  assert(phi_grad_);
  assert(nop < phi_nop());

  for (ia = 0; ia < 3; ia++) {
    grad[ia] = phi_grad_[3*(phi_nop()*index + nop) + ia];
  }

  return;
}

/*****************************************************************************
 *
 *  phi_gradients_grad
 *
 *****************************************************************************/

void phi_gradients_grad(const int index, double grad[3]) {

  int ia;

  assert(phi_grad_);
  assert(phi_nop() == 1);

  for (ia = 0; ia < 3; ia++) {
    grad[ia] = phi_grad_[3*index + ia];
  }
 
  return;
}

/*****************************************************************************
 *
 *  phi_gradients_delsq
 *
 *****************************************************************************/

double phi_gradients_delsq(const int index) {

  assert(phi_delsq_);
  assert(phi_nop() == 1);

  return phi_delsq_[index];
}

/*****************************************************************************
 *
 *  phi_gradients_vector_delsq
 *
 *  Return \nabla^2 for vector order parameter.
 *
 *****************************************************************************/

void phi_gradients_vector_delsq(const int index, double delsq[3]) {

  int ia;

  assert(phi_delsq_);
  assert(phi_nop() == 3);

  for (ia = 0; ia < 3; ia++) {
    delsq[ia] = phi_delsq_[3*index + ia];
  }

  return;
}

/*****************************************************************************
 *
 *  phi_vector_gradient
 *
 *  Return the gradient tensor for vector order parameter.
 *  This is currently dq[ia][ib] = d_a q_b
 *
 *****************************************************************************/

void phi_gradients_vector_gradient(const int index, double dq[3][3]) {

  int ia, ib;

  assert(phi_grad_);
  assert(phi_nop() == 3);

  for (ia = 0; ia < 3; ia++) {
    for (ib = 0; ib < 3; ib++) {
      /* Address is 3*(nop*index + ib) + ia */ 
      dq[ia][ib] = phi_grad_[3*(3*index + ib) + ia];
    }
  }

  return;
}

/*****************************************************************************
 *
 *  phi_gradients_tensor_gradient
 *
 *  Return the rank 3 gradient tensor of q.
 *
 *****************************************************************************/

void phi_gradients_tensor_gradient(const int index, double dq[3][3][3]) {

  int ia;

  assert(phi_grad_);
  assert(phi_nop() == 5);

  for (ia = 0; ia < 3; ia++) {
    dq[ia][X][X] = phi_grad_[3*(5*index + XX) + ia];
    dq[ia][X][Y] = phi_grad_[3*(5*index + XY) + ia];
    dq[ia][X][Z] = phi_grad_[3*(5*index + XZ) + ia];
    dq[ia][Y][X] = dq[ia][X][Y];
    dq[ia][Y][Y] = phi_grad_[3*(5*index + YY) + ia];
    dq[ia][Y][Z] = phi_grad_[3*(5*index + YZ) + ia];
    dq[ia][Z][X] = dq[ia][X][Z];
    dq[ia][Z][Y] = dq[ia][Y][Z];
    dq[ia][Z][Z] = 0.0 - dq[ia][X][X] - dq[ia][Y][Y];
  }

  return;
}

/*****************************************************************************
 *
 *  phi_gradients_tensor_delsq
 *
 *  Return the delsq Q_ab tensor at site index.
 *
 *****************************************************************************/

void phi_gradients_tensor_delsq(const int index, double dsq[3][3]) {

  assert(phi_delsq_);
  assert(phi_nop() == 5);

  dsq[X][X] = phi_delsq_[5*index + XX];
  dsq[X][Y] = phi_delsq_[5*index + XY];
  dsq[X][Z] = phi_delsq_[5*index + XZ];
  dsq[Y][X] = dsq[X][Y];
  dsq[Y][Y] = phi_delsq_[5*index + YY];
  dsq[Y][Z] = phi_delsq_[5*index + YZ];
  dsq[Z][X] = dsq[X][Z];
  dsq[Z][Y] = dsq[Y][Z];
  dsq[Z][Z] = 0.0 - dsq[X][X] - dsq[Y][Y];

  return;
}
