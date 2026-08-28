/*****************************************************************************
 *
 *  field_impl.h
 *
 *  Static inline function implementations. This file is included in
 *  field.h so that the field_t declaration is available.
 *
 *  Edinburgh Soft Matter and Statistical Physics Group and
 *  Edinburgh Parallel Computing Centre
 *
 *  (c) 2026 The University of Edinburgh
 *
 *  Kevin Stratford (kevin@epcc.ed.ac.uk)
 *
 *****************************************************************************/

#ifndef LUDWIG_FIELD_IMPL_H
#define LUDWIG_FIELD_IMPL_H

#define HDSI_ __host__ __device__ static inline

/*****************************************************************************
 *
 *  field_nf
 *
 *****************************************************************************/

HDSI_ int field_nf(const field_t * obj, int * nf) {

  assert(obj);
  assert(nf);

  *nf = obj->nf;

  return 0;
}

/*****************************************************************************
 *
 *  field_scalar
 *
 *****************************************************************************/

HDSI_ int field_scalar(const field_t * obj, int index, double * phi) {

  assert(obj);
  assert(obj->nf == 1);
  assert(obj->data);
  assert(phi);

  *phi = obj->data[addr_rank1(obj->nsites, 1, index, 0)];

  return 0;
}

/*****************************************************************************
 *
 *  field_scalar_set
 *
 *****************************************************************************/

HDSI_ int field_scalar_set(field_t * obj, int index, double phi) {

  assert(obj);
  assert(obj->nf == 1);
  assert(obj->data);

  obj->data[addr_rank1(obj->nsites, 1, index, 0)] = phi;

  return 0;
}

/*****************************************************************************
 *
 *  field_vector
 *
 *****************************************************************************/

HDSI_ int field_vector(const field_t * obj, int index, double p[3]) {
  assert(obj);
  assert(obj->nf == 3);
  assert(obj->data);

  for (int ia = 0; ia < 3; ia++) {
    p[ia] = obj->data[addr_rank1(obj->nsites, 3, index, ia)];
  }

  return 0;
}

/*****************************************************************************
 *
 *  field_vector_set
 *
 *****************************************************************************/

HDSI_ int field_vector_set(field_t * obj, int index, const double p[3]) {

  assert(obj);
  assert(obj->nf == 3);
  assert(obj->data);
  assert(p);

  for (int ia = 0; ia < 3; ia++) {
    obj->data[addr_rank1(obj->nsites, 3, index, ia)] = p[ia];
  }

  return 0;
}

/*****************************************************************************
 *
 *  field_tensor
 *
 *  The tensor is expanded from the compressed form.
 *
 *****************************************************************************/

HDSI_ int field_tensor(const field_t * obj, int index, double q[3][3]) {

  assert(obj);
  assert(obj->nf == NQAB);
  assert(obj->data);
  assert(q);

  q[X][X] = obj->data[addr_rank1(obj->nsites, NQAB, index, XX)];
  q[X][Y] = obj->data[addr_rank1(obj->nsites, NQAB, index, XY)];
  q[X][Z] = obj->data[addr_rank1(obj->nsites, NQAB, index, XZ)];
  q[Y][X] = q[X][Y];
  q[Y][Y] = obj->data[addr_rank1(obj->nsites, NQAB, index, YY)];
  q[Y][Z] = obj->data[addr_rank1(obj->nsites, NQAB, index, YZ)];
  q[Z][X] = q[X][Z];
  q[Z][Y] = q[Y][Z];
  q[Z][Z] = 0.0 - q[X][X] - q[Y][Y];

  return 0;
}

/*****************************************************************************
 *
 *  field_tensor_set
 *
 *  The tensor supplied should be traceless and symmetric, as it will
 *  be stored in 'compressed' form.
 *
 *****************************************************************************/

HDSI_ int field_tensor_set(field_t * obj, int index, double q[3][3]) {

  assert(obj);
  assert(obj->nf == NQAB);
  assert(obj->data);
  assert(q);

  obj->data[addr_rank1(obj->nsites, NQAB, index, XX)] = q[X][X];
  obj->data[addr_rank1(obj->nsites, NQAB, index, XY)] = q[X][Y];
  obj->data[addr_rank1(obj->nsites, NQAB, index, XZ)] = q[X][Z];
  obj->data[addr_rank1(obj->nsites, NQAB, index, YY)] = q[Y][Y];
  obj->data[addr_rank1(obj->nsites, NQAB, index, YZ)] = q[Y][Z];

  return 0;
}

/*****************************************************************************
 *
 *  field_scalar_array
 *
 *  Return whatever field data there are for this index in a flattened
 *  1d array of length obj->nf.
 *
 *  Array must be of at least obj->nf, but there is no check.
 *
 *****************************************************************************/

HDSI_ int field_scalar_array(const field_t * obj, int index, double * array) {

  assert(obj);
  assert(obj->data);
  assert(array);

  for (int n = 0; n < obj->nf; n++) {
    array[n] = obj->data[addr_rank1(obj->nsites, obj->nf, index, n)];
  }

  return 0;
}

/*****************************************************************************
 *
 *  field_scalar_array_set
 *
 *****************************************************************************/

HDSI_ int field_scalar_array_set(field_t * obj, int index,
                                 const double * array) {

  assert(obj);
  assert(obj->data);
  assert(array);

  for (int n = 0; n < obj->nf; n++) {
    obj->data[addr_rank1(obj->nsites, obj->nf, index, n)] = array[n];
  }

  return 0;
}

#undef HDSI_

#endif
