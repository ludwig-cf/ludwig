#ifndef _UTILITIES_H
#define _UTILITIES_H

typedef double Float;

typedef struct{
  Float x,y,z;
} FVector;

typedef struct{
  int x,y,z;
} IVector;

#define PI 3.1415926535897932385E0

Float   UTIL_fdistance_sq(FVector r1, FVector r2);
Float   UTIL_fvector_mod(FVector r);
Float   UTIL_dot_product(FVector v1, FVector v2);
FVector UTIL_fvector_zero(void);
FVector UTIL_fvector_add(FVector v1, FVector v2);
FVector UTIL_fvector_subtract(FVector v1, FVector v2);
FVector UTIL_cross_product(FVector v1, FVector v2);
FVector UTIL_rotate_vector(FVector, FVector);

#endif /* _UTILITIES_H */
