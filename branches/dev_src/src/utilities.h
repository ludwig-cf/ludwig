#ifndef _UTILITIES_H
#define _UTILITIES_H


typedef struct{
  double x,y,z;
} FVector;

typedef struct{
  int x,y,z;
} IVector;

#define PI 3.1415926535897932385E0

double   UTIL_fvector_mod(FVector r);
double   UTIL_dot_product(FVector v1, FVector v2);
FVector UTIL_fvector_zero(void);
FVector UTIL_fvector_add(FVector v1, FVector v2);
FVector UTIL_fvector_subtract(FVector v1, FVector v2);
FVector UTIL_cross_product(FVector v1, FVector v2);
FVector UTIL_rotate_vector(FVector, FVector);

double dot_product(const double [], const double []);
void rotate_vector(double [], const double []);

#endif /* _UTILITIES_H */
