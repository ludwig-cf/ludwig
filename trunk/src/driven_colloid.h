#ifndef DRIVEN_COLLOID_H
#define DRIVEN_COLLOID_H

void driven_colloid_fmod_set(const double f0);
double driven_colloid_fmod_get(void);
void driven_colloid_force(const double s[3], double force[3]);
void driven_colloid_total_force(double ftotal[3]);
int is_driven(void);

#endif
