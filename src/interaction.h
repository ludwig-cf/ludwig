/******************************************************************************
 *
 *  interaction.h
 *
 *  Colloid-colloid and colloid-lattce interactions.
 *
 *  Kevin Stratford (kevin@epc.ed.ac.uk)
 *
 ******************************************************************************/

#ifndef _INTERACTION_H
#define _INTERACTION_H

void COLL_compute_phi_gradients(void);
void COLL_compute_phi_fluid(void);

Colloid * COLL_add_colloid(int, double, double, FVector, FVector, FVector);
Colloid * COLL_add_colloid_no_halo(int, double, double, FVector, FVector,
				   FVector);
void      COLL_zero_forces(void);
void      COLL_init(void);
void      COLL_finish(void);

double    COLL_interactions(void);
void      COLL_update(void);
void      COLL_forces(void);

double    soft_sphere_energy(const double);
double    soft_sphere_force(const double);

#endif /* _INTERACTION_H */
