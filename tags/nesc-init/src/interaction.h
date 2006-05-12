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

void COLL_bounce_back_pass1(void);
void COLL_bounce_back_pass2(void);
void COLL_compute_phi_gradients(void);
void COLL_compute_phi_fluid(void);
void COLL_remove_or_replace_fluid(void);
void COLL_update_links(void);
void COLL_update_map(void);

#endif /* _INTERACTION_H */
