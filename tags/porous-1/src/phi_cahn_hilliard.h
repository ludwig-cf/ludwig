/*****************************************************************************
 *
 *  phi_cahn_hilliard.h
 *
 *****************************************************************************/

#ifndef _PHICAHNHILLIARD
#define _PHICAHNHILLIARD

void phi_cahn_hilliard(void);
void phi_ch_set_upwind(void);
void phi_ch_set_utopia(void);

double phi_ch_get_mobility(void);
void   phi_ch_set_mobility(const double);

#endif
