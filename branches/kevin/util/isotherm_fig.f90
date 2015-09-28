program isotherm_fig

  use isotherm

  ! Computes equilibrium isotherms
  ! for a number of figures.

  implicit none

  ! Binary fluid parameters
  real (dp), parameter :: A = -0.00625
  real (dp), parameter :: B = -A
  real (dp), parameter :: kappa = 0.016
  real (dp), parameter :: xi0sq = -2.0*kappa/A
  real (dp)            :: xi0

  ! Diamant and Andelman parameters
  real (dp), parameter :: da_alpha = 12.0
  real (dp), parameter :: da_beta  =  0.0
  real (dp), parameter :: da_kt    =  1.0
  real (dp)            :: psi_b = 5.0e-06

  ! Diffuse interface surfactant parameters
  real (dp), parameter :: epsilon = kappa
  real (dp), parameter :: beta = (0.0/24.0)*epsilon
  real (dp), parameter :: kT = epsilon/(24.0*xi0sq)

  real (dp)          :: W = 0.00
  real (dp)          :: psi, psi_c_langmuir
  real (dp)          :: psi_0
  real (dp)          :: sigma0
  real (dp)          :: phi

  real (dp)          :: z0 = 9.0
  real (dp)          :: z

  xi0 = sqrt(xi0sq)
  sigma0 = (4.0/3.0)*kappa/xi0
  psi_c_langmuir = exp((1.0/kT)*(-0.5*epsilon/xi0sq - 0.5*W))

  write (unit = 0, fmt = '(a,e13.6)') 'Interfactial width = ', &
       sqrt(-2.0*kappa/A)

  write (unit = 0, fmt = '(a,e13.6)') 'psi_c (langmuir) ', psi_c_langmuir
  write (unit = 0, fmt = '(a,e13.6)') 'psi0 diamant andelman ', &
       psi_0_diamant(psi_b, da_alpha, da_beta, da_kt)

  phi = psi_frumkin(psi_b, beta, epsilon, W, kT, 0.0d0, 0.0d0, xi0)
  write (unit = 0, fmt = '(a,e13.6)') 'psi0 diffuse ', phi

  !call figx()
  !call fig3()

contains

  !---------------------------------------------------------------------------
  ! Isotherms
  ! beta != 0, ie., general case of Frumpkin adsorption
  !---------------------------------------------------------------------------

  subroutine fig2()

    integer   :: n
    real (dp) :: y
    real (dp) :: psi0_langmuir

    do n = 1, 190
       psi_b = exp(log(0.00001) - log(0.00001)*n/190.)
       psi0_langmuir = psi_b / (psi_b + psi_c_langmuir)

       write (unit = *, fmt = '(3(1x,e14.7))') psi_b, psi0_langmuir, &
            psi_frumkin(psi_b, beta, epsilon, W, kT, 0.0_dp, 0.0_dp, xi0)
    end do

  end subroutine fig2

  !---------------------------------------------------------------------------
  ! Change in surface tension vs psi_0
  !---------------------------------------------------------------------------

  subroutine fig3()

    integer   :: n
    real (dp) :: psi0, psi0_eq
    real (dp) :: sigma
    real (dp) :: psi_c_frumkin

    psi0_eq = psi_frumkin(psi_b, beta, epsilon, W, kT, 0.0d0, 0.0d0, xi0)

    do n = 1, 100
       psi0 = 0.01*n

       if (psi0 < psi0_eq) then

          ! set phi_b to get desired psi0
          psi_c_frumkin = exp((1.0/kT)*(-0.5*epsilon/xi0sq - beta*psi0/xi0sq))
          psi_b = psi_c_frumkin*psi0/(1.0 - psi0)
          call free_energy(sigma, .false.)

          write (unit = *, fmt = '(2(1x,e13.6))') psi0, (sigma - sigma0)/sigma0
       end if
    end do

  end subroutine fig3

  !---------------------------------------------------------------------------
  !  Diamant & Andelman \delta\gamma
  !---------------------------------------------------------------------------

  subroutine fig3a()

    integer   :: n
    real (dp) :: psi0, psi0_eq
    real (dp) :: dsigma_theory

    psi0_eq = psi_0_diamant(psi_b, da_alpha, da_beta, da_kt)

    do n = 1, 100
       psi0 = 0.01*n
       if (psi0 < psi0_eq) then
          dsigma_theory = log(1.0-psi0) + 0.5*(da_beta/da_kt)*psi0*psi0
          write (unit = *, fmt = '(2(1x,e13.6))') psi0, dsigma_theory
       end if
    end do

    return

  end subroutine fig3a

  !---------------------------------------------------------------------------
  ! Equilibrium profiles of various terms in the free energy
  ! We also integrate the excess free energy to get the surface
  ! tension (sigma is returned).
  !---------------------------------------------------------------------------

  subroutine free_energy(sigma, verbose)

    real (dp), intent(out)  :: sigma
    logical, intent(in)     :: verbose

    integer :: n
    real (dp) :: phi_bulk, psi_bulk
    real (dp) :: phi_surf, psi_surf, psi_w
    real (dp) :: delta
    real (dp) :: psi_background
    real (dp) :: ktlocal

    ! Take epsilon = 0 to mean no surfactant, in which case, we
    ! must also set kT = 0 for the energy calculations (only).

    ktlocal = kT
    !if (epsilon < tiny(epsilon)) ktlocal = 0.0

    psi_background = ktlocal*(psi_b*log(psi_b) + (1.0-psi_b)*log(1.0-psi_b))
    sigma = 0.0

    do n = 0, 1900
       z = 0.01*n
       phi = tanh((z0-z)/xi0)
       psi = psi_frumkin(psi_b, beta, epsilon, W, kT, z, z0, xi0)

       delta = 1.0/(xi0sq*cosh((z-z0)/xi0)**4)

       phi_bulk = 0.5*A*phi*phi + 0.25*B*phi*phi*phi*phi
       phi_surf = 0.5*kappa*delta
       psi_bulk = ktlocal*(psi*log(psi) + (1.0 - psi)*log(1.0 - psi))
       psi_surf = -0.5*epsilon*psi*delta -0.5*beta*psi*psi*delta
       psi_w    = 0.5*W*psi*phi*phi

       if (verbose) then
          write (unit = *, fmt = '(9(1x,e12.5))') (z-z0)/xi0, phi, psi, &
               phi_bulk, phi_surf, psi_bulk, psi_surf, psi_w, &
               phi_bulk + phi_surf + psi_bulk + psi_surf
       end if

       sigma = sigma + 0.01*(phi_surf + phi_bulk - (0.5*A + 0.25*B))
       sigma = sigma + 0.01*(psi_surf + psi_bulk - psi_background)
       sigma = sigma + 0.01*(psi_w - 0.5*W*psi_b*1.0*1.0)
    end do

    if (verbose) then
       write (unit = 0, fmt = '(a,e13.6)') 'Interfactial width = ', &
            sqrt(-2.0*kappa/A)
       write (unit = 0, fmt = '(a,e13.6,1x,e13.6)') 'Surface tension = ', &
            sigma, sigma0
       write (unit = 0, fmt = '(a,e13.6)') 'Surfactant bulk background = ', &
            psi_background
    end if

  end subroutine free_energy

end program isotherm_fig
