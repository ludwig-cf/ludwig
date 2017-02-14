module isotherm

  !---------------------------------------------------------------------------
  !
  !  These functions compute the equilibrium isotherms.
  !
  !---------------------------------------------------------------------------

  implicit none
  public
  integer, parameter :: dp = kind(1.d0)

contains

  !---------------------------------------------------------------------------
  !
  !  Langmuir isotherm
  !
  !  Computes psi(z) as function of background concentration psi_b,
  !  the free energy parameters, and position z.
  !
  !  Assumes an equilibrium compositional order parameter profile
  !  phi(z) = tanh ((z - z0)/xi0)
  !
  !---------------------------------------------------------------------------

  function psi_langmuir(psi_b, epsilon, w, kt, z, z0, xi0)

    real (dp)             :: psi_langmuir
    real (dp), intent(in) :: psi_b
    real (dp), intent(in) :: epsilon, w, kt
    real (dp), intent(in) :: z, z0, xi0

    real (dp) :: phi, dphi, psi_c

    phi = tanh((z - z0)/xi0)
    dphi = 1.0/(xi0*xi0*cosh((z-z0)/xi0)**4)

    psi_c = exp((1.0/kt)*(-0.5*epsilon*dphi + 0.5*w*(phi*phi - 1)))
    psi_langmuir = psi_b / (psi_b + psi_c)

  end function psi_langmuir

  !--------------------------------------------------------------------------
  !
  !  Frumkin isotherm
  !
  !  Use a bisection method to find a value of psi on [0,1]
  !  which satisfies psi = psi_b / (psi_b + exp(-a - b psi)/kT)
  !  using the equilibrium profile of phi.
  !
  !  The bisection method is from Press et al. chapter 9.
  !
  !--------------------------------------------------------------------------

  function psi_frumkin(psi_b, beta, epsilon, w, kt, z, z0, xi0)

    real (dp)             :: psi_frumkin
    real (dp), intent(in) :: psi_b               ! background concentration
    real (dp), intent(in) :: beta, epsilon, w, kt! free energy parameters
    real (dp), intent(in) :: z, z0, xi0          ! phi = tanh((z-z0)/xi0)

    integer :: nmax = 100
    integer :: n

    real (dp) :: x, dx, xmid, fmid
    real (dp) :: a, b

    a = 0.5*epsilon/(xi0*xi0*cosh((z-z0)/xi0)**4) &
         - 0.5*w*(tanh((z-z0)/xi0)**2 - 1.0)
    b = beta/(xi0*xi0*cosh((z-z0)/xi0)**4)

    ! We know f(x = 1.0) > f(x = 0.0)
    x = 0.0
    dx = 1.0

    do n = 1, nmax
       dx = dx*0.5
       xmid = x + dx
       fmid = xmid*(psi_b + exp(-(a + b*xmid)/kt)) - psi_b
       if (fmid <= 0.0) x = xmid
       if (fmid == 0.0) continue
    end do

    psi_frumkin = xmid
    return

  end function psi_frumkin

  !---------------------------------------------------------------------------
  !
  !  psi_0_diamant
  !
  !  This is again essentially the Frumkin isotherm as used by
  !  Diamant and Andelman J. Phys. Chem 100, 13732 (1996).
  !
  !---------------------------------------------------------------------------

  function psi_0_diamant(psi_b, alpha, beta, kT)

    real (dp)             :: psi_0_diamant
    real (dp), intent(in) :: psi_b               ! background concentration
    real (dp), intent(in) :: alpha, beta, kt     ! free energy parameters

    integer :: nmax = 100
    integer :: n

    real (dp) :: x, dx, xmid, fmid

    ! We know f(x = 1.0) > f(x = 0.0)
    x = 0.0
    dx = 1.0

    do n = 1, nmax
       dx = dx*0.5
       xmid = x + dx
       fmid = xmid*(psi_b + exp(-(alpha + beta*xmid)/kt)) - psi_b
       if (fmid <= 0.0) x = xmid
       if (fmid == 0.0) continue
    end do

    psi_0_diamant = xmid
    return

  end function psi_0_diamant

end module isotherm
