module reader

  implicit none
  public

  integer, parameter :: dp = kind(1.d0)

contains

  subroutine sq_file(filename, npoints, q, sq)

    character (len = *), intent(in) :: filename
    integer, intent(in)             :: npoints

    real (dp), dimension(npoints), intent(out) :: q
    real (dp), dimension(npoints), intent(out) :: sq

    integer :: n, nd, np

    open (unit = 20, file = filename, form = 'formatted', &
          action = 'read', status = 'old')

    do n = 1, npoints
       read(unit = 20, *) nd, np, q(n), sq(n)
    end do

    close (unit = 20, status = 'keep')
    return

  end subroutine sq_file

  !---------------------------------------------------------------------------
  !
  !  sq_theory
  !
  !  We have S(q) = kT / (A + kappa q^2)
  !
  !  with a correction to q^2 (from Markus)
  !  q^2 = -(1/3) (2*(cos(q_x)*cos(q_y)) + 4*(cos(q_x) + cos(q_y)) - 10)
  !
  !  Our parameters: A = 9.4e-04, B=0.01, K=0.0001, kT = 5e-05
  !
  !---------------------------------------------------------------------------

  subroutine sq_theory(filename)

    character (len = *), intent(in) :: filename

    integer, parameter   :: npoints = 100
    real (dp), parameter :: a = 0.00094
    real (dp), parameter :: kappa = 0.0001
    real (dp), parameter :: kt = 0.00005

    integer   :: n
    real (dp) :: q, qsq, sq

    open (unit = 20, file = filename, form = 'formatted', &
          action = 'write', status = 'new')

    do n = 1, npoints
       q = 4.0*n*atan(1.d0)/npoints
       qsq = q*q ! naive value
       qsq = -(1.0/3.0)*(2.0*(cos(q)*1.0) + 4.0*(cos(q) + 1.0) - 10.0)
       sq = kt / (a + kappa*qsq)
       write(unit = 20, fmt = "(2(1x,e14.7))") q, sq
    end do

    close (unit = 20, status = 'keep')
    return

  end subroutine sq_theory

end module reader


program average

  use reader
  implicit none
 
  integer, parameter :: n = 32
  integer               np

  real (dp), dimension(n) :: sqtot
  real (dp), dimension(n) :: q, sq

  sqtot(:) = 0.0d0

  call sq_file('sq045.dat', n, q, sq)
  sqtot(:) = sqtot(:) + sq(:)

  call sq_file('sq050.dat', n, q, sq)
  sqtot(:) = sqtot(:) + sq(:)

  call sq_file('sq055.dat', n, q, sq)
  sqtot(:) = sqtot(:) + sq(:)

  call sq_file('sq060.dat', n, q, sq)
  sqtot(:) = sqtot(:) + sq(:)

  call sq_file('sq065.dat', n, q, sq)
  sqtot(:) = sqtot(:) + sq(:)

  call sq_file('sq070.dat', n, q, sq)
  sqtot(:) = sqtot(:) + sq(:)

  call sq_file('sq075.dat', n, q, sq)
  sqtot(:) = sqtot(:) + sq(:)

  call sq_file('sq080.dat', n, q, sq)
  sqtot(:) = sqtot(:) + sq(:)

  call sq_file('sq085.dat', n, q, sq)
  sqtot(:) = sqtot(:) + sq(:)

  call sq_file('sq090.dat', n, q, sq)
  sqtot(:) = sqtot(:) + sq(:)

  call sq_file('sq095.dat', n, q, sq)
  sqtot(:) = sqtot(:) + sq(:)

  call sq_file('sq100.dat', n, q, sq)
  sqtot(:) = sqtot(:) + sq(:)

  sqtot(:) = sqtot(:) / 12.0

  ! final result

  open (unit = 20, file = 'sqav.dat', form = 'formatted', &
        action = 'write', status = 'new')

  do np = 1, 25
     write (unit = 20, fmt = '(2(1x,e14.7))') q(np), sq(np)
  end do

  close (unit = 20, status = 'keep')

  call sq_theory('sqth.dat')
  stop


end program average
