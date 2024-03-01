program save_rng_state
    implicit none
    integer, allocatable :: seed(:)
    integer :: n, unit
    real :: x
    character(len=20) :: filename

    ! Initialize the random seed from the system clock
    call random_seed(size = n)
    allocate(seed(n))

    
    ! set a seed
    seed(:) = 1234
    call random_seed(put=seed)
    call random_number(x)
    print*, "x = ", x
    call 

    ! Write the seed to a file
    filename = 'rng_state.txt'
    open(newunit=unit, file=filename, status='unknown')
    write(unit, *) seed
    close(unit)

    print*, "Random number generator state saved to ", filename
end program save_rng_state

