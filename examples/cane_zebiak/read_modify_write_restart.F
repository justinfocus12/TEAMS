!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
! Read CZ restart file, modify it by adding noise to the SST and write
! back to a modified restart file. Used for finding extreme El Nino
! events using Dorian's algorithm.
! Eli, 20240702
! INPUT:
!     restart time to read data for (input restart file may contain many times);
!     input restart file name;
!     output restart file name.
! OUTPUT:
!     restart file (single time) with noise added to SST.
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

! Compile:
! $ cd ~/el_nino/CZ_model/
! $ gfortran -std=legacy -O3 -I${HOME}/el_nino/CZ_model -ffixed-line-length-none -o modify_restart.exe read_modify_write_restart.F

! Run:
! $ cd ~/Out/1
! $ ~/el_nino/CZ_model/modify_restart.exe < input.txt
! input.txt has three lines, example is (without the !):
!120.5
!outhst_1
!outhst_001


      IMPLICIT REAL(A-H,O-Z)
      INTEGER OUTHST
      REAL :: noise
      REAL ::  max_noise =.001  ! max noise to add to restart file
      character*8 filename_input
      character*10 filename_output
      INCLUDE 'zeq.common'
 
      COMMON/ZDATA/WM(30,34,2,12),DIVM(30,34,12),Q0O(30,34),
     A SSTM(30,34,12),UO(30,34),VO(30,34),DO(30,34),HTAU(34,30,2),
     B WEM(30,34,12),TO(30,34),UV(30,34,2,12),US(30,34),VS(30,34),
     C WP(30,34),DT1(30,34,7),TT(30,34),UV1(30,34),UV2(30,34),
     D WM1(30,34),UAT(30,34),VAT(30,34),DIVT(30,34)
      COMMON/ZDAT2/H1(30,34),U1(30,34),V1(30,34)
      DATA  INHST/61/,OUTHST/62/

!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
!     read parameters: input and output file, etc
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

      read(5,*) TFIND
      read(5,*) filename_input
      read(5,*) filename_output
      
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
!     Open input and output restart files:
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

      OPEN(UNIT=61, FILE=filename_input, FORM='UNFORMATTED')
      OPEN(UNIT=62, FILE=filename_output, FORM='UNFORMATTED')


!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
!     Read restart:
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

      TD=0.0
      ! read until the specified time step, modify that one and write
      ! only a single restart for that time:
      write (*,*) 'Starting to run, TFIND, TD = ', TFIND, TD
      NSKIP = 3 * (TFIND - TD - 0.5)
      IF(NSKIP .gt. 0) then  
      DO  NNN=1,NSKIP
        READ(INHST) TD,NT,NXP,NYP,NSEG,NX,NY,HEQUIV,TZERO,
     A       TENDD,DTD,XWD,XED,YSD,YND,TDECAY,TPLMIN,TSTPRT,CHKSUM,
     A       NPRINT,MASKW,MWNDGF,WMXW,WMXE,WMYS,WMYN,PERIOD,YNORTH,
     A       YSOUTH,XWEST,NSEG,NTAPE,NREWND,NATM
        NS=NSEG+1
        READ(INHST) HEADER,(AKB(I),I=1,NXP),((UB(J,I),
     A       J=1,NYP),I=1,NXP),((HB(J,I),J=1,NYP),I=1,NXP),(ABC(K),
     A       ROSS(K),TR(K),K=1,NS),((V(J,I),J=1, NYP),I=1,NXP),((UBNDY(J,K),
     A       J=1,NYP),K=1,NS),((HBNDY(J,K),J=1,NYP),K=1,NS)
        READ(INHST) Q0O,UO,VO,DO,TO,U1,V1,H1
      end do
      end if

      
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
!     Modify restart:
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

!     add noise:
      do I=1,30
        do J=1,34
          ! write(*,*) TO(I,J)
          call random_seed()
          call random_number(noise)
          noise = 2.0 * max_noise * (noise - 0.5)
          TO(I,J) = TO(I,J) + noise
          ! write(*,*) TO(I,J)
        end do
      end do
 

!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
!     Write modified restart:
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
 
      WRITE(OUTHST) TD,NT,NXP,NYP,NSEG,NX,NY,HEQUIV,
     A TZERO,TENDD,DTD,XWD,XED,YSD,YND,TDECAY,TPLMIN,TSTPRT,
     A CHKSUM,NPRINT,MASKW,MWNDGF,WMXW,WMXE,WMYS,WMYN,PERIOD,
     A YNORTH,YSOUTH,XWEST,NSEG,NTAPE,NREWND,NATM
      NS=NSEG+1
         WRITE(OUTHST)HEADER,(AKB(I),I=1,NXP),((UB(J,I),
     A J=1,NYP),I=1,NXP),((HB(J,I),J=1,NYP),I=1,NXP),(ABC(K),
     A ROSS(K),TR(K),K=1,NS),((V(J,I),J=1, NYP),I=1,NXP),((UBNDY(J,K),
     A J=1,NYP),K=1,NS),((HBNDY(J,K),J=1,NYP),K=1,NS)
      WRITE(OUTHST) Q0O,UO,VO,DO,TO,U1,V1,H1

      
      stop
      end
