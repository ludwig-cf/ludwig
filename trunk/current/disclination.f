      program disc
      implicit none
      integer*4 i,j,k
      double precision order,threshold
      threshold=0.23d0
 10   read(2,*,end=20) i,j,k,order
      if(order.le.threshold) write(8,*) i,j,k 
      goto 10
 20   continue
      stop
      end	
