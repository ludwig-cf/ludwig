all:
	mpirun -np 8 ./Ludwig.exe input 
	python3 extract.py

noex:
	mpirun -np 8 ./Ludwig.exe input 

clean:
	rm config.cds0* 
	#rm dist*
	rm vel*
	rm phi*
	rm col-*
	rm temperature*
	rm map_vel*

remake:
	make clean
	make
