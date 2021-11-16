all:
	mpirun -np 8 ./Ludwig.exe input
	python3 extract.py

clean:
	rm dist*
	rm vel*
	rm phi*
	rm temp*
