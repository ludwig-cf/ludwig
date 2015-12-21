##############################################################################
#
#  test-updater.sh
#
#  An interactive script which asks me whether the test log file
#  should be updated if the test has failed.
#
#  Answering 'yes' means I understand that the new result is
#  correct and acceptable.
#
#  cd regression/d3q19
#  ../../test-updater,sh
#
#  Edinburgh Soft Matter and Statistical Physics Group and
#  Edinburgh Parallel Computing Centre
#
#  (c) 2015 The University of Edinburgh
#  Kevin Stratford (kevin@epcc.ed.ac.uk)
#
##############################################################################
#!/bin/bash

for f in *new
do
    input=$f
    stub=`echo $input | sed 's/.new//'`

    # We test for both the log file and the .inp file. If the latter
    # is not present, there is no test in operation

    if [[ -e $stub.log && -e $stub.inp ]]
	then
	../../test-diff.sh -v $stub.log $stub.new
	if [ $? -ne 0 ]
	    then
	    echo -n "Failed: $stub.new -> $stub.log (yes/no): "
	    read RESPONSE
	    if [ "$RESPONSE" = "yes" ]
		then
		# swap the file
		mv $stub.new $stub.log
	    fi
	    echo
	fi
    fi
done
