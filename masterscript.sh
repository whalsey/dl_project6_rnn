#!/bin/bash
#
# masterscript.sh
# 
# To use run
# chmod +x masterscript.sh
#
# Then run 
# ./masterscript.sh <python_program_name>.py

if [ $# != 1 ]
then
	echo
	echo "ERROR!"
	echo "One command line argument is required (the"
	echo "name of your python script)."
	echo
	echo "EXAMPLE USAGE"
	echo "./masterscript.sh <python_program_name>.py"
	echo
	echo "OUTPUTS"
	echo "<YYYY-MM-DD_HH-MM-SS>_out.log and "
	echo "<YYYY-MM-DD_HH-MM-SS>_err.log"
	echo
	echo "DESCRIPTION"
	echo "This script will run your program and pipe"
	echo "output to a log as well as print it on the"
	echo "screen. It will also log error messages as"
	echo "well as print them to the screen. Any file"
	echo "IO that occurs within your program will "
	echo "still behave normally. Once the script is "
	echo "complete (whether completing successfully" 
	echo "or crashing) this script will shutdown"
	echo "your instance."
	echo
else

	python $1 > >(tee `date "+%Y-%m-%d_%H-%M-%S_out"`.log) 2> >(tee `date "+%Y-%m-%d_%H-%M-%S_err"`.log)

	#sudo shutdown -h now
	sudo shutdown

fi
