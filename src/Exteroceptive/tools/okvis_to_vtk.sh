#!/bin/sh
# Convert OKVIS2 trajectories into VTK polylines for visualization in ParaView.
# Change the color in the last line of the VTK file as desired.
set -eu

if [ "$#" -eq 0 ]
then
	printf 'Usage: %s FILE ...\n' "${0##*/}"
	exit 2
fi

while [ "$#" -gt 0 ]
do
	num_lines=$(wc -l "$1" | cut -f 1 -d ' ')
	awk -v filename="$1" -v num_points=$((num_lines - 1)) '
	BEGIN {
		FS = ", "
		num_points += 0
		print "# vtk DataFile Version 2.0"
		print filename
		print "ASCII"
		print "DATASET POLYDATA"
		printf "\nPOINTS %d float\n", num_points
	}
	NR == 1 {
		for (i=1; i<=NF; i++) {
			if ($i == "p_WS_W_x") x = i
			else if ($i == "p_WS_W_y") y = i
			else if ($i == "p_WS_W_z") z = i
		}
	}
	NR > 1 { printf "%f %f %f\n", $x, $y, $z }
	END {
		printf "\nLINES 1 %d\n", num_points + 1
		printf "%d", num_points
		for (i=0; i<num_points; i++) printf " %d", i
		printf "\n\n"
		print "CELL_DATA 1"
		print "SCALARS color unsigned_char 3"
		print "LOOKUP_TABLE default"
		print "255 0 0"
	}
	' "$1" > "${1%.csv}.vtk"
	shift
done
