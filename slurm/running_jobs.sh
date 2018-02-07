#!/bin/bash -l
jobs=$(sqme | tail -n+3 | awk '{print $1}')

results=''
for job in $jobs; do
  results="$results $(find $HOME/.costar -name $job)"
done

for r in $results; do
	job="${r##*/}"
  dir="${r%/*}"
	dir2="${dir##*/}"
  echo "$job $dir2"
done




