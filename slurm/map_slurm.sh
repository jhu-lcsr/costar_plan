#!/bin/bash -l
jobs=$(sqme | tail -n+3 | awk '{print $1}')

results=''
for job in $jobs; do
  results="$results $(find $HOME/.costar -name $job)"
done

for r in $results; do
  r2=${r##*/}
  echo $r2
done




