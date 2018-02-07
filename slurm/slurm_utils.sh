#!/bin/bash
# Source this file to get these utility functions

function running_jobs() {
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
}

function find_job() {
	find $HOME/.costar -name '*'$1
}

function cd_job() {
	dir="$(find $HOME/.costar -name '*'$1)"
	dir="${dir%/*}"
	cd $dir
}




