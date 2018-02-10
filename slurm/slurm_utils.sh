#!/bin/bash
# Source this file to get these utility functions

function running_jobs() {
  jobs=$(sqme | tail -n+3 | awk '{print $1","$6}')

  declare -A jobtimes
  results=''
  for j in $jobs; do
    job="${j%,*}"
    time="${j##*,}"
    jobtimes[$job]=$time
    results="$results $(find $HOME/.costar -name $job)"
  done

  count=0
  for r in $results; do
    job="${r##*/}"
    dir="${r%/*}"
    dir="${dir##*/}"
    echo "$job ${jobtimes[$job]} $dir"
    count=$((count+1))
  done

  echo "$count running jobs"
}

function find_job() {
        find $HOME/.costar -name '*'$1
}

function job_dir() {
        dir="$(find_job $1)"
        dir="${dir%/*}"
        echo $dir
}

function cd_job() {
  dir="$(job_dir $1)"
        cd $dir
}

function feh_job() {
  dir="$(job_dir $1)"
        feh $dir/debug
}

function clean_job_outs() {
  declare -A job_table
  jobs=$(sqme | tail -n+3 | awk '{print $1}')
  for j in $jobs; do job_table[$j]=true; done
  files="./slurm-*.out"
  for f in $files; do
    fjob=${f/\.\/slurm-/}
    fjob=${fjob/\.out/}
    if [[ ${job_table[$fjob]} == true ]]; then
      echo Not deleting $f
    else
      echo Deleting $f
      rm $f
    fi
  done
}






