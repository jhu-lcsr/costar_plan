#!/bin/bash
# Source this file to get these utility functions

function running_jobs() {
  local jobs=$(sqme | tail -n+3 | awk '{print $1","$6}')

  declare -A jobtimes
  local results=''
  for j in $jobs; do
    local job="${j%,*}"
    local time="${j##*,}"
    jobtimes[$job]=$time
    local results="$results $(find $HOME/.costar -name $job)"
  done

  local count=0
  for r in $results; do
    local job="${r##*/}"
    local dir="${r%/*}"
    local dir="${dir##*/}"
    echo "$job ${jobtimes[$job]} $dir"
    local count=$((count+1))
  done

  echo "$count running jobs"
}

function job_status() {
  # Get all jobs from out files
  local files="./slurm-*.out"
  local all_jobs=''
  for f in $files; do
    local prefix="${f%.out}"
    local job="${prefix##*slurm-}"
    all_jobs="$all_jobs $job"
  done

  # Get running jobs
  local running_jobs2=$(sqme | tail -n+3 | awk '{print $1}')
  declare -A running_jobs
  for j in running_jobs2; do
    running_jobs[$j]=true
  done

  for j in $all_jobs; do
    # get the last part of the dir
    local dir="$(find $HOME/.costar -name $j)"
    dir="${dir%/*}"
    dir="${dir##*/}"
    local status=UNKNOWN
    if [[ ${running_jobs[$j]} == true ]]; then
      status=RUNNING
    elif grep TIME "./slurm-$j.out" > /dev/null; then
      status=TIMEOUT
    elif grep error "./slurm-$j.out" > /dev/null; then
      status=ERROR
    else
      status=SUCCESS
    fi
    echo $j $status $dir
  done
}

function find_job() {
        find $HOME/.costar -name '*'$1
}

function job_dir() {
        local dir="$(find_job $1)"
        echo $dir
}

function job_descr_of_dir() {
    local dir="${1%/*}"
    dir="${dir##*/}"
    echo $dir
}

function cd_job() {
  local dir="$(job_dir $1)"
  dir="${dir%/*}"
  cd $dir
}

function feh_job() {
  local dir2="$(job_dir $1)"
  local job="${dir2##*/}"
  local dir="${dir2%/*}"
  echo "Looking at job $job"
  job_descr_of_dir $dir2
  if [[ $2 != '' ]]; then
    feh $dir/debug/*$2*
  else
    feh $dir/debug
  fi
}

function clean_job_outs() {
  declare -A job_table
  local jobs=$(sqme | tail -n+3 | awk '{print $1}')
  for j in $jobs; do job_table[$j]=true; done
  local files="./slurm-*.out"
  for f in $files; do
    local fjob=${f/\.\/slurm-/}
    fjob=${fjob/\.out/}
    if [[ ${job_table[$fjob]} == true ]]; then
      echo Not deleting $f
    else
      echo Deleting $f
      rm $f
    fi
  done
}

