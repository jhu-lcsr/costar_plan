#!/bin/bash
# Source this file to get these utility functions

function find_job() {
  find $HOME/.costar -name '*'$1 2> /dev/null
  find $HOME/work/dev_yb/models -name '*'$1 2> /dev/null
}

function find_job_exact() {
  find $HOME/.costar -name $1 2> /dev/null
  find $HOME/work/dev_yb/models -name $1 2> /dev/null
}

function running_jobs() {
  local jobs=$(sqme | tail -n+3 | awk '{print $1","$6}')

  declare -A jobtimes
  local results=''
  for j in $jobs; do
    local job="${j%,*}"
    local time="${j##*,}"
    jobtimes[$job]=$time
    local results="$results $(find_job_exact $job)"
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
  for j in running_jobs2; do running_jobs[$j]=true; done

  for j in $all_jobs; do
    # get the last part of the dir
    local dir="$(find_job_exact $j)"
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

function del_job_outs() {
  (($# < 1)) && echo Use \'running\' or max_job number && return 1
  local running=false
  if [[ $1 == 'running' ]]; then
    running=true
    declare -A job_table
    local jobs=$(sqme | tail -n+3 | awk '{print $1}')
    for j in $jobs; do job_table[$j]=true; done
  fi
  local files="./slurm-*.out"
  # Find matching files and delete them
  for f in $files; do
    local fjob=${f/\.\/slurm-/}
    fjob=${fjob/\.out/}
    if $running && [[ ${job_table[$fjob]} == true ]]; then
      echo Not deleting $f
    elif ! $running && (($fjob >= $1)); then
      echo Not deleting $f
    else
      echo Deleting $f
      rm $f
    fi
  done
}

# Get the latest images in a directory
function latest_images_dir() {
  # get all prefixes
  declare -A image_prefixes
  for f in $1/*; do
    #echo $f
    local suffix="${f##*/}"
    #echo $suffix
    local prefix="${suffix%_epoch*}"
    #echo $prefix
    image_prefixes[$prefix]=0
  done

  # Get latest epoch per prefix
  for prefix in ${!image_prefixes[@]}; do
    local max=0
    for f in $1/$prefix*; do
      # Isolate number
      local file="${f##*/}"
      file="${file%_result*}"
      local num="${file##*epoch}"
      [[ $num > $max ]] && max=$num
    done
    image_prefixes[$prefix]=$max
  done

  num_images=2
  images=''
  for prefix in ${!image_prefixes[@]}; do
    local max=${image_prefixes[$prefix]}
    local min=$(($max - $num_images))
    (( $min < 0 )) && min=0
    for ((i=$min;i<=$max;i++)); do
      # adjust for 0s in front
      local j=$i
      (( $i < 100 )) && j="0"$j
      (( $i < 10 )) && j="0"$j
      images=$images" $1"/${prefix}_epoch${j}*
    done
  done

  echo $images
}

# show the latest images for a job
function feh_latest_job() {
  local dir2="$(job_dir $1)"
  local job="${dir2##*/}"
  local dir="${dir2%/*}"
  echo "Looking at latest of job $job"
  job_descr_of_dir $dir2
  files=$(latest_images_dir $dir/debug)
  feh $files
}
