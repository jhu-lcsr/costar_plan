#!/bin/bash
dir="$(find $HOME/.costar -name '*'$1)"
dir="${dir%/*}"
cd $dir




