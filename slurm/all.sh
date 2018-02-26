#!/bin/bash -l

./start_ctp_stacks.sh
./start_ctp_suturing.sh
./start_ctp_husky.sh
./compare_suturing.sh
./compare_stack.sh
./compare_husky.sh
./start_ctp_gans.sh
