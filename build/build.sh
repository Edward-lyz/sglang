#!/usr/bin/env bash

# This script is used to build the project.

set -e

workspace=`pwd`
echo $workspace

rm -rf output && mkdir output

#ls | grep -v output |awk '{print $1}'|xargs -i{} cp -r {} output/
tar -zcvf ../sglang.tar.gz ../aiak_sglang/
mv ../sglang.tar.gz ./output/
