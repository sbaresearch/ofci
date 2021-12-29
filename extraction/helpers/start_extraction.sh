#!/bin/bash

declare -A categories
categories['1']='O0'
categories['2']='O1'
categories['3']='O2'
categories['4']='O3'
categories['5']='bcfobf'
categories['6']='cffobf'
categories['7']='indibran'
categories['8']='splitobf'
categories['9']='subobf'

for target in "${!categories[@]}"
do
    category="${categories[$target]}"
    /usr/bin/time -o "time_${category}" ./extract.sh ${category}
done

