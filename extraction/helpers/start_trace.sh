#!/bin/bash

declare -A categories
categories['1']='virt'
categories['2']='virt-ea'

for target in "${!categories[@]}"
do
    category="${categories[$target]}"
    /usr/bin/time -o "time_${category}" ./trace.sh ${category}
done

