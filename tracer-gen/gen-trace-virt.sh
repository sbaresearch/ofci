#!/bin/bash
PIN_ROOT=/home/anon/projects/obfuscated-fci/src/tracer/pin-3.20-98437-gf02b61307-gcc-linux
PIN_TOOL=/home/anon/projects/obfuscated-fci/src/tracer/OFCITracer/obj-intel64/OFCITracer.so

for i in {1..70}
do
    echo vt$i
    $PIN_ROOT/pin -t $PIN_TOOL -- virt/virt-1.0-vt$i 4097993829 371910969 3718900527 673787262 471341500
    mv bbl_trace.out virt-traces/virt-1.0-vt$i
done

