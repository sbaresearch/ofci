#!/usr/bin/env python3

if __name__ == '__main__':
    print('#!/bin/sh')
    print('TIGRESS_BIN=$TIGRESS_HOME/tigress')

    print('$TIGRESS_BIN --Seed=42 --Transform=Virtualize --VirtualizeDispatch=switch --Functions=\\')
    for i in range(1000):
        print(f'fn_{i:08},\\')
    print(' --out=virt/$1.c original/$1.c')
