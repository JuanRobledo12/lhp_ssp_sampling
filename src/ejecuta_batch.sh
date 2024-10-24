#!/usr/bin/bash

i=$1
final=$2

while [ $i -le $final ];
do
    echo "Ejecutamos el experimento $i"
    python genera_salidas.py $i
    ((i++))
done
