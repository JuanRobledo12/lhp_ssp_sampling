#!/usr/bin/bash

i=$1
final=$2
country=$3

while [ $i -le $final ];
do
    echo "------------ COMIENZA EJECUCION DEL EXPERIMENTO $i ------------"
    python genera_salidas.py $country $i
    ((i++))
done
