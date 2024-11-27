#!/usr/bin/bash

i=$1
final=$2
country=$3
batch_id=$4

while [ $i -le $final ];
do
    echo "------------ STARTING EXPERIMENT $i ------------"
    python genera_salidas.py $country $i $batch_id
    ((i++))
done
