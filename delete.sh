#!/bin/bash


cd out
for i in *
do
    if [[ -d "$i" ]]; then
        cd "$i"
        for j in *
        do
            rm "$j"
        done
        cd ..
    fi
done

