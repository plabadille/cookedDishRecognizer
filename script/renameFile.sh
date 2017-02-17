#!/bin/bash

i=0
for img in *.jpg; do
    ((i++))
    mv $img ${i}".jpg"
done

echo "You have rename $i files"