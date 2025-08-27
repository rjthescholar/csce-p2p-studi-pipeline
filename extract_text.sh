#!/bin/bash

mkdir "data_text"

mkdir "data_text/CS-0007 Lecture Slides"
mkdir "data_text/CS-0441 Lecture Slides"
mkdir "data_text/CS-0447 Lecture Slides"
mkdir "data_text/CS-0449 Lecture Slides"
mkdir "data_text/CS-1502 Lecture Slides"
mkdir "data_text/CS-1541 Lecture Notes"
mkdir "data_text/CS-1550 Lecture Slides"
mkdir "data_text/CS-1567 Lecture Notes"
mkdir "data_text/CS-1622 Lecture Slides"

cd "./data/CS-0007 Lecture Slides"
for file in *.pdf
do
	echo "${file}"
    name=${file##*/}
    name=${name%.*}
	pdftotext "${file}" "../../data_text/CS-0007 Lecture Slides/${name}.txt"
done

cd "../CS-0441 Lecture Slides"
for file in *.pdf
do
	echo "${file}"
    name=${file##*/}
    name=${name%.*}
	pdftotext "${file}" "../../data_text/CS-0441 Lecture Slides/${name}.txt"
done

cd "../CS-0447 Lecture Slides"
for file in *.pdf
do
	echo "${file}"
    name=${file##*/}
    name=${name%.*}
	pdftotext  "${file}" "../../data_text/CS-0447 Lecture Slides/${name}.txt"
done

cd "../CS-0449 Lecture Slides"
for file in *.pdf
do
	echo "${file}"
    name=${file##*/}
    name=${name%.*}
	pdftotext  "${file}" "../../data_text/CS-0449 Lecture Slides/${name}.txt"
done

cd "../CS-1502 Lecture Slides"
for file in **/*.pdf
do
	echo "${file}"
    name=${file##*/}
    name=${name%.*}
	pdftotext  "${file}" "../../data_text/CS-1502 Lecture Slides/${name}.txt"
done

cd "../CS-1541 Lecture Notes"
for file in *.pdf
do
	echo "${file}"
    name=${file##*/}
    name=${name%.*}
	pdftotext  "${file}" "../../data_text/CS-1541 Lecture Notes/${name}.txt"
done

cd "../CS-1550 Lecture Slides"
for file in *.pdf
do
	echo "${file}"
    name=${file##*/}
    name=${name%.*}
	pdftotext  "${file}" "../../data_text/CS-1550 Lecture Slides/${name}.txt"
done

cd "../CS-1567 Lecture Notes"
for file in *.pdf
do
	echo "${file}"
    name=${file##*/}
    name=${name%.*}
	pdftotext  "${file}" "../../data_text/CS-1567 Lecture Notes/${name}.txt"
done

cd "../CS-1622 Lecture Slides"
for file in *.pdf
do
	echo "${file}"
    name=${file##*/}
    name=${name%.*}
	pdftotext  "${file}" "../../data_text/CS-1622 Lecture Slides/${name}.txt"
done


