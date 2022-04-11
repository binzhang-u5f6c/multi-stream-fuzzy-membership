#!/bin/bash

rsync -r fuzzm bzhang3@access.ihpc.uts.edu.au:/home/bzhang3/Running/Code.0$1/
rsync *.py bzhang3@access.ihpc.uts.edu.au:/home/bzhang3/Running/Code.0$1/
