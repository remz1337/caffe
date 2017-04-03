#!/bin/bash

TOOLS=../../build/tools
WEIGHTS=TVG_CRFRNN_COCO_VOC.caffemodel
SOLVER=TVG_CRFRNN_new_solver.prototxt
MODEL=TVG_CRFRNN_new_traintest.prototxt

$TOOLS/caffe.bin test -model $MODEL -weights $WEIGHTS
