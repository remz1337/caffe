#!/bin/bash

TOOLS=../../build/tools
WEIGHTS=TVG_CRFRNN_COCO_VOC.caffemodel
SOLVER=TVG_CRFRNN_new_solver.prototxt

$TOOLS/caffe.bin train -solver $SOLVER -weights $WEIGHTS
