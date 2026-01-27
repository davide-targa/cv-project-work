#!/bin/bash

module load apptainer

apptainer build copernico/train_ssd300_vgg16.sif train_ssd300_vgg16.def
apptainer build copernico/train_fasterrcnn_resnet50.sif train_fasterrcnn_resnet50.def
apptainer build copernico/train_fasterrcnn_resnet50_v2.sif train_fasterrcnn_resnet50_v2.def
