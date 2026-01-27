#!/bin/bash

module load apptainer

apptainer build train_ssd300_vgg16.sif copernico/train_ssd300_vgg16.def
apptainer build train_fasterrcnn_resnet50.sif copernico/train_fasterrcnn_resnet50.def
apptainer build train_fasterrcnn_resnet50_v2.sif copernico/train_fasterrcnn_resnet50_v2.def
