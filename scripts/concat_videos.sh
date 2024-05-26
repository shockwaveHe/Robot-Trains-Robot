#!/bin/bash

V1="mujoco.mp4"
V2="IMG_3840.MOV"

ffmpeg -i ${V1} -i ${V2} -filter_complex "\
[0:v]scale=-1:720,pad=ceil(iw/2)*2:ih[v0]; \
[1:v]scale=-1:720,pad=ceil(iw/2)*2:ih[v1]; \
[v0][v1]hstack=inputs=2" -c:v libx264 -shortest output.mp4