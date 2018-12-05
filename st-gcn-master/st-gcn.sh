#!/bin/bash
video_input_dir="/home/chenhaiquan/openpose/examples/media/"
video_path=$video_input_dir$1
video_input_name=$1
video_output_name=${video_input_name%.*}".mp4"
video_output_path="/home/chenhaiquan/st-gcn-master/data/demo_result/"$video_output_name
# run st-gcn
python main.py demo --openpose /home/chenhaiquan/openpose/build --video $video_path --device 0 1 2
# play video
ffplay $video_output_path

