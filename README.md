# wantedGen
# Distributed Edge AI Theft Detection System  

Built at **CUHackit (24-hour hackathon)**, this project is a distributed edge-AI system for real-time laptop theft detection using two NVIDIA Jetson devices connected over Ethernet.

## Overview
- **Jetson A** runs YOLO for live object detection, event triggering, and suspect face cropping.
- **Jetson B** performs facial recognition, image processing, and automatically generates a dynamic “wanted” poster.
- The generated alert is published to a live website in real time.

## Architecture
We combined HPC-inspired distributed system design with edge AI by running all models locally.  
The setup simulates both a front-end capture layer and a cloud-style backend — without relying on external servers.

## Key Features
- Real-time computer vision inference
- Inter-device communication over Ethernet
- Automated event pipeline (detect → crop → transmit → recognize → publish)
- Full-stack embedded AI deployment
