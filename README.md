# Project_UOA
Capstone Project UOA

### Correlation of Object Detection Performance with Visual Saliency and Depth Estimation

## Abstract
Accurate object detection remains a critical challenge in computer vision, particularly in complex scenes involving occlusion, scale variation, and small objects. Although modern deep learning models achieve high detection accuracy, understanding how complementary visual tasks contribute to detection performance remains an important research direction. Human visual perception relies on attention and spatial reasoning, motivating the exploration of auxiliary tasks such as visual saliency and depth estimation.

Previous research has explored multi-task learning approaches that combine object detection with saliency or depth estimation, but these studies mainly evaluate end-to-end performance rather than analysing the underlying relationships between tasks. Bartolo et al. investigated correlations between object detection accuracy, visual saliency, and depth estimation using offline experiments and annotated datasets, showing stronger correlations between saliency and detection performance.

The VisualCue Framework extends that analysis by introducing a real-time, annotation-free framework integrating RT-DETR for object detection, Score-CAM for attention mapping, the Segment Anything Model (SAM) for object segmentation, and Zoe Depth for monocular depth estimation. Experiments conducted on the COCO 2017 unlabelled dataset evaluate spatial correlations between attention maps, segmentation masks, and depth representations. Results show stronger alignment between model attention and segmented object regions than with depth cues, supporting the effectiveness of the framework for practical computer vision applications.

## Dataset
The framework was evaluated on the COCO 2017 Unlabelled Dataset, a large-scale benchmark widely used in computer vision research. A subset of 100 images spanning 15 object categories was selected for analysis: Airplane, Bed, Elephant, Giraffe, Pizza, Person, Motorcycle, Parking Meter, Horse, Stop Sign, Zebra, Bear, Cake, Teddy Bear, and Hot Dog. These categories represent a diverse range of object shapes, sizes, and visual complexity. 

## Architecture
<img width="816" height="591" alt="image" src="https://github.com/user-attachments/assets/be06f3d6-22ca-4c14-bacc-a60e3fb32d30" />

## Results
<img width="857" height="767" alt="image" src="https://github.com/user-attachments/assets/9a5f71de-3c8c-46e5-ad2c-6891850ad3a5" />


