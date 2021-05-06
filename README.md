# Face-Semantical-Segmentation
Test task

# Task
The main task is to segment face in several classes: background, nose, eyebrows, ears, eyes, face. I used two models (**DeepLab + ResNet** and **DeepLab + MobileNet**) on [http://massimomauro.github.io/FASSEG-repository/](FASSEG dataset).

# Solution
Apparently, the main task was connected with augmentations approaches: I used horizontal flip (face position won't change due to this fact), small RGB-shift (lightning), random crop (in order to limit out image).
Then I encountered with the problem that proposal masks are in shape of rgb-tensors (num_channels x width x height). Therefore, I decided to transform them into the matrix, where each pixel corresponds to distinct class. I solved it via KMeans, since we had several points, where figures are not discrete (not 0, 127, 255). I thought about simple threshold, but it was like a "crutch" (due to the fact that each triple of values are always differ among themselves). 

Below you'll be able to contemplate the obtained results on test script:

| Model | Dataset | Mean IoU | Mean Class Acc | Mean Acc | Execution Time (CPU) | Total estimated model params size (MB)
| ----- | ------- | -------- | -------------- | -------- | -------------------- | -------------------------------------
| **DeepLab + ResNet** | **V2** | 67.086 | 83.675 | 89.854 | 61.017 | 254.553
| **DeepLab + ResNet** | **V3** | 67.579 | 79.318 | 90.618 | 54.496 | 254.553
| **DeepLab + MobileNet** | **V2** | 70.687 | 82.685 | 90.485 | 9.564 | 75.353
| **DeepLab + MobileNet** | **V3** | 58.542 | 69.159 | 87.532 | 9.251 | 75.353

Concerning the execution time **MobileNet** $>$ **ResNet** (in terms of trade-off). In both test scripts **MobileNet** demonstrated better results, but initially during the training part **ResNet** attained higher figures.
