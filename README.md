Goal: Action classification from a video by different topologies 
For example, the trained model can idenfify actions liek jump, lunges, cutting vegetables, push-up, brush teeths etc.

Dataset:  UCF101 dataset and own filmed videos.

Topologies for comparison: 
- CNN by RGB input features
- CNN by Optical Flow features
- CNN by Multiple Optical Flow features
- Two-Stream CNN by RGB and Multiple Optical Flow features

Techniques:
- Cross-Validation
- Parameter tuning (Batch size, Learning rate, Dropout ratio, and Training Usage ratio)
- Data augmentation (Reduced resolution, Rotation, Mirroring, Cropping)

Reference:
- UCF101 dataset (here) [http://crcv.ucf.edu/data/UCF101.php]

Result:
 
