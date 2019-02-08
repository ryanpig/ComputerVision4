Goal: Action classification from a video by different topologies 
For example, the trained model can idenfify actions like jumping, lunge, cutting vegetables, push-up, brushing teeths etc.

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
- UCF101 dataset [here](http://crcv.ucf.edu/data/UCF101.php)

Result:
<img src="Result/Accuracy_table_by_actions.jpg">
<img src="Result/Training-Evaluation-Record.jpg">
