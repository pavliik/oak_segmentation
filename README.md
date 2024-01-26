# oak_segmentation
***Train a model***  
```python train.py --arch fcn_resnet50 --aux-loss --batch-size 2 --workers 1 --epochs 1 20240126_1```
where "20240126_1" is an example of the output folder for a model in the ./models folder.

***Convert a pytorch model to ONNX format***  
```python convert_model_to_onnx.py --input ./models/20240126_1/best_model.pth --output ./models/model.onnx```

***Convert an ONNX model to blob format for Oak-D***  
```models/20240126_1$ ~/develop/myriad-export/docker-run.sh --input best_model.pth --input-shape '[1, 3, 240, 320]' --output model.blob --reverse-input-channels --mean '[123.675,116.28,103.53]' --scale '[58.4,57.12,57.38]' --nshaves 13 --nslices 13```

***Run the trained model on real Oak-D camera***  
```python oak_real.py --model ./models/20240126_1/model.blob```

***Replay local images on real Oak-D camera using the trained model***  
```python oak_sil.py --model ./models/20240126_1/model.blob --images ./dataset/Images/```
