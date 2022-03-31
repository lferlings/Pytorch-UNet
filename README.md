# U-Net Implementation zum Lösen von Captchas

Das trainierte Netzwerk kann unter https://uni-muenster.sciebo.de/s/bgipVjcAtUISJWr heruntergeladen werden.

##Detect
Mit ``python detect.py <index>`` kann ein beliebiges Captcha des Testdatasets erkannt werden. Der Pfad zum 
Testset muss in dem Script angepasst werden.

##Verify
Mit ``python verify.py`` kann das Testen gestartet werden, um die Performance des Netzes zu bewerten.
Der Pfad zum Testset und zum zu testenden Netz müssen im Script angepasst werden.

##Training
Das Training kann mit ``train.py`` gestartet werden.
```console
> python train.py -h
usage: train.py [-h] [--epochs E] [--batch-size B] [--learning-rate LR]
                [--load LOAD] [--scale SCALE] [--validation VAL] [--amp]

Train the UNet on images and target masks

optional arguments:
  -h, --help            show this help message and exit
  --epochs E, -e E      Number of epochs
  --batch-size B, -b B  Batch size
  --learning-rate LR, -l LR
                        Learning rate
  --load LOAD, -f LOAD  Load model from a .pth file
  --scale SCALE, -s SCALE
                        Downscaling factor of the images
  --validation VAL, -v VAL
                        Percent of the data that is used as validation (0-100)
  --amp                 Use mixed precision
```
Empfohlene Nutzung:

```console
python train.py --epochs 10 --batch-size 2 -learning-rate 0.001 --validation 20 --amp
```

Das Datenset und die Labelmasken werden werden aus ``./data/imgs/`` und ``./data/masks/`` geladen.


#### Für weitere Infos ``original_README.md`` öffnen.

##Credits

Diese Implementation basiert auf der PyTorch U-Net Implementation von milesial:
https://github.com/milesial/Pytorch-UNet  