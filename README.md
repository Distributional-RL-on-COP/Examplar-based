# Run Instruction

make sure you are under the directory where has:

```
/img /model
```

### **run default**

```
python model\run.py 
```

choose to do segmentation 

```
python model\run.py --mask_type cut
```

1. First select a part to by the blue window and press `space` buttom
2. Use blue color to paint foreground object and red color to paint background.
3. The pen is initially blue, left click the image to start drawing (put down the pen), and left click the image again to end drawing (put up the pen).
4. Press "x" to change the color of the pen. Press "q" if you finish drawing. Press "a" to increase the pencil size, Press "b" to decrease the pencil size.
5. Wait patiently because this procedure may take some time.

![painted2](\img\sling\painted2.jpg)

### **Arguments to run**
The `--ratio` from 0 to 1, is a parameter is how much you want to fill with Exemplar method, and (1-ratio) range will use Scene Ccompletion method
The `--match_path` is a file folder to store the potential matching pictures.
The `--img_path` is the path where the original picture store
The `--mask_path` is used if you want to use segmentation we generated directly.
The `--write_path` is where you want to save pictures.

### **Samples to refer**
rhino

```
python model\run.py --ratio 0.2 --img_path img\rhino\original.jpg --mask_path img\rhino\mask.jpg --match_path img\rhino\match --write_path img\rhino
```

swiss fill with 2k images  
If you want to do the Scene Completion refer to 2k image dataset, please make sure the original image is **not** in data set.

```
python model\run.py --ratio 0.01 --img_path img\swiss\original.jpg --mask_path img\swiss\mask.jpg --match_path img\2k_random_test --write_path img\swiss
```

swim

```
python model\run.py --ratio 0.2 --img_path img\swim\original.jpg --mask_path img\swim\mask.jpg --match_path img\swim\match --write_path img\swim
```

sling

```
python model\run.py --img_path img\sling\original.jpg --mask_path img\sling\mask.jpg --match_path img\sling\match --write_path img\sling --ratio 1
```

goose

```
python model\run.py --ratio 0.2 --img_path img\goose\original.jpg --mask_path img\goose\mask.jpg --match_path img\goose\match --write_path img\goose
```

jeep

```
python model\run.py --ratio 0.2 --img_path img\jeep\original.jpg --mask_path img\jeep\mask.jpg --match_path img\jeep\match --write_path img\jeep
```

peacock

```
python model\run.py --ratio 0.2 --img_path img\peacock\original.jpg --mask_path img\peacock\mask.jpg --match_path img\peacock\match --write_path img\peacock
```
