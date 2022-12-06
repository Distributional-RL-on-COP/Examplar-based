# examplar-based image inpainting
I, TongZhen, am responsible for the examplar image inpainting part 
## Run
```
python Examplar.py
```
这个项目中使用的一些trick
  1. 用户可以选择在`Inpainter.exe_inpaint`函数选择 `approx=True`作为随机搜索`step`步近似匹配的patch
  2. 待办事项
```
python Musk.py
```
用来获得矩形的

```python
inpainter = Inpainter(img, mask, patch_size, show=False)
```
show = True的时候展示每步填充步骤
- 12-6整理文件
