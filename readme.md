### 系统配置：

---

ubuntu20.04 

RealSense2 API

OpenCV4.5.5

### 系统结构：

---

1. 使用双目深度相机（intel realsense D435）获取左右眼的红外图像
2. 使用SIFT算法进行特征检测
3. 进行BFMatch暴力匹配筛选匹配结果
4. RANSAC算法计算单应矩阵
5. 完成图像融合

---

###### 编译：

```
. make.sh
```

###### 使用相机时的运行方法：

```
. mrun.sh
```

###### 直接读取两个图像的运行方法：

```
在image文件夹下复制需要读取的两个文件a.jpg与b.jpg
然后执行： ./build/imshow 1
```

