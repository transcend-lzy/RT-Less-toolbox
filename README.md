## RTL_Toolbox

RTL is a new public dataset of reflective texture-less metal parts. The dataset contains 38392 RGB images and the same number of masks, including 275k(25k true images + 250K Synthetic images) training images and 13312 testing images captured in 32 different scenes. This repository provides some python scripts that may be used when working with RTL dataset.  You can see more details at: [http://www.zju-rtl.cn/RTL/](http://www.zju-rtl.cn/RTL/).

## Requirements

Python >= 3.6

opencv-python == 4.4.0.44

numpy == 1.19.1

ruamel.yaml == 0.16.12

matplotlib == 3.1.2

tqdm == 4.64.0

PIL == 5.2.0

torch>=1.7.1

scipy == 1.5.2

open3d == 0.12.0

pygame == 1.9.6

PyOpenGL==3.1.5

## Scripts

dataset_rtl.py + data_utils.py: 

-  读取图片、ground-truth（yaml格式）、bounding-box(yaml格式)
- 能够实现dataset的功能（返回一份训练或者测试数据）

eval_rtl.py :

- 读取直径（npy文件），并根据以直径的10%为threshold计算add（-s）
- 计算平均旋转和平移误差
- 计算projection_2d误差
- 计算5°5cm误差

convex_hull_3d  + getDiameter.py :

- convex_hull_3d  : 读取ply文件生成最小凸包
- getDiameter.py ：根据最小凸包的顶点集合，计算得到物体直径，即物体表面两个最远点的距离

overlyRender : 将rtl的render数据集的gt渲染出来叠加到原始图片上，即可获得如图1所示的效果

![1](./assest/overlay.png)

​							                                                       图1

overlayTrue：将rtl的真实数据集的ground-truth渲染出来叠加到原始图片上，可以获得如图1所示的效果

overlayYours：按照要求提供图像和ground-truth以及模型的，得到如图1（b）所示效果的图像

read_stl.py : 读取stl文件

overlayUtils.py : overlyRender, overlayTrue, overlayYours等脚本依赖的脚本

randomBackground.py: 随机替换背景，需要提前下载sun数据集

baseline_methods

- psgmn : 基线方法psgmn，内含readme.md
- surfemb : 基线方法surfemb, 内涵readme.md

render_data : 用来生成渲染数据集的脚本，内含readme.md

## References

[1] psgmn

[] [psgmn](https://github.com/Ray0089/PSGMN)

[3] surfemb

[4] [surfemb](https://github.com/rasmushaugaard/surfemb)

[5] [convex_hull_3d  ](https://github.com/swapnil96/Convex-hull)

[6] [pvnet-rendering](https://github.com/zju3dv/pvnet-rendering)

