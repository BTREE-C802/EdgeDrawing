# EdgeDrawing
该程序是边缘检测代码，论文是ED那篇，用锚点法，里面的内容对论文还原度很高。参数也想同，锚点设定方法是我自己改进的锚点设定方法。我们将该改进的锚点法用于我们在ORB-SLAM方法基础上改进而来的3DLine-SLAM，图片测试完后会写下三张测试图片，分别是梯度图（GradientImg.jpg）/锚点图（AnchorImg.jpg）/边缘图（SaveImage.png）。注意：在将一张输入图进行边缘描绘测试时，会显示一些列过程图且会保存一些试验结果，如果要将该程序运行于您自己的程序中，则需要将一部分程序段进行删除。

启动方式：./EdgeDrawing ../TestImage/Lena.jpg 
第一项地址为在build下面的启动文件
第二项地址为需要检测的图片的地址（"Lena.jpg"可自由更改，该图片可为自己的数据及其地址）

使用步骤：
第一步：在EdgeDrawing目录下新建一个名为build的文件夹
第二步：cd build
第三步：cmake ..
第四步：make 
则在build下有一个启动方式了。

