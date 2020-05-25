#include <iostream>
#include <chrono>

#include <opencv2/opencv.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

using namespace std;
using namespace cv;

#define Horizontal  255//if |dx|<|dy|;
#define Vertical    0//if |dy|<=|dx|;
#define UpDir       1
#define RightDir    2
#define DownDir     3
#define LeftDir     4
#define TryTime     6
#define SkipEdgePoint 2

#define DEBUGEdgeDrawing

int Drawing( InputArray _src,OutputArray _dst,double low_thresh,double high_thresh,bool L2gradient );

int main( int argc,char** argv )
{
	cv::Mat src,dst;
	//src = cv::imread( argv[1],CV_LOAD_IMAGE_COLOR );
	src = cv::imread( argv[1], cv::IMREAD_GRAYSCALE);
	dst.create(src.size(),CV_8U );
	namedWindow( "src",WINDOW_AUTOSIZE );
	imshow( "src",src );
	std::chrono::steady_clock::time_point t1 = chrono::steady_clock::now();
	Drawing(src,dst,10,25,1);
	std::chrono::steady_clock::time_point t2 = chrono::steady_clock::now();
	std::chrono::duration<double> time_used = chrono::duration_cast<chrono::duration<double>>(t2-t1);
	cout << "EdgeDrawing函数所用时" << time_used.count() << "秒" << endl;
	cout << "Canny函数对比开始" << endl;
	t1 = chrono::steady_clock::now();
	Canny( src,dst,30,60,5,1 );
	t2 = chrono::steady_clock::now();
	time_used = chrono::duration_cast<chrono::duration<double>>(t2-t1);
	cout << "Canny函数所用时" << time_used.count() << "秒" << endl;
	cout << "输入图像尺寸为: rows= " << src.rows << " cols = " << src.cols << endl;
	namedWindow( "dst",WINDOW_AUTOSIZE );
	imshow( "dst",dst );
	waitKey(0);
	return 0;
}

int Drawing( InputArray _src,OutputArray _dst,double low_thresh,double high_thresh, bool L2gradient)
{
	Mat src = _src.getMat();
	//输入图像判断
	CV_Assert( src.depth() == CV_8U );
	_dst.create( src.size(),CV_8U );
	Mat dst = _dst.getMat();
	const int cn = src.channels();
	cout << L2gradient <<endl;
	
	//后续使用的参数初始化
	int aperture_size = 3;//孔径数目
	//short int gradienThreshold_ = 80;
	unsigned int imageWidth = src.cols;
	unsigned int imageHeight= src.rows;
	unsigned int gradienThreshold_ = 30; // ***** ORIGINAL WAS 25
	unsigned int pixelNum = imageWidth*imageHeight;//输入图像总像素值
    unsigned int edgePixelArraySize = pixelNum/5;//像素总数的五分之一
	unsigned int maxNumOfEdge = edgePixelArraySize/20;//像素总数的百分之一
	unsigned int scanIntervals = 5;//寻找锚点的扫描高间隙
	unsigned int scanIncreseX = 0;//寻找锚点的扫描间隙X增量
	unsigned int scanIncreseY = 0;//寻找锚点的扫描间隙Y增量
	unsigned char anchorThreshold_ = 8;//锚点与它4领域像素的梯度值之差必须大于anchorThreshold
	int minLineLen_ = 15;//minimal acceptable line length
	
	/*For example, there two edges in the image:
	 *edge1 = [(7,4), (8,5), (9,6),| (10,7)|, (11, 8), (12,9)] and
	 *edge2 = [(14,9), (15,10), (16,11), (17,12),| (18, 13)|, (19,14)] ; then we store them as following:
	 *pFirstPartEdgeX_ = [10, 11, 12, 18, 19];//store the first part of each edge[from middle to end]
	 *pFirstPartEdgeY_ = [7,  8,  9,  13, 14];
	 *pFirstPartEdgeS_ = [0,3,5];// the index of start point of first part of each edge
	 *pSecondPartEdgeX_ = [10, 9, 8, 7, 18, 17, 16, 15, 14];//store the second part of each edge[from middle to front]
	 *pSecondPartEdgeY_ = [7,  6, 5, 4, 13, 12, 11, 10, 9];//anchor points(10, 7) and (18, 13) are stored again
	 *pSecondPartEdgeS_ = [0, 4, 9];// the index of start point of second part of each edge
	 *This type of storage order is because of the order of edge detection process.
	 *For each edge, start from one anchor point, first go right, then go left or first go down, then go up*/
	unsigned int *pFirstPartEdgeX_ = new unsigned int[edgePixelArraySize];//store the X coordinates of the first part of the pixels for chains
	unsigned int *pFirstPartEdgeY_ = new unsigned int[edgePixelArraySize];//store the Y coordinates of the first part of the pixels for chains
	unsigned int *pFirstPartEdgeS_ = new unsigned int[maxNumOfEdge];//store the start index of every edge chain in the first part arrays
	unsigned int *pSecondPartEdgeX_ = new unsigned int[edgePixelArraySize];//store the X coordinates of the second part of the pixels for chains
	unsigned int *pSecondPartEdgeY_ = new unsigned int[edgePixelArraySize];//store the Y coordinates of the second part of the pixels for chains
	unsigned int *pSecondPartEdgeS_ = new unsigned int[maxNumOfEdge];//store the start index of every edge chain in the second part arrays
	unsigned int *pAnchorX_ = new unsigned int[edgePixelArraySize];//储存锚点的X轴坐标，即列数
	unsigned int *pAnchorY_ = new unsigned int[edgePixelArraySize];//储存锚点的Y轴坐标，即行数
	
	//定义边缘图像
	cv::Mat edgeImage_( imageHeight, imageWidth, CV_8UC1 );
	//定义高斯滤波后的矩阵
	Mat src_gauss( imageHeight, imageWidth, CV_8UC(cn) );
	//定义两个装梯度的矩阵
	Mat dxImg_( imageHeight, imageWidth, CV_8UC1 );
	Mat dyImg_( imageHeight, imageWidth, CV_8UC1 );
	//显示用的容器
	Mat Img_view( imageHeight, imageWidth, CV_8UC(cn) );
	Mat dx_view( imageHeight, imageWidth, CV_8UC(cn) );
	Mat dy_view( imageHeight, imageWidth, CV_8UC(cn) );
	//定义x，y方向灰度值融合矩阵容器
	Mat dxy_grad( imageHeight, imageWidth, CV_8UC(cn) );
	//存放xy梯度范数的容器
	Mat dxyNorm( imageWidth, imageHeight,CV_16UC(cn) );
	//梯度值取绝对值容器
	cv::Mat dxABS_max( imageHeight, imageWidth, CV_16UC1 );//三个通道中梯度值最大处绝对值
	cv::Mat dyABS_max( imageHeight, imageWidth, CV_16UC1 );//三个通道中梯度值最大处绝对值
	cv::Mat SumDxDy( imageHeight, imageWidth, CV_16UC1 );//各通道梯度值绝对值求和，一范数或者二范数
	cv::Mat gImgWO_( imageHeight, imageWidth, CV_16UC(cn) );
	//高低阈值筛选
	cv::Mat gImg_( imageHeight, imageWidth,CV_16UC1 );//求得最大范数的子集图，因为有阈值将梯度小的地方滤除了,只有一个通道
	cv::Mat dirImg_;//储存梯度方向的图像
	//步骤一
	//高斯平滑滤波，3×3
	GaussianBlur( src, src_gauss, Size(3, 3), 1 );

	//步骤二
	//计算梯度值并且计算边缘方向
	//compute dx, dy images, gImag_ = gradient image
	if(gImg_.cols!= imageWidth||gImg_.rows!= imageHeight)
	{
		if(pFirstPartEdgeX_!= NULL)
		{
			delete [] pFirstPartEdgeX_;
			delete [] pFirstPartEdgeY_;
			delete [] pSecondPartEdgeX_;
			delete [] pSecondPartEdgeY_;
			delete [] pFirstPartEdgeS_;
			delete [] pSecondPartEdgeS_;
			delete [] pAnchorX_;
			delete [] pAnchorY_;
		}
		dxImg_.create(imageHeight, imageWidth, CV_16SC1);
		dyImg_.create(imageHeight, imageWidth, CV_16SC1 );
		gImgWO_.create(imageHeight, imageWidth, CV_16SC1 );
		gImg_.create(imageHeight, imageWidth, CV_16SC1 );
		dirImg_.create(imageHeight, imageWidth, CV_8UC1 );
		edgeImage_.create(imageHeight, imageWidth, CV_8UC1 );
		pFirstPartEdgeX_ = new unsigned int[edgePixelArraySize];//对pFirstPartEdgeX_开辟edgePixelArraySize个无符号整形空间
		pFirstPartEdgeY_ = new unsigned int[edgePixelArraySize];
		pSecondPartEdgeX_ = new unsigned int[edgePixelArraySize];
		pSecondPartEdgeY_ = new unsigned int[edgePixelArraySize];
		pFirstPartEdgeS_ = new unsigned int[maxNumOfEdge];
		pSecondPartEdgeS_ = new unsigned int[maxNumOfEdge];
		pAnchorX_ = new unsigned int[edgePixelArraySize];//锚点pAnchorX_
		pAnchorY_ = new unsigned int[edgePixelArraySize];//锚点pAnchorX_
	}
	//Sobel( src_gauss,dxImg_,CV_8UC1, 1, 0, 3 );//对平滑后的图像dx方向sobel求梯度，算子尺寸为3
	//Sobel( src_gauss,dyImg_,CV_8UC1, 0, 1, 3 );//对平滑后的图像dy方向sobel求梯度，算子尺寸为3
	cv::Sobel( src_gauss, dxImg_, CV_16SC1, 1, 0, 3);//对平滑后的图像dx方向sobel求梯度，算子尺寸为3
	cv::Sobel( src_gauss, dyImg_, CV_16SC1, 0, 1, 3);//对平滑后的图像dy方向sobel求梯度，算子尺寸为3
#ifdef DEBUGEdgeDrawing
	namedWindow( "src_gauss",CV_WINDOW_AUTOSIZE );
        cv::imshow("src_gauss", src_gauss);
#endif
	
	//compute gradient and direction images
	double t = (double)cv::getTickCount();//记录当前系统时间保存在双精度变量t内
	cv::Mat dxABS_m = cv::abs(dxImg_);//对梯度值取绝对值
	cv::Mat dyABS_m = cv::abs(dyImg_);
#ifdef DEBUGEdgeDrawing
	namedWindow( "dxImg_1",CV_WINDOW_AUTOSIZE );
        cv::imshow("dxImg_1", dxABS_m);
	namedWindow( "dyImg_2",CV_WINDOW_AUTOSIZE );
        cv::imshow("dyImg_2", dyABS_m);
#endif
	cv::Mat sumDxDy;//梯度值一范数图
	cv::add(dyABS_m, dxABS_m, sumDxDy);//矩阵相加函数,???难道不会超出范围？
	
	cv::threshold(sumDxDy,gImg_, gradienThreshold_+1, 255, cv::THRESH_TOZERO);//阈值处理后gImg_图
	gImg_ = gImg_/4;
	gImgWO_ = sumDxDy/4;
	cv::compare(dxABS_m, dyABS_m, dirImg_, cv::CMP_LT);//水平方向梯度变化则值为255,垂直方向大则为0,dst(i)=src1(i) cmpop src2(i)
#ifdef DEBUGEdgeDrawing
	namedWindow( "方向图",CV_WINDOW_AUTOSIZE );
	imshow( "方向图",dirImg_ );
#endif
	t = ((double)cv::getTickCount() - t)/cv::getTickFrequency();
	std::cout<<"FOR ABS: 1 "<<t<<"s"<<std::endl;
	short *pdxImg = dxImg_.ptr<short>();//dx方向图像的数据指针，没有指定矩阵行
	short *pdyImg = dyImg_.ptr<short>();//dy方向图像的数据指针，没有指定矩阵行
	short *pgImg  = gImg_.ptr<short>();//x,y方向梯度绝对值求和再阈值处理后的图像的数据指针，没有指定矩阵行
	unsigned char *pdirImg = dirImg_.ptr();//方向图像的数据指针，没有指定矩阵行
	
	//extract the anchors in the gradient image, store into a vector
	memset(pAnchorX_,  0, edgePixelArraySize*sizeof(unsigned int));//initialization，将X方向锚点清零
	memset(pAnchorY_,  0, edgePixelArraySize*sizeof(unsigned int));//将Y方向锚点清零
	unsigned int anchorsSize = 0;
	int indexInArray;
	unsigned char gValue1, gValue2, gValue3;
	
	cv::Mat AnchorImg( imageHeight,imageWidth,CV_8UC1 );//锚点图显示用，仅调试阶段使用
	u_char* pAnchorImg = AnchorImg.ptr<u_char>();
    	//scanIntervals为扫描刻度值，默认为2

    u_char* pAnchordata = new u_char [pixelNum];
    delete []pAnchordata;//这个数组指针主要是给标记锚点做一个总的记录，为的是判断当前的点旁边是否有锚点。
	//如果是X方向梯度则判断其左边是否已经是锚点，如果是Y方向梯度则判断其上边是否已经是锚点。如果是则当前
	//位置的锚点标记是多余的，否则可将其标记为锚点。

	int temp_w;//临时变量存列号
	int temp_h;//临时变量存行号
	/*
	for(unsigned int w=1; w<imageWidth-1; w=w+scanIntervals)
	{
		for(unsigned int h=1; h<imageHeight-1; h=h+scanIntervals)
		{
	*/
	for(unsigned int h=1; h<imageHeight-1; h=h+scanIncreseY  )
	{
		for(unsigned int w=1; w<imageWidth-1; w=w+scanIncreseX )
		{
			indexInArray = h*imageWidth+w;
			//gValue1 = pdirImg[indexInArray];
			if(pdirImg[indexInArray]==Horizontal)//Horizontal的宏定义值为255,即水平方向梯度变化大
			{	//if the direction of pixel is horizontal, then compare with up and down
				//gValue2 = pgImg[indexInArray];
                if(pgImg[indexInArray]>=pgImg[indexInArray-imageWidth]+anchorThreshold_
                        &&pgImg[indexInArray]>=pgImg[indexInArray+imageWidth]+anchorThreshold_)
                {			// (w,h) is accepted as an anchor
					//要是处于栅格行,则要判断左边是不是已经是锚点了
                    if( w != 0 && h%scanIntervals == 0 && pdirImg[indexInArray-1]==Horizontal && pAnchordata[ h*imageWidth+w-1 ] == 255 );
                    else
                    {
						pAnchorX_[anchorsSize] = w;
						pAnchorY_[anchorsSize] = h;
                        pAnchorImg[ h*imageWidth+w ] = 255;//显示用
                        pAnchordata[ h*imageWidth+w ] = 255;
						anchorsSize++;
                    }
                }
			}
			else
			{	// if(pdirImg[indexInArray]==Vertical){//it is vertical edge, should be compared with left and right
				//gValue2 = pgImg[indexInArray];
                if(pgImg[indexInArray]>=pgImg[indexInArray-1]+anchorThreshold_
                        &&pgImg[indexInArray]>=pgImg[indexInArray+1]+anchorThreshold_)
                {			// (w,h) is accepted as an anchor
                    if( h != 0 && w%scanIntervals == 0 && pdirImg[indexInArray-imageWidth]==Vertical && pAnchordata[ (h-1)*imageWidth+w ] == 255 );
                    else
                    {
						pAnchorX_[anchorsSize] = w;
						pAnchorY_[anchorsSize] = h;
						pAnchorImg[ h*imageWidth+w ] = 255;
                        pAnchordata[ h*imageWidth+w ] = 255;
						anchorsSize++;	
                    }
                }
			}
			//扫描栅格增量求解，需要5×5增量，直线最大漏检长度控制在7个像素值内
			temp_w = w;//存当前列号
			temp_h = h;//存当前行号
			if( (imageWidth - temp_w) > scanIntervals)
			{
				if( temp_h%scanIntervals == 0 )
					scanIncreseX = 1;//只要是处于栅格行上列号只都1
				else 
					if( temp_w%scanIntervals == 0 )
						scanIncreseX = scanIntervals;//不处于栅格行上，但是处于栅格列上，所以列号加的扫描间隙为栅格宽度
					else 
						scanIncreseX = (scanIntervals- temp_w%scanIntervals);//让其再次附着在栅格列上
			}
			else 
			{
				if( temp_h%scanIntervals == 0 )
				{
					scanIncreseX = 1;
				}
				else 
				{
					scanIncreseX = scanIntervals;//不处于栅格行上,所以列号加大于最后间隔，则会超出范围，故循环结束
				}
			}
        }
        scanIncreseY = 1 ;//In the Y direction,行号都加1
    }
	std::cout << "anchorsSize: " << anchorsSize << std::endl;
	if(anchorsSize>edgePixelArraySize){
		cout<<"anchor size is larger than its maximal size. anchorsSize="<<anchorsSize
		<<", maximal size = "<<edgePixelArraySize <<endl;
		//return -1;
	}
	gImg_.convertTo(Img_view, CV_8U);
#ifdef DEBUGEdgeDrawing
	cout<<"Anchor point detection, anchors.size="<<anchorsSize<<endl;
	cv::namedWindow( "gImg_",CV_WINDOW_NORMAL );
	cv::imshow( "gImg_",Img_view );
	imwrite("GradientImg.jpg",Img_view);//存储含检测到的梯度图像
	namedWindow( "锚点图",CV_WINDOW_NORMAL );
	imshow( "锚点图",AnchorImg );
	imwrite("AnchorImg.jpg",AnchorImg);//存储含检测到的锚点图像
#endif
	//link the anchors by smart routing
	edgeImage_.setTo(0);
	unsigned char *pEdgeImg = edgeImage_.data;
	memset(pFirstPartEdgeX_,  0, edgePixelArraySize*sizeof(unsigned int));//initialization
	memset(pFirstPartEdgeY_,  0, edgePixelArraySize*sizeof(unsigned int));
	memset(pSecondPartEdgeX_, 0, edgePixelArraySize*sizeof(unsigned int));
	memset(pSecondPartEdgeY_, 0, edgePixelArraySize*sizeof(unsigned int));
	memset(pFirstPartEdgeS_,  0, maxNumOfEdge*sizeof(unsigned int));
	memset(pSecondPartEdgeS_, 0, maxNumOfEdge*sizeof(unsigned int));
	unsigned int offsetPFirst=0, offsetPSecond=0;
	unsigned int offsetPS=0;
	unsigned int x, y;
	unsigned int lastX, lastY;
	unsigned char lastDirection;//up = 1, right = 2, down = 3, left = 4;
	unsigned char shouldGoDirection;//up = 1, right = 2, down = 3, left = 4;
	int edgeLenFirst, edgeLenSecond;
	for(unsigned int i=0; i<anchorsSize; i++){
		x = pAnchorX_[i];
		y = pAnchorY_[i];
		indexInArray = y*imageWidth+x;
		if(pEdgeImg[indexInArray]){//if anchor i is already been an edge pixel.
			continue;
		}
		/*The walk stops under 3 conditions:
		 * 1. We move out of the edge areas, i.e., the thresholded gradient value
		 *    of the current pixel is 0.
		 * 2. The current direction of the edge changes, i.e., from horizontal
		 *    to vertical or vice versa.?? (This is turned out not correct. From the online edge draw demo
		 *    we can figure out that authors don't implement this rule either because their extracted edge
		 *    chain could be a circle which means pixel directions would definitely be different
		 *    in somewhere on the chain.)
		 * 3. We encounter a previously detected edge pixel. */
		pFirstPartEdgeS_[offsetPS] = offsetPFirst;
		if(pdirImg[indexInArray]==Horizontal){//if the direction of this pixel is horizontal, then go left and right.
			//fist go right, pixel direction may be different during linking.
			lastDirection = RightDir;
			while(pgImg[indexInArray]>0&&!pEdgeImg[indexInArray]){//阈值化图该像素值大于零且该像素点不是边缘
				pEdgeImg[indexInArray] = 1;        // Mark this pixel as an edge pixel，则将该点视作边缘
				pFirstPartEdgeX_[offsetPFirst] = x;//记录该点横坐标
				pFirstPartEdgeY_[offsetPFirst++] = y;//记录该点纵坐标
				shouldGoDirection = 0;//unknown
				if(pdirImg[indexInArray]==Horizontal)//如果方向为水平方向
				{//should go left or right
					if(lastDirection == UpDir ||lastDirection == DownDir){//出现锯齿状，change the pixel direction now
						if(x>lastX)//lastX好像没有初始化值
						{//should go right，出现锯齿状，但是主体还是水平，所以往水平方向走
							shouldGoDirection = RightDir;
						}else{//should go left
							shouldGoDirection = LeftDir;
						}
					}
					lastX = x;//记录坐标
					lastY = y;
					if(lastDirection == RightDir||shouldGoDirection == RightDir){//go right
						if(x==imageWidth-1||y==0||y==imageHeight-1){//reach the image border，到边界了
							break;//跳出最近封闭的循环，不用判断
						}
						// Look at 3 neighbors to the right and pick the one with the max. gradient value
						gValue1 = pgImg[indexInArray-imageWidth+1];//右上像素
						gValue2 = pgImg[indexInArray+1];//右像素
						gValue3 = pgImg[indexInArray+imageWidth+1];//右下像素
						if(gValue1>=gValue2&&gValue1>=gValue3){//up-right
							x = x+1;
							y = y-1;
						}else if(gValue3>=gValue2&&gValue3>=gValue1){//down-right
							x = x+1;
							y = y+1;
						}else{//straight-right，直接往右
							x = x+1;
						}
						lastDirection = RightDir;//最近的方向设置为往右
					} 
					else if(lastDirection == LeftDir || shouldGoDirection==LeftDir){//go left
						if(x==0||y==0||y==imageHeight-1){//reach the image border
							break;
						}
						// Look at 3 neighbors to the left and pick the one with the max. gradient value
						gValue1 = pgImg[indexInArray-imageWidth-1];
						gValue2 = pgImg[indexInArray-1];
						gValue3 = pgImg[indexInArray+imageWidth-1];
						if(gValue1>=gValue2&&gValue1>=gValue3){//up-left
							x = x-1;
							y = y-1;
						}else if(gValue3>=gValue2&&gValue3>=gValue1){//down-left
							x = x-1;
							y = y+1;
						}else{//straight-left
							x = x-1;
						}
						lastDirection = LeftDir;
					}
				}
				else{//should go up or down.梯度判断边缘为垂直方向
					if(lastDirection == RightDir || lastDirection == LeftDir){//出现锯齿状，change the pixel direction now
						if(y>lastY){//should go down，出现锯齿状，但是主体还是垂直，所以往垂直方向走
							shouldGoDirection = DownDir;
						}else{//should go up
							shouldGoDirection = UpDir;
						}
					}
					lastX = x;//记录坐标
					lastY = y;
					if(lastDirection==DownDir || shouldGoDirection == DownDir){//go down
						if(x==0||x==imageWidth-1||y==imageHeight-1){//reach the image border，到边界了
							break;
						}
						// Look at 3 neighbors to the down and pick the one with the max. gradient value
						gValue1 = pgImg[indexInArray+imageWidth+1];//右下像素
						gValue2 = pgImg[indexInArray+imageWidth];//下边像素
						gValue3 = pgImg[indexInArray+imageWidth-1];//左下像素
						if(gValue1>=gValue2&&gValue1>=gValue3){//down-right
							x = x+1;
							y = y+1;
						}else if(gValue3>=gValue2&&gValue3>=gValue1){//down-left
							x = x-1;
							y = y+1;
						}else{//straight-down
							y = y+1;
						}
						lastDirection = DownDir;//记录最近方向为往下
					}else if(lastDirection==UpDir || shouldGoDirection == UpDir){//go up，往上
						if(x==0||x==imageWidth-1||y==0){//reach the image border
							break;//直接退出最近循环体，无须再判断
						}
						// Look at 3 neighbors to the up and pick the one with the max. gradient value
						gValue1 = pgImg[indexInArray-imageWidth+1];//右上像素
						gValue2 = pgImg[indexInArray-imageWidth];//上面像素
						gValue3 = pgImg[indexInArray-imageWidth-1];//左上像素
						if(gValue1>=gValue2&&gValue1>=gValue3){//up-right
							x = x+1;
							y = y-1;
						}else if(gValue3>=gValue2&&gValue3>=gValue1){//up-left
							x = x-1;
							y = y-1;
						}else{//straight-up
							y = y-1;
						}
						lastDirection = UpDir;//记录最近方向为往上
					}
				}
				indexInArray = y*imageWidth+x;//检索序号更替
			}//end while go right
			//then go left, pixel direction may be different during linking.
			x = pAnchorX_[i];
			y = pAnchorY_[i];
			indexInArray = y*imageWidth+x;
			pEdgeImg[indexInArray] = 0;//mark the anchor point be a non-edge pixel and
			lastDirection = LeftDir;
			pSecondPartEdgeS_[offsetPS] = offsetPSecond;
			while(pgImg[indexInArray]>0&&!pEdgeImg[indexInArray]){
				pEdgeImg[indexInArray] = 1;        // Mark this pixel as an edge pixel
				pSecondPartEdgeX_[offsetPSecond] = x;
				pSecondPartEdgeY_[offsetPSecond++] = y;
				shouldGoDirection = 0;//unknown
				if(pdirImg[indexInArray]==Horizontal){//should go left or right
					if(lastDirection == UpDir ||lastDirection == DownDir){//change the pixel direction now
						if(x>lastX){//should go right
							shouldGoDirection = RightDir;
						}else{//should go left
							shouldGoDirection = LeftDir;
						}
					}
					lastX = x;
					lastY = y;
					if(lastDirection == RightDir||shouldGoDirection == RightDir){//go right
						if(x==imageWidth-1||y==0||y==imageHeight-1){//reach the image border
							break;
						}
						// Look at 3 neighbors to the right and pick the one with the max. gradient value
						gValue1 = pgImg[indexInArray-imageWidth+1];
						gValue2 = pgImg[indexInArray+1];
						gValue3 = pgImg[indexInArray+imageWidth+1];
						if(gValue1>=gValue2&&gValue1>=gValue3){//up-right
							x = x+1;
							y = y-1;
						}else if(gValue3>=gValue2&&gValue3>=gValue1){//down-right
							x = x+1;
							y = y+1;
						}else{//straight-right
							x = x+1;
						}
						lastDirection = RightDir;
					}else	if(lastDirection == LeftDir || shouldGoDirection==LeftDir){//go left
						if(x==0||y==0||y==imageHeight-1){//reach the image border
							break;
						}
						// Look at 3 neighbors to the left and pick the one with the max. gradient value
						gValue1 = pgImg[indexInArray-imageWidth-1];
						gValue2 = pgImg[indexInArray-1];
						gValue3 = pgImg[indexInArray+imageWidth-1];
						if(gValue1>=gValue2&&gValue1>=gValue3){//up-left
							x = x-1;
							y = y-1;
						}else if(gValue3>=gValue2&&gValue3>=gValue1){//down-left
							x = x-1;
							y = y+1;
						}else{//straight-left
							x = x-1;
						}
						lastDirection = LeftDir;
					}
				}else{//should go up or down.
					if(lastDirection == RightDir || lastDirection == LeftDir){//change the pixel direction now
						if(y>lastY){//should go down
							shouldGoDirection = DownDir;
						}else{//should go up
							shouldGoDirection = UpDir;
						}
					}
					lastX = x;
					lastY = y;
					if(lastDirection==DownDir || shouldGoDirection == DownDir){//go down
						if(x==0||x==imageWidth-1||y==imageHeight-1){//reach the image border
							break;
						}
						// Look at 3 neighbors to the down and pick the one with the max. gradient value
						gValue1 = pgImg[indexInArray+imageWidth+1];
						gValue2 = pgImg[indexInArray+imageWidth];
						gValue3 = pgImg[indexInArray+imageWidth-1];
						if(gValue1>=gValue2&&gValue1>=gValue3){//down-right
							x = x+1;
							y = y+1;
						}else if(gValue3>=gValue2&&gValue3>=gValue1){//down-left
							x = x-1;
							y = y+1;
						}else{//straight-down
							y = y+1;
						}
						lastDirection = DownDir;
					}else	if(lastDirection==UpDir || shouldGoDirection == UpDir){//go up
						if(x==0||x==imageWidth-1||y==0){//reach the image border
							break;
						}
						// Look at 3 neighbors to the up and pick the one with the max. gradient value
						gValue1 = pgImg[indexInArray- imageWidth+1];
						gValue2 = pgImg[indexInArray- imageWidth];
						gValue3 = pgImg[indexInArray- imageWidth-1];
						if(gValue1>=gValue2&&gValue1>=gValue3){//up-right
							x = x+1;
							y = y-1;
						}else if(gValue3>=gValue2&&gValue3>=gValue1){//up-left
							x = x-1;
							y = y-1;
						}else{//straight-up
							y = y-1;
						}
						lastDirection = UpDir;
					}
				}
				indexInArray = y*imageWidth+x;
			}//end while go left
			//end anchor is Horizontal
		}
		else{//the direction of this pixel is vertical, go up and down，初始方向是垂直的
			//fist go down, pixel direction may be different during linking.
			lastDirection = DownDir;
			while(pgImg[indexInArray]>0&&!pEdgeImg[indexInArray]){
				pEdgeImg[indexInArray] = 1;        // Mark this pixel as an edge pixel
				pFirstPartEdgeX_[offsetPFirst] = x;
				pFirstPartEdgeY_[offsetPFirst++] = y;
				shouldGoDirection = 0;//unknown
				if(pdirImg[indexInArray]==Horizontal){//should go left or right
					if(lastDirection == UpDir ||lastDirection == DownDir){//change the pixel direction now
						if(x>lastX){//should go right
							shouldGoDirection = RightDir;
						}else{//should go left
							shouldGoDirection = LeftDir;
						}
					}
					lastX = x;
					lastY = y;
					if(lastDirection == RightDir||shouldGoDirection == RightDir){//go right
						if(x==imageWidth-1||y==0||y==imageHeight-1){//reach the image border
							break;
						}
						// Look at 3 neighbors to the right and pick the one with the max. gradient value
						gValue1 = pgImg[indexInArray- imageWidth+1];
						gValue2 = pgImg[indexInArray+1];
						gValue3 = pgImg[indexInArray+ imageWidth+1];
						if(gValue1>=gValue2&&gValue1>=gValue3){//up-right
							x = x+1;
							y = y-1;
						}else if(gValue3>=gValue2&&gValue3>=gValue1){//down-right
							x = x+1;
							y = y+1;
						}else{//straight-right
							x = x+1;
						}
						lastDirection = RightDir;
					}else	if(lastDirection == LeftDir || shouldGoDirection==LeftDir){//go left
						if(x==0||y==0||y==imageHeight-1){//reach the image border
							break;
						}
						// Look at 3 neighbors to the left and pick the one with the max. gradient value
						gValue1 = pgImg[indexInArray- imageWidth-1];
						gValue2 = pgImg[indexInArray-1];
						gValue3 = pgImg[indexInArray+ imageWidth-1];
						if(gValue1>=gValue2&&gValue1>=gValue3){//up-left
							x = x-1;
							y = y-1;
						}else if(gValue3>=gValue2&&gValue3>=gValue1){//down-left
							x = x-1;
							y = y+1;
						}else{//straight-left
							x = x-1;
						}
						lastDirection = LeftDir;
					}
				}else{//should go up or down.
					if(lastDirection == RightDir || lastDirection == LeftDir){//change the pixel direction now
						if(y>lastY){//should go down
							shouldGoDirection = DownDir;
						}else{//should go up
							shouldGoDirection = UpDir;
						}
					}
					lastX = x;
					lastY = y;
					if(lastDirection==DownDir || shouldGoDirection == DownDir){//go down
						if(x==0||x==imageWidth-1||y==imageHeight-1){//reach the image border
							break;
						}
						// Look at 3 neighbors to the down and pick the one with the max. gradient value
						gValue1 = pgImg[indexInArray+ imageWidth+1];
						gValue2 = pgImg[indexInArray+ imageWidth];
						gValue3 = pgImg[indexInArray+ imageWidth-1];
						if(gValue1>=gValue2&&gValue1>=gValue3){//down-right
							x = x+1;
							y = y+1;
						}else if(gValue3>=gValue2&&gValue3>=gValue1){//down-left
							x = x-1;
							y = y+1;
						}else{//straight-down
							y = y+1;
						}
						lastDirection = DownDir;
					}else	if(lastDirection==UpDir || shouldGoDirection == UpDir){//go up
						if(x==0||x==imageWidth-1||y==0){//reach the image border
							break;
						}
						// Look at 3 neighbors to the up and pick the one with the max. gradient value
						gValue1 = pgImg[indexInArray- imageWidth+1];
						gValue2 = pgImg[indexInArray- imageWidth];
						gValue3 = pgImg[indexInArray- imageWidth-1];
						if(gValue1>=gValue2&&gValue1>=gValue3){//up-right
							x = x+1;
							y = y-1;
						}else if(gValue3>=gValue2&&gValue3>=gValue1){//up-left
							x = x-1;
							y = y-1;
						}else{//straight-up
							y = y-1;
						}
						lastDirection = UpDir;
					}
				}
				indexInArray = y*imageWidth+x;
			}//end while go down
			//then go up, pixel direction may be different during linking.
			lastDirection = UpDir;
			x = pAnchorX_[i];
			y = pAnchorY_[i];
			indexInArray = y*imageWidth+x;
			pEdgeImg[indexInArray] = 0;//mark the anchor point be a non-edge pixel and
			pSecondPartEdgeS_[offsetPS] = offsetPSecond;
			while(pgImg[indexInArray]>0&&!pEdgeImg[indexInArray]){
				pEdgeImg[indexInArray] = 1;        // Mark this pixel as an edge pixel
				pSecondPartEdgeX_[offsetPSecond] = x;
				pSecondPartEdgeY_[offsetPSecond++] = y;
				shouldGoDirection = 0;//unknown
				if(pdirImg[indexInArray]==Horizontal){//should go left or right
					if(lastDirection == UpDir ||lastDirection == DownDir){//change the pixel direction now
						if(x>lastX){//should go right
							shouldGoDirection = RightDir;
						}else{//should go left
							shouldGoDirection = LeftDir;
						}
					}
					lastX = x;
					lastY = y;
					if(lastDirection == RightDir||shouldGoDirection == RightDir){//go right
						if(x==imageWidth-1||y==0||y==imageHeight-1){//reach the image border
							break;
						}
						// Look at 3 neighbors to the right and pick the one with the max. gradient value
						gValue1 = pgImg[indexInArray- imageWidth+1];
						gValue2 = pgImg[indexInArray+1];
						gValue3 = pgImg[indexInArray+ imageWidth +1];
						if(gValue1>=gValue2&&gValue1>=gValue3){//up-right
							x = x+1;
							y = y-1;
						}else if(gValue3>=gValue2&&gValue3>=gValue1){//down-right
							x = x+1;
							y = y+1;
						}else{//straight-right
							x = x+1;
						}
						lastDirection = RightDir;
					}else	if(lastDirection == LeftDir || shouldGoDirection==LeftDir){//go left
						if(x==0||y==0||y==imageHeight-1){//reach the image border
							break;
						}
						// Look at 3 neighbors to the left and pick the one with the max. gradient value
						gValue1 = pgImg[indexInArray- imageWidth -1];
						gValue2 = pgImg[indexInArray-1];
						gValue3 = pgImg[indexInArray+ imageWidth -1];
						if(gValue1>=gValue2&&gValue1>=gValue3){//up-left
							x = x-1;
							y = y-1;
						}else if(gValue3>=gValue2&&gValue3>=gValue1){//down-left
							x = x-1;
							y = y+1;
						}else{//straight-left
							x = x-1;
						}
						lastDirection = LeftDir;
					}
				}else{//should go up or down.
					if(lastDirection == RightDir || lastDirection == LeftDir){//change the pixel direction now
						if(y>lastY){//should go down
							shouldGoDirection = DownDir;
						}else{//should go up
							shouldGoDirection = UpDir;
						}
					}
					lastX = x;
					lastY = y;
					if(lastDirection==DownDir || shouldGoDirection == DownDir){//go down
						if(x==0||x==imageWidth-1||y==imageHeight-1){//reach the image border
							break;
						}
						// Look at 3 neighbors to the down and pick the one with the max. gradient value
						gValue1 = pgImg[indexInArray+ imageWidth +1];
						gValue2 = pgImg[indexInArray+ imageWidth];
						gValue3 = pgImg[indexInArray+ imageWidth -1];
						if(gValue1>=gValue2&&gValue1>=gValue3){//down-right
							x = x+1;
							y = y+1;
						}else if(gValue3>=gValue2&&gValue3>=gValue1){//down-left
							x = x-1;
							y = y+1;
						}else{//straight-down
							y = y+1;
						}
						lastDirection = DownDir;
					}else	if(lastDirection==UpDir || shouldGoDirection == UpDir){//go up
						if(x==0||x==imageWidth-1||y==0){//reach the image border
							break;
						}
						// Look at 3 neighbors to the up and pick the one with the max. gradient value
						gValue1 = pgImg[indexInArray- imageWidth +1];
						gValue2 = pgImg[indexInArray- imageWidth];
						gValue3 = pgImg[indexInArray- imageWidth -1];
						if(gValue1>=gValue2&&gValue1>=gValue3){//up-right
							x = x+1;
							y = y-1;
						}else if(gValue3>=gValue2&&gValue3>=gValue1){//up-left
							x = x-1;
							y = y-1;
						}else{//straight-up
							y = y-1;
						}
						lastDirection = UpDir;
					}
				}
				indexInArray = y*imageWidth+x;
			}//end while go up
		}//end anchor is Vertical
		//only keep the edge chains whose length is larger than the minLineLen_;
		edgeLenFirst = offsetPFirst - pFirstPartEdgeS_[offsetPS];
		edgeLenSecond = offsetPSecond - pSecondPartEdgeS_[offsetPS];
		if(edgeLenFirst+edgeLenSecond<minLineLen_+1){//short edge, drop it
			offsetPFirst = pFirstPartEdgeS_[offsetPS];
			offsetPSecond = pSecondPartEdgeS_[offsetPS];
		}else{
			offsetPS++;
		}
	}
	
#ifdef DEBUGEdgeDrawing
	//显示用，后续删除
	cv::Mat edge_view_ = edgeImage_.clone();
	for( int i = 0; i < imageHeight; i ++ )
	{
	  uchar* pedge_view = edge_view_.ptr<uchar>(i);
	  for( int j = 0; j < imageWidth; j ++ )
	  {
	    if( pedge_view[j] == 1 )
	      pedge_view[j] = 255;
	      else;
	  }
	}
	namedWindow( "edge_v1",CV_WINDOW_NORMAL );
	cv::imshow( "edge_v1",edge_view_ );	
	imwrite("SaveImage.png",edge_view_);//存储含检测到的边缘图像
#endif
	
	for( int i = 0; i < src.rows; i++ )
	{
	  short int* _dxImg = dxImg_.ptr<short>(i);
	  short int* _dyImg = dyImg_.ptr<short>(i);
	 
	  for( int j = 0; j < src.cols; j++ )
		if( abs(_dyImg[j]) > 600 )
		{
		  cout << " 警告，梯度值绝对值大于600,有可能超出范围 " << endl;
		  cout << "列: " << j << "   行: " << i << " 梯度值为: " << _dyImg[j] << endl;
		}
	}
	
	//调试显示用，最终需要修改
	dxImg_.convertTo(dx_view, CV_8U);//像素值无符号
	namedWindow( "dx_view:CV_8U",CV_WINDOW_AUTOSIZE );
	imshow( "dx_view:CV_8U",dx_view );//像素有三原色数字显示
	dyImg_.convertTo(dy_view, CV_8U);//像素值无符号
	namedWindow( "dy_view:CV_8U",CV_WINDOW_AUTOSIZE );
	imshow( "dy_view:CV_8U",dy_view );//像素有三原色数字显示
	
	//检查梯度图并且显示	
	addWeighted(dx_view,0.5,dy_view,0.5,0,dxy_grad);
	dxy_grad.convertTo(Img_view, CV_8U);//像素值无符号
	namedWindow( "总梯度图",WINDOW_AUTOSIZE );
	imshow( "总梯度图",Img_view );
	
	//高低阈值调整
	if(L2gradient)
	{
	  low_thresh = std::min(32767.0,low_thresh);
	  high_thresh = std::min(32767.0,high_thresh);
	  if(low_thresh > 0) low_thresh *= low_thresh;
	  if(high_thresh > 0) high_thresh *= high_thresh;
	}
	else
	{
	  low_thresh = std::min(255.0,low_thresh);
	  high_thresh = std::min(255.0,high_thresh);
	}
	int low = cvFloor(low_thresh);
	int high = cvFloor(high_thresh);
	cout << "low:" << low << "   " << "high:" << high << endl;
	
	for( int i = 0; i < src.rows; i++ )
	{
	  if( i < src.rows )
	  {
	    short* pdxImg = dxImg_.ptr<short>(i);
	    short* pdyImg = dyImg_.ptr<short>(i);
	    u_short* pdxABS_max = dxABS_max.ptr<u_short>(i);
	    u_short* pdyABS_max = dyABS_max.ptr<u_short>(i);
	    u_short* pdxyNorm = dxyNorm.ptr<u_short>(i);
	    u_short* pSumDxDy = SumDxDy.ptr<u_short>(i);
	    u_short* pgImg = gImg_.ptr<u_short>(i);
	    
	    if(L2gradient)//用二范数求
	    {
	      for( int j = 0; j < src.cols*cn; j++ )
	      {
		pdxyNorm[j] = ushort(std::abs(float(pdxImg[j]/2.5))*std::abs(float(pdxImg[j]/2.5)) + std::abs(float(pdyImg[j]/2.5))*std::abs(float(pdyImg[j]/2.5)));//二范数求和
		if( int(pdxImg[j]) > 600 )
		  cout << " 警告，梯度值大于600,有可能超出范围 " << endl;
	      }
	    }
	    else//用一范数求
	    {
	      for( int j = 0; j < src.cols*cn; j++ )
	      {
		pdxyNorm[j] = std::abs(int(pdxImg[j])) + std::abs(int(pdyImg[j]));//一范数求和
	      }
	    }
	    
	    if(cn > 1)//多通道，一般为三通道BGR
	    {
	      for( int j = 0,jn = 0; j < src.cols; ++j,jn += cn )
	      {
		int maxIdx = jn;
		for( int k = 1; k < cn; ++k )
		  if( pdxyNorm[jn+k] > pdxyNorm[maxIdx] ) { maxIdx = jn + k; }
		  if( pdxyNorm[maxIdx] > high ) 
		  { 
		    pdxABS_max[j] =  std::abs(int (pdxImg[maxIdx]));//最大的那个灰度值取绝对值
		    pdyABS_max[j] =  std::abs(int (pdyImg[maxIdx]));//最大的那个灰度值取绝对值
		    pSumDxDy[j] = pdxyNorm[maxIdx];//最大的梯度值求和
		  }
	      }
	    }   
	  }
	}
    /*
	namedWindow( "dxyNorm",WINDOW_AUTOSIZE );
	imshow( "dxyNorm",dxyNorm );
	(Img_view=SumDxDy/255).convertTo(Img_view, CV_8U);//像素值无符号
	namedWindow( "范数求和后梯度",WINDOW_AUTOSIZE );
	imshow( "范数求和后梯度",Img_view );
		
	//if(L2gradient)
	//cv::threshold(SumDxDy,gImg_, gradienThreshold*gradienThreshold, 65535, cv::THRESH_TOZERO);//阈值处理后gImg_图
	//else
	//cv::threshold(SumDxDy,gImg_, gradienThreshold+1, 255, cv::THRESH_TOZERO);//阈值处理后gImg_图
	gImg_ = gImg_/4;
	gImg_.convertTo(Img_view, CV_8U,1,0);//像素值无符号
	namedWindow( "阈值处理后最大梯度",WINDOW_AUTOSIZE );
	imshow( "阈值处理后最大梯度",Img_view );

	//显示求得的梯度图
	cv::namedWindow( "gImg_",WINDOW_AUTOSIZE );
	cv::imshow( "gImg_",gImg_ );
	gImgWO_ = SumDxDy/4;
	cv::compare(dxABS_max, dyABS_max, dirImg_, cv::CMP_LT);//水平方向梯度变化则值为255,垂直方向大则为0,dst(i)=src1(i) cmpop src2(i)
	namedWindow( "方向图",WINDOW_AUTOSIZE );
	imshow( "方向图",dirImg_ );
	short *pgImg  = gImg_.ptr<short>();//阈值处理后的图像的数据指针，没有指定矩阵行
	unsigned char *pdirImg = dirImg_.ptr();//方向图像的数据指针，没有指定矩阵行
	//extract the anchors in the gradient image, store into a vector
	memset(pAnchorX_,  0, edgePixelArraySize*sizeof(unsigned int));//initialization，将X方向锚点清零
	memset(pAnchorY_,  0, edgePixelArraySize*sizeof(unsigned int));//将Y方向锚点清零
	unsigned int anchorsSize = 0;
	int indexInArray;
	unsigned int gValue1, gValue2, gValue3;
	
	cv::Mat AnchorImg( src.rows,src.cols,CV_8UC1 );//锚点图显示用，仅调试阶段使用
	u_char* pAnchorImg = AnchorImg.ptr<u_char>();
	//scanIntervals为扫描刻度值，默认为2
	int temp_w;//临时变量存列号
	int temp_h;//临时变量存行号
	for(unsigned int h=1; h<imageHeight-1; h=h+scanIncreseY)
	{
		for(unsigned int w=1; w<imageWidth-1; w=w+scanIncreseX)
		{
			indexInArray = h*imageWidth+w;
			//gValue1 = pdirImg[indexInArray];
			if(pdirImg[indexInArray]==Horizontal)//Horizontal的宏定义值为255,即水平方向梯度变化大
			{	//if the direction of pixel is horizontal, then compare with up and down
				//gValue2 = pgImg[indexInArray];
				if(pgImg[indexInArray]>=pgImg[indexInArray-imageWidth]+anchorThreshold_
						&&pgImg[indexInArray]>=pgImg[indexInArray+imageWidth]+anchorThreshold_)
				{			// (w,h) is accepted as an anchor
							pAnchorX_[anchorsSize] = w;
							pAnchorY_[anchorsSize] = h;
							pAnchorImg[ h*imageWidth+w ] = 255;
							anchorsSize++;
				}
			}
			else
			{	// if(pdirImg[indexInArray]==Vertical){//it is vertical edge, should be compared with left and right
				//gValue2 = pgImg[indexInArray];
				if(pgImg[indexInArray]>=pgImg[indexInArray-1]+anchorThreshold_
						&&pgImg[indexInArray]>=pgImg[indexInArray+1]+anchorThreshold_)
				{			// (w,h) is accepted as an anchor
							pAnchorX_[anchorsSize] = w;
							pAnchorY_[anchorsSize] = h;
							pAnchorImg[ h*imageWidth+w ] = 255;
							anchorsSize++;	
				}
			}
			//扫描栅格增量求解，需要5×5增量，直线最大漏检长度控制在7个像素值内
			temp_w = w;//存当前列号
			temp_h = h;//存当前行号
			if( (imageWidth - temp_w) > scanIntervals )
			{
			  if( temp_h%scanIntervals == 0 )
			    scanIncreseX = 1;//只要是处于栅格行上列号只都1
			  else 
			    if( temp_w%scanIntervals == 0 )
			      scanIncreseX = scanIntervals;//不处于栅格行上，但是处于栅格列上，所以列号加的扫描间隙为栅格宽度
			    else 
			      scanIncreseX = (scanIntervals- temp_w%scanIntervals);//让其再次附着在栅格列上
			}
			else 
			{
			  if( temp_h%scanIntervals == 0 )
			  {
			    scanIncreseX = 1;
			  }
			  else 
			  {
			    scanIncreseX = scanIntervals;//不处于栅格行上,所以列号加大于最后间隔，则会超出范围，故循环结束
			  }
			}
		}
		if( (imageHeight - temp_h) > scanIntervals )
		{
		  if( temp_w%scanIntervals ==0 )//在栅格列上
		  {
		    scanIncreseY = 1 ;//只要是处于栅格列上，行号都加1
		    //前一行处于栅格行上则行号加的扫描间隙为栅格高度
		  }
		  else
		    if( temp_h%scanIntervals == 0 )//处于栅格行上，但是不在栅格列上
		    {
		      scanIncreseY = scanIntervals ;//处于栅格行上，但不处于栅格列上，所以行号加的扫描间隙为栅格高度
		      //前一行处于栅格行上则行号加的扫描间隙为栅格高度
		    }
		    else//既不在栅格行上，又不在栅格列上
		    {
		      scanIncreseY =  - temp_h%scanIntervals);//让其再次附着在栅格行上
		    }
		}
		else 
		{
		  if( temp_w%scanIntervals == 0 )//在栅格列上，但是是图像最后几行
		  {
		    scanIncreseY = 1;
		  }
		  else//不在栅格列上，但是是图像最后几行 
		  {
		    scanIncreseY = scanIntervals;//不处于栅格列上,所以行号加大于最后间隔，则会超出范围，故循环结束
		  }
		}
	}
	if(anchorsSize>edgePixelArraySize){
		cout<<"anchor size is larger than its maximal size. anchorsSize="<<anchorsSize
		<<", maximal size = "<<edgePixelArraySize <<endl;
		return -1;
	}
#ifdef DEBUGEdgeDrawing
	cout<<"Anchor point detection, anchors.size="<<anchorsSize<<endl;
#endif	
	namedWindow( "锚点图",CV_WINDOW_NORMAL );
	imshow( "锚点图",AnchorImg );
	//link the anchors by smart routing
	edgeImage_.setTo(0);
	unsigned char *pEdgeImg = edgeImage_.data;
	memset(pFirstPartEdgeX_,  0, edgePixelArraySize*sizeof(unsigned int));//initialization
	memset(pFirstPartEdgeY_,  0, edgePixelArraySize*sizeof(unsigned int));
	memset(pSecondPartEdgeX_, 0, edgePixelArraySize*sizeof(unsigned int));
	memset(pSecondPartEdgeY_, 0, edgePixelArraySize*sizeof(unsigned int));
	memset(pFirstPartEdgeS_,  0, maxNumOfEdge*sizeof(unsigned int));
	memset(pSecondPartEdgeS_, 0, maxNumOfEdge*sizeof(unsigned int));
	unsigned int offsetPFirst=0, offsetPSecond=0;
	unsigned int offsetPS=0;
	unsigned int x, y;
	unsigned int lastX, lastY;
	unsigned char lastDirection;//up = 1, right = 2, down = 3, left = 4;
	unsigned char shouldGoDirection;//up = 1, right = 2, down = 3, left = 4;
	int edgeLenFirst, edgeLenSecond;
	for(unsigned int i=0; i<anchorsSize; i++){
        x = pAnchorX_[i];
		y = pAnchorY_[i];
		indexInArray = y*imageWidth+x;
        if(pEdgeImg[indexInArray]){//if anchor i is already been an edge pixel.
			continue;
		}
*/
}
