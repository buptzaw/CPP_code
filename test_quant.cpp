#include<iostream>
#include<cmath>
#include<math.h>
#include"tools.h"
#include<opencv2/opencv.hpp>
#include<sstream>
using namespace std;
using namespace cv;
Mat distortion_correction_old(Mat img)
{
	Mat out;
	//int img_rows = img.rows;
	//int img_cols = img.cols;

	Mat camera_matrixa = (cv::Mat_<float>(3, 3) << 1565.3, 0, 722.88, 0, 1562.659, 541.263, 0, 0, 1);//相机参数
	//[fx,0,cx,0,fy,cy,0,0,1]
	//Mat distortion_coefficientsa = (cv::Mat_<float >(1, 4) << -1.467, 1.036, 0, 0);
	Mat distortion_coefficientsa = (cv::Mat_<float >(1, 4) << -0.6, 0.34, 0, 0);
	//k1, k2, p1, p2
	Mat mapx, mapy;
	cv::initUndistortRectifyMap(camera_matrixa, distortion_coefficientsa, cv::Mat(), camera_matrixa, cv::Size(img.cols, img.rows), CV_32FC1, mapx, mapy);

	float k1 = distortion_coefficientsa.at<float>(0, 0);
	float k2 = distortion_coefficientsa.at<float>(0, 1);

	float fx = camera_matrixa.at<float>(0, 0);
	float fy = camera_matrixa.at<float>(1, 1);//提取矩阵元素
	float cx = camera_matrixa.at<float>(0, 2);
	float cy = camera_matrixa.at<float>(1, 2);

	float x = cx / fx;
	float y = cy / fy;
	if (k1 < 0)
	{
		float r;
		if (x < y)
			r = x;
		else
			r = y;
		float div = 1 / (1 + k1 * r*r + k2 * r * r * r * r);
		mapx.convertTo(mapx, CV_32F, div, cx*(1 - div));
		mapy.convertTo(mapy, CV_32F, div, cy*(1 - div));
	}
	else
	{
		float r = sqrt(x*x + y*y);
		float div = 1 / (1 + k1 * r*r + k2 * r * r * r * r);
		mapx.convertTo(mapx, CV_32F, div, cx*(1 - div));
		mapy.convertTo(mapy, CV_32F, div, cy*(1 - div));
	}

	cv::remap(img, out, mapx, mapy, cv::INTER_LINEAR);
	return out;
}


void distortion_correction_quant(Mat img, Mat& mapx_new, Mat& mapy_new)
{
	//Mat out;


	double k1 = -0.6, k2 = 0.34, p1 = 0.0, p2 = 0.0;              //k1,k2 影响很大
	double fx = 1565.3, fy = 1562.659, cx = 722.88, cy = 541.263;

	cv::Mat image = img;
	int rows = image.rows, cols = image.cols;
	Mat mapx(rows, cols, CV_32F, 0.00);
	Mat mapy(rows, cols, CV_32F, 0.00);
	float x = cx / fx;
	float y = cy / fy;
	if (k1<0)
	{
		for (int v = 0; v < rows; v++) {
			for (int u = 0; u < cols; u++) {
				float x = (u - cx) / fx, y = (v - cy) / fy;
				float r = sqrt(x * x + y * y);
				float x_distorted = x * (1 + k1 * r * r + k2 * r * r * r * r) + 2 * p1 * x * y + p2 * (r * r + 2 * x * x);
				float y_distorted = y * (1 + k1 * r * r + k2 * r * r * r * r) + p1 * (r * r + 2 * y * y) + 2 * p2 * x * y;
				float u_distorted = fx * x_distorted + cx;
				float v_distorted = fy * y_distorted + cy;
				mapx.at<float>(v, u) = u_distorted;//x映射到u
				mapy.at<float>(v, u) = v_distorted;//y映射到v

				float rx = cx / fx;
				float ry = cy / fy;
				float div;
				if (rx < ry)
					div = 1 / (1 + k1 * rx*rx + k2 * rx * rx * rx * rx);
				else
					div = 1 / (1 + k1 * ry*ry + k2 * ry * ry * ry * ry);

				mapx.at<float>(v, u) = (u_distorted - cx)*div + cx;
				mapy.at<float>(v, u) = (v_distorted - cy)*div + cy;
			}
		}
	}
	else
	{
		for (int v = 0; v < rows; v++) {
			for (int u = 0; u < cols; u++) {
				float x = (u - cx) / fx, y = (v - cy) / fy;
				float r = sqrt(x * x + y * y);
				float x_distorted = x * (1 + k1 * r * r + k2 * r * r * r * r) + 2 * p1 * x * y + p2 * (r * r + 2 * x * x);
				float y_distorted = y * (1 + k1 * r * r + k2 * r * r * r * r) + p1 * (r * r + 2 * y * y) + 2 * p2 * x * y;
				float u_distorted = fx * x_distorted + cx;
				float v_distorted = fy * y_distorted + cy;
				mapx.at<float>(v, u) = u_distorted;
				mapy.at<float>(v, u) = v_distorted;

				float rx = cx / fx;
				float ry = cy / fy;
				float rr = sqrt(rx*rx + ry * ry);
				float div = 1 / (1 + k1 * rr*rr + k2 * rr * rr * rr * rr);;

				mapx.at<float>(v, u) = (u_distorted - cx)*div + cx;
				mapy.at<float>(v, u) = (v_distorted - cy)*div + cy;
			}
		}
	}

	//求出矩阵mapx所包含的数据范围即(min,max)
	float max = mapx.at<float>(0, 0);
	float min = mapx.at<float>(0, 0);
	for (int i = 0; i < mapx.rows; i++){
		for (int j = 0; j < mapx.cols; j++){
			if (mapx.at<float>(i, j)>max){
				max = mapx.at<float>(i, j);
			}
			if (mapx.at<float>(i, j)<min){
				min = mapx.at<float>(i, j);
			}

		}
	}
	
	//求出矩阵mapy所包含的数据范围即(min,max)
	float max_y = mapy.at<float>(0, 0);
	float min_y = mapy.at<float>(0, 0);
	for (int i = 0; i < mapy.rows; i++){
		for (int j = 0; j < mapy.cols; j++){
			if (mapy.at<float>(i, j)>max_y){
				max_y = mapy.at<float>(i, j);
			}
			if (mapy.at<float>(i, j)<min_y){
				min_y = mapy.at<float>(i, j);
			}

		}
	}

	//把mxpx右移8位，把mapy右移16位
	for (int i = 0; i < rows; i++){
		for (int j = 0; j < cols; j++){
			//mapx.at<float>(i,j) = (mapx.at<float>(i, j)*256*256);
			int mapx_1 = (mapx.at<float>(i, j) * 256);
			mapx.at<float>(i, j) = mapx_1;
		}
	}

	for (int i = 0; i < rows; i++){
		for (int j = 0; j < cols; j++){
			int mapx_2 = (mapy.at<float>(i, j) * 256 * 256);
			mapy.at<float>(i, j) = mapx_2;
		}
	}

	//mapx = mapx * 256; mapy= mapy*256*256范围
	float max_x_int = mapx.at<float>(0, 0);
	float min_x_int = mapx.at<float>(0, 0);
	for (int i = 0; i < mapx.rows; i++){
		for (int j = 0; j < mapx.cols; j++){
			if (mapx.at<float>(i, j)>max_x_int){
				max_x_int = mapx.at<float>(i, j);
			}
			if (mapx.at<float>(i, j)<min_x_int){
				min_x_int = mapx.at<float>(i, j);
			}

		}
	}

	float max_y_int = mapy.at<float>(0, 0);
	float min_y_int = mapy.at<float>(0, 0);
	for (int i = 0; i < mapy.rows; i++){
		for (int j = 0; j < mapy.cols; j++){
			if (mapy.at<float>(i, j)>max_y){
				max_y_int = mapy.at<float>(i, j);
			}
			if (mapy.at<float>(i, j)<min_y){
				min_y_int = mapy.at<float>(i, j);
			}
		}
	}

	mapx.convertTo(mapx_new, CV_32S, 1, 0);//转换数据类型到32位有符号整型，返回到mapx_new,和mapy_new中保存
	mapy.convertTo(mapy_new, CV_32S, 1, 0);

}

Mat ldc(Mat img,Mat& mapx_new,Mat& mapy_new){
	//Mat out;
	mapx_new.convertTo(mapx_new, CV_32FC1, 1.0 / 256, 0);
	mapy_new.convertTo(mapy_new, CV_32FC1, 1.0 / 65536, 0);
	Mat out_new;                                               //量化之后输出是out
	//Mat mapx_renew, mapy_renew;
	cv::remap(img, out_new, mapx_new, mapy_new, cv::INTER_LINEAR);
	return out_new;
}

int test_ldc()
//int main()
{
	string path = "C:\\Users\\25757\\Documents\\Visual Studio 2013\\Projects\\Project2\\Project2\\10xgain21.png";
	Mat img = imread(path);

	int img_rows = img.rows;
	int img_cols = img.cols;
	cout << img_rows << endl;
	cout << img_cols << endl;;

	if (img.empty())
	{
		printf("%s\n", "File not be found!");
		system("pause");
		return 0;
	}
	else{
		
		Mat out_old = distortion_correction_old(img);   //量化之前输出是out_old
		imwrite("C:\\Users\\25757\\Documents\\Visual Studio 2013\\Projects\\Project2\\Project2\\ldc_new_1.png", out_old);
		imshow("after ldc", out_old);


		Mat mapx_new, mapy_new;
		distortion_correction_quant(img, mapx_new, mapy_new);

		//mapx_new.convertTo(mapx_new, CV_32FC1, 1.0 / 256, 0);
		//mapy_new.convertTo(mapy_new, CV_32FC1, 1.0 / 65536, 0);
		//Mat out;                                               //量化之后输出是out
		//Mat mapx_renew, mapy_renew;
		//cv::remap(img, out, mapx_new, mapy_new, cv::INTER_LINEAR);

		Mat out = ldc(img, mapx_new, mapy_new);


		Mat d;
		absdiff(out_old, out, d);//求量化误差 原来的输出是out_old,新的输出是out,两者做差求绝对值再取平均值
		float ave = mean(d)[0];
		cout << "量化误差是：" << endl;
		cout << ave << endl;

	    cv:Scalar tempVal = cv::mean(d);  //求误差矩阵d的平均值
		float matMean = tempVal.val[0];
		cout << matMean << endl;

		Mat out_LD_Quant = out;
		imwrite("C:\\Users\\25757\\Documents\\Visual Studio 2013\\Projects\\Project2\\Project2\\ldc_Quant_1.png", out_LD_Quant);
		imshow("after ldc", out_LD_Quant);
		waitKey(0);
	}
	return 0;
}