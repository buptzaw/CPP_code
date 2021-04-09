#include<iostream>
#include<cmath>
#include<math.h>
#include<opencv2/opencv.hpp>
#include"sstream"
using namespace std;
using namespace cv;
//remove distort in img///
Mat distortion_correction_w(Mat img)
{
	Mat out;
	//int img_rows = img.rows;
	//int img_cols = img.cols;
	Mat camera_matrixa = (cv::Mat_<float>(3, 3) << 1565.3, 0, 722.88, 0, 1562.659, 541.263, 0, 0, 1);
	//[fx,0,cx,0,fy,cy,0,0,1]
	Mat distortion_coefficientsa = (cv::Mat_<float >(1, 4) << -1.467, 1.036, 0, 0);
	//k1, k2, p1, p2
	Mat mapx, mapy;
	cv::initUndistortRectifyMap(camera_matrixa, distortion_coefficientsa, cv::Mat(), camera_matrixa, cv::Size(img.cols, img.rows), CV_32FC1, mapx, mapy);

	float k1 = distortion_coefficientsa.at<float>(0, 0);
	float k2 = distortion_coefficientsa.at<float>(0, 1);

	float fx = camera_matrixa.at<float>(0, 0);
	float fy = camera_matrixa.at<float>(1, 1);
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
		float div = 1/(1 + k1 * r*r + k2 * r * r * r * r);
		mapx.convertTo(mapx, CV_32F, div, cx*(1 - div));
		mapy.convertTo(mapy, CV_32F, div, cy*(1 - div));
	}
	else
	{
		float r = sqrt(x*x+y*y);
		float div = 1 / (1 + k1 * r*r + k2 * r * r * r * r);
		mapx.convertTo(mapx, CV_32F, div, cx*(1 - div));
		mapy.convertTo(mapy, CV_32F, div, cy*(1 - div));
	}



	cv::remap(img, out, mapx, mapy, cv::INTER_LINEAR);
	return out;
}
Mat distortion_correction1(Mat img)
{
	Mat out;


	// This program realizes the code of distortion removing part. Although we can call OpenCV to remove distortion, it is helpful to realize it by ourselves.
//Distortion parameters
	double k1 = -1.467, k2 = 1.036, p1 = 0.0, p2 = 0.0;
	// 内参
	double fx = 1565.3, fy = 1562.659, cx = 722.88, cy = 541.263;

	cv::Mat image = img;   // The image is grayscale，CV_8UC1
	int rows = image.rows, cols = image.cols;
	//cv::Mat image_undistort = cv::Mat(rows, cols, CV_8UC1);   // The picture after removing distortion
	Mat mapx(rows, cols, CV_32F, 0.00);
	Mat mapy(rows, cols, CV_32F, 0.00);
	float x = cx / fx;
	float y = cy / fy;
	// 
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
				mapx.at<float>(v, u) = u_distorted;
				mapy.at<float>(v, u) = v_distorted;

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
	cv::remap(img, out, mapx, mapy, cv::INTER_LINEAR);
	//// 
	//cv::imshow("distorted", image);
	//cv::imshow("undistorted", image_undistort);
	//cv::waitKey();

	return out;
}
int main()
{
	string image_name = "out_3.jpg";
	string path = "C:\\Users\\25757\\Documents\\Visual Studio 2013\\Projects\\Project2\\Project2\\test_images\\ldc_auto\\";
	Mat img = imread(path + image_name);

	int img_rows = img.rows;
	int img_cols = img.cols;
	cout << img_rows << endl;
	cout << img_cols << endl;

	if (img.empty())
	{
		printf("%s\n", "File not be found!");
		system("pause");
		return 0;
	}
	//imshow("origin", img);
	//waitKey(0);
	Mat out_LDC = distortion_correction_w(img);
	imwrite("C:\\Users\\25757\\Documents\\Visual Studio 2013\\Projects\\Project2\\Project2\\test_images\\ldc_auto\\cpp_out_3.jpg", out_LDC);
	imshow("after ldc", out_LDC);
	waitKey(0);
	return 0;
}