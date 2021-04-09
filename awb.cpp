#include <highgui/highgui.hpp>
#include <opencv2/opencv.hpp>
#include <iostream>
#include <stdio.h>
#include <stdlib.h>
#include <windows.h>
#include <algorithm>
#include <math.h>

#include "tools.h"
using namespace cv;
using namespace std;
Mat GrayWorld(Mat src)
{
	vector<Mat> imageRGB;
	split(src, imageRGB);

	double R, G, B;
	B = mean(imageRGB[0])[0];
	G = mean(imageRGB[1])[0];
	R = mean(imageRGB[2])[0];

	double KR, KG, KB;
	KB = (R + G + B) / (3 * B);
	KG = (R + G + B) / (3 * G);
	KR = (R + G + B) / (3 * R);

	imageRGB[0] = imageRGB[0] * KB;
	imageRGB[1] = imageRGB[1] * KG;
	imageRGB[2] = imageRGB[2] * KR;
	merge(imageRGB, src);
	return src;
}
Mat PerfectReflectionAlgorithm(Mat src) {
	int row = src.rows;
	int col = src.cols;
	Mat dst(row, col, CV_8UC3);
	int HistRGB[767] = { 0 };
	int MaxVal = 0;
	for (int i = 0; i < row; i++) {
		for (int j = 0; j < col; j++) {
			MaxVal = max(MaxVal, (int)src.at<Vec3b>(i, j)[0]);
			MaxVal = max(MaxVal, (int)src.at<Vec3b>(i, j)[1]);
			MaxVal = max(MaxVal, (int)src.at<Vec3b>(i, j)[2]);
			int sum = src.at<Vec3b>(i, j)[0] + src.at<Vec3b>(i, j)[1] + src.at<Vec3b>(i, j)[2];
			HistRGB[sum]++;
		}
	}
	int Threshold = 0;
	int sum = 0;
	for (int i = 766; i >= 0; i--) {
		sum += HistRGB[i];
		if (sum > row * col * 0.1) {
			Threshold = i;
			break;
		}
	}
	int AvgB = 0;
	int AvgG = 0;
	int AvgR = 0;
	int cnt = 0;
	for (int i = 0; i < row; i++) {
		for (int j = 0; j < col; j++) {
			int sumP = src.at<Vec3b>(i, j)[0] + src.at<Vec3b>(i, j)[1] + src.at<Vec3b>(i, j)[2];
			if (sumP > Threshold) {
				AvgB += src.at<Vec3b>(i, j)[0];
				AvgG += src.at<Vec3b>(i, j)[1];
				AvgR += src.at<Vec3b>(i, j)[2];
				cnt++;
			}
		}
	}
	AvgB /= cnt;
	AvgG /= cnt;
	AvgR /= cnt;
	for (int i = 0; i < row; i++) {
		for (int j = 0; j < col; j++) {
			int Blue = src.at<Vec3b>(i, j)[0] * MaxVal / AvgB;
			int Green = src.at<Vec3b>(i, j)[1] * MaxVal / AvgG;
			int Red = src.at<Vec3b>(i, j)[2] * MaxVal / AvgR;
			if (Red > 255) {
				Red = 255;
			}
			else if (Red < 0) {
				Red = 0;
			}
			if (Green > 255) {
				Green = 255;
			}
			else if (Green < 0) {
				Green = 0;
			}
			if (Blue > 255) {
				Blue = 255;
			}
			else if (Blue < 0) {
				Blue = 0;
			}
			dst.at<Vec3b>(i, j)[0] = Blue;
			dst.at<Vec3b>(i, j)[1] = Green;
			dst.at<Vec3b>(i, j)[2] = Red;
		}
	}
	return dst;
}

int EstimateIlluminantGrey(Mat I, double p) {

	int Ic = 0, L = 256;
	int width = I.cols, height = I.rows;

	int pixelTh = 0;
	int histI[256] = { 0 };

	pixelTh = int(p * width * height);
	//histI=imhist(I);
	//calc image histogram,I =B or G or R channel
	for (int i = 0; i < height; i++) {
		for (int j = 0; j < width; j++) {
			int pixel = I.at<uchar>(i, j);
			histI[pixel]++;
		}
	}
	//get max&min value 	
	//minMaxIdx(I, &minVal, &maxVal);
	int minVal = I.at<uchar>(0, 0), maxVal = I.at<uchar>(0, 0);
	for (int i = 0; i < height; i++) {
		for (int j = 0; j < width; j++) {
			if (I.at<uchar>(i, j) > maxVal) {
				maxVal = I.at<uchar>(i, j);
			}
			if (I.at<uchar>(i, j) < minVal) {
				minVal = I.at<uchar>(i, j);
			}

		}
	}
	int sum = 0;
	for (int k = maxVal; k >= minVal; k--)
	{

		sum = sum + histI[k];

		if (sum > pixelTh)
		{
			Ic = k;
			break;
		}

	}
	return Ic;
}
double EstimateCCT(int R, int G, int B) {

	//Constant parameters from 
	double A0 = -949.86315;
	double A1 = 6253.80338;
	double A2 = 28.70599;
	double A3 = 0.00004;

	double t1 = 0.92159;
	double t2 = 0.20039;
	double t3 = 0.07125;

	double xe = 0.3366;
	double ye = 0.1735;
	double x = 0, y = 0;
	double H = 0, CCT = 0;
	//Calculate x and y from estimated illuminant values

	//XYZ_Conv_matrix = [ 0.4124 0.3576 0.1805;
	//					0.2126 0.7152 0.0722;
	//					0.0193 0.152 0.9505];
	//XYZ = XYZ_Conv_matrix * double(iEstm');

	double X = 0.4124 * R + 0.3576 * G + 0.1805 * B;
	double Y = 0.2126 * R + 0.7152 * G + 0.0722 * B;
	double Z = 0.0193 * R + 0.152 * G + 0.9505 * B;
	x = X / (X + Y + Z);
	y = Y / (X + Y + Z);

	H = -((x - xe) / (y - ye));

	CCT = A0 + (A1 * exp(H / t1)) + (A2 * exp(H / t2)) + (A3 * exp(H / t3));

	return CCT;

}
Mat performAWB(Mat src, double gray_percent) {

	Mat BGRChannel[3];
	//Mat dst;
	int width = src.cols, height = src.rows;
	Mat dst(height, width, CV_8UC3);
	split(src, BGRChannel);
	int Bc, Gc, Rc;
	Bc = EstimateIlluminantGrey(BGRChannel[0], gray_percent);
	Gc = EstimateIlluminantGrey(BGRChannel[1], gray_percent);
	Rc = EstimateIlluminantGrey(BGRChannel[2], gray_percent);
	cout << "Bc  " << Bc << endl;
	cout << "Gc  " << Gc << endl;
	cout << "Rc  " << Rc << endl;
	//calc current_CCT and reference_CCT
	double CCT_Estm = 0, CCT_Ref = 0;
	CCT_Estm = EstimateCCT(Rc, Gc, Bc);
	cout << "CT_estimate " << CCT_Estm << "K" << endl;
	CCT_Ref = EstimateCCT(Gc, Gc, Gc);
	cout << "CT_reference " << CCT_Ref << "K" << endl;
	//int iEstm[] = {Rc,Gc,Bc};
	//int iRef[] = {Gc,Gc,Gc};
	double Kr, Kg, Kb;
	Kr = (double)Gc / Rc;//ref_R/estm_R
	Kg = (double)Gc / Gc;//ref_G/estm_G
	Kb = (double)Gc / Bc;//ref_B/estm_B

	cout << "Kr " << Kr << endl;
	cout << "Kg " << Kg << endl;
	cout << "Kb " << Kb << endl;

	//Mat K = (Mat_<double>(3,3)<< Kr, 0, 0, 0, Kg, 0, 0, 0, Kb);
	//Mat K = (Mat_<double>(3, 3) << Kb, 0, 0, 0, Kg, 0, 0, 0, Kr);

	double Tr, Tg, Tb;
	Tr = fmax(1, (CCT_Estm - CCT_Ref) / 100) * (Kr - 1);
	Tg = 0;
	Tb = fmax(1, (CCT_Ref - CCT_Estm) / 100) * (Kb - 1);
	cout << "Tr " << Tr << endl;
	cout << "Tg " << Tg << endl;
	cout << "Tb " << Tb << endl;
	//Mat T = (Mat_<double>(3,1)<< Tr, Tg, Tb);
	//Mat T = (Mat_<double>(3, 1) << Tb, Tg, Tr);
	int FWB[3];
	for (int i = 0; i < height; i++) {
		for (int j = 0; j < width; j++) {
			//Vec<int,3>  Fxy(src.at<Vec3b>(i, j)[0], src.at<Vec3b>(i, j)[1], src.at<Vec3b>(i, j)[2])£»
			//Mat Fxy = (Mat_<double>(3, 1) << src.at<Vec3b>(i, j)[0], src.at<Vec3b>(i, j)[1], src.at<Vec3b>(i, j)[2]);
			//Mat FWB = int(K * Fxy + T);
			FWB[0] = round(Kb * src.at<Vec3b>(i, j)[0] + Tb);
			FWB[1] = round(Kg * src.at<Vec3b>(i, j)[1] + Tg);
			FWB[2] = round(Kr * src.at<Vec3b>(i, j)[2] + Tr);

			for (int n = 0; n < 3; n++) {

				if (FWB[n] > 255) {
					FWB[n] = 255;
				}
				else if (FWB[n] < 0) {
					FWB[n] = 0;
				}

				dst.at<Vec3b>(i, j)[n] = FWB[n];

				//if (dst.at<Vec3b>(i,j)[n] > 255) {
				//	dst.at<Vec3b>(i,j)[n] = 255;
				//}else if (dst.at<Vec3b>(i, j)[n] < 0){
				//	dst.at<Vec3b>(i, j)[n] = 0;
				//}

			}
		}
	}

	return dst;

}


//wb gains
float rgb_gains[9][3] = {
		{1.435142234586458, 1.090118818222525, 0.7215664450015953},
		{1.2336051895391902, 1.0752630801753014, 0.7940522879505862},
		{1.1045598304781448, 1.0714376833897976, 0.8610768023244689},
		{1.0125843398723582, 1.0726349864311384, 0.9258021867994429},
		{0.9460532737957751, 1.0775160166094673,0.9853026124793317 },
		{0.9121108201469847, 1.081190916605073, 1.021726020409411},
		{0.8778454398505593, 1.0854843263829383, 1.0642831862853068},
		{0.847059586960524, 1.0918622373642781, 1.1067100322301977},
		{0.8289882729297019, 1.0947860396945495, 1.135989497043594} };


/// <summary>
/// Detect white point in img and calculate R/G,B/G
/// </summary>
/// <param name="img"></param>
/// <param name="x">B/G ,R/G</param>
void white_detect(Mat img, float* x)
{
	Mat yuv_img;
	cvtColor(img, yuv_img, COLOR_RGB2YUV);

	//YUV region
	Mat YUV[3];
	split(yuv_img, YUV);
	Mat y = YUV[0], u = YUV[1], v = YUV[2];

	double avl_u = mean(u)[0], avl_v = mean(v)[0];
	double avl_du = mean(abs(u - avl_u))[0], avl_dv = mean(abs(v - avl_v))[0];
	double ratio = 0.5; //If the value is too large or too small, the color temperature will develop to two extremes

	Mat valuekey;

	bitwise_or(
		(abs(u - (avl_u + avl_du)) < avl_du * ratio),
		(abs(v - (avl_v + avl_dv)) < avl_dv * ratio),
		valuekey);

	int h = img.rows, w = img.cols;
	Mat num_y = Mat::zeros(h, w, CV_8UC1);
	y.copyTo(num_y, valuekey);

	Mat hist;
	float range[] = { 0, 255 };
	const float* histRange[] = { range };
	int histBinNum = 256;

	calcHist(&y, 1, 0, valuekey, hist, 1, &histBinNum, histRange, true, false);

	int ysum = countNonZero(valuekey);
	int Y = 250; // lower for avoiding detecting exposure points
	int num = 0, key = 0;
	while (Y >= 0)
	{
		num += hist.at<float>(Y, 0);
		if (num > 0.05 * ysum) // get the first 5% points for white
		{
			key = Y;
			break;
		}
		Y--;
	}

	Mat sumkey = (num_y >= key);

	//Mat tmp;
	//img.copyTo(tmp);
	//tmp.setTo(Scalar(0, 255, 0), sumkey);
	//cvtColor(tmp, tmp, COLOR_RGB2BGR);
	//imwrite("tmp-mask.png", tmp);

	Mat white = Mat::zeros(h, w, CV_8UC3);
	img.copyTo(white, sumkey);

	Mat RGB[3];
	split(white, RGB);
	cout << ysum << "," << num << "," << key << endl;

	int gscale = 7;
	unsigned short ave_g = mean(RGB[1])[0] * pow(2.0, gscale);

	char N = 0;
	unsigned int inv = xf_Inverse(ave_g, 16 - gscale, &N);
	double iave_g = inv * pow(0.5, int(N));


	x[0] = mean(RGB[2])[0] * iave_g;
	x[1] = mean(RGB[0])[0] * iave_g;

	cout << "x:" << x[0] << "  " << x[1] << endl;
}

/// <summary>
/// get wb gains based color temperature
/// </summary>
/// <param name="img"></param>
/// <param name="gain">wb gians:R G B</param>
/// <param name="y0"></param>
/// <returns></returns>
Mat rgb_wb_CT(Mat img, float* gain, float& y0, float gains[][3], float a = 27.284, float b = -81.096, float c = 64.436)
{
	int CT[9] = { 3000, 3500,4000, 4500, 5000, 5500,6000, 6500, 7000 };
	Mat RGB[3];
	split(img, RGB);
	float x[2];
	white_detect(img, x);


	y0 = a * (pow(x[0], 2.0)) + b * x[0] + c;

	cout << "Color Temperature " << y0 * 500 << endl;
	if (y0 < 6)
	{
		y0 = 6;
	}
	else if (y0 > 14)
	{
		y0 = 14;
	}

	int t0 = int(y0 - 6);
	int t1 = int(y0 - 5);

	float w = (y0 - (t0 + 6));
	for (int i = 0; i < 3; i++)
	{
		gain[i] = gains[t0][i] + w * (gains[t1][i] - gains[t0][i]);
	}
	cout << "Color Temperature Interval " << CT[t0] << "," << CT[t1] << "," << gain[0] << endl << gain[1] << endl << gain[2] << endl;

	for (int i = 0; i < 3; i++)
	{
		RGB[i].convertTo(RGB[i], CV_8UC1, gain[i]);
	}
	Mat output;
	merge(RGB, 3, output);
	return output;
}

void test_CT_awb()
{
	cout << "=================CT AWB=====================" << endl;
	//Mat img = imread("F:/python/test/raw/CT/DNG/test.png");
	Mat img = imread("C:\\Users\\25757\\Documents\\Visual Studio 2013\\Projects\\Project2\\Project2\\ct-awb_bpc_nr\\test.png");
	if (!img.data)
	{
		cout << "ERROR : could not load image.";
		return;
	}
	cvtColor(img, img, COLOR_BGR2RGB);

	float gain[3];
	float y0;
	Mat rgb_img = rgb_wb_CT(img, gain, y0, rgb_gains);
	cout << "CT:" << y0 << endl;

	cvtColor(rgb_img, rgb_img, COLOR_RGB2BGR);
	//imwrite("./CTAWB.png", rgb_img);
	imwrite("C:\\Users\\25757\\Documents\\Visual Studio 2013\\Projects\\Project2\\Project2\\ct-awb_bpc_nr\\CTAWB.png", rgb_img);
	imshow("after ldc", rgb_img);
	waitKey(0);

}

