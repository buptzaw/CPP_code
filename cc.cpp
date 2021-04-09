#include<iostream>
#include<cmath>
#include<math.h>
#include<opencv2/opencv.hpp>
#include<vector>
#include<string>
#include<random>
#include"tools.h"

using namespace std;
using namespace cv;

Mat enhance(Mat G)
{
	float ave = mean(G)[0];
	float tmp1 = ave * ave;
	unsigned int d;
	if (tmp1 == 0)
	{
		d = 0;
	}
	else
	{
		int tmp2 = 256 * 256 - (int(tmp1 * (3 * 256 - 2 * ave)) >> 8);
		char N = 0;
		int rst = xf_Inverse(tmp2, 16, &N);
		int Nint = (int)N - 24;
		d = (rst >> Nint) - 256.;
	}
    unsigned short scale = 2;
    Mat g2;
    multiply(G, G, g2, 1.0 / pow(2.0, scale), CV_16SC1);

    Mat minus_g_2;
    multiply(255 - G, 255 - G, minus_g_2, 1.0 / pow(2.0, scale), CV_16SC1);

    Mat minus_g_2_mul_d;
    Mat D(minus_g_2.size(), CV_16SC1, Scalar(d));
    multiply(minus_g_2, D, minus_g_2_mul_d, 1 / 256.0, CV_16SC1);

    Mat denominator;
    add(minus_g_2_mul_d, g2, denominator, noArray(), CV_16SC1);

    Mat out(denominator.size(), CV_16SC1);
    for (int i = 0; i < denominator.rows; ++i)
    {
        for (int j = 0; j < denominator.cols; ++j)
        {
            char N;
            unsigned int res = xf_Inverse((unsigned short)denominator.at<short>(i, j), 16, &N);
            out.at<short>(i, j) = short((res * g2.at<short>(i, j)) >> (N - 8));
        }
    }
    return out;
}

Mat vessel_enhance(Mat img)
{
	img.convertTo(img, CV_16SC3);
	img = img;
	Mat RGB[3];
	split(img, RGB);
	Mat &G = RGB[1];
	Mat new_g = enhance(G);
	RGB[0] = RGB[0] + new_g - G;
	RGB[2] = RGB[2] + new_g - G;
	RGB[1] = new_g;
	Mat out;
	merge(RGB, 3, out);
	out.convertTo(out, CV_8UC3);

	Mat hsv_img;
	cvtColor(out, hsv_img, COLOR_RGB2HSV);
	hsv_img.convertTo(hsv_img, CV_16SC3);
	hsv_img = hsv_img;
	Mat HSV[3];
	split(hsv_img, HSV);
	Mat &S = HSV[1];
	HSV[1] = enhance(S);
	merge(HSV, 3, hsv_img);
	hsv_img.convertTo(hsv_img, CV_8UC3);
	cvtColor(hsv_img, out, COLOR_HSV2RGB);

	return out;
}

Mat contrast_auto_control(Mat img)
{
	Mat contrast_img;
	equalizeHist(img, contrast_img);

	return contrast_img;
}
Mat contrast_nlinear(Mat img,double m_coeffcient,double E_coeffcient)
{
	Mat Y = img.clone();
	Mat Y_32F;
	Y.convertTo(Y_32F, CV_32F);


	Mat tmp_m, tmp_sd;
	//double mean_c = 0;
	meanStdDev(img, tmp_m, tmp_sd);
	int mean_img = tmp_m.at<double>(0, 0);
	//std_c = tmp_sd.at<double>(0, 0);

	int d =  mean_img - 128;
	int abs_d = abs(d);

	int m = 128 + d * m_coeffcient;
	double E = 2.8 - abs_d * E_coeffcient;

	for (int i = 0; i < Y_32F.rows; i++)
	{
		for (int j = 0; j < Y_32F.cols; j++)
		{
			double e = pow(E,(m - Y_32F.at<float>(i, j))/m);
			Y_32F.at<float>(i, j) = pow((Y_32F.at<float>(i, j)) / 255, e) * 255;
		}
	}

	Y_32F.convertTo(Y, CV_8UC1);
	return Y;
}


bool getVarianceMean_Quantization(Mat &scr_Y, Mat &localMeanMatrix, Mat &localVarianceVarMatrix, int winSize)
{
	if (winSize % 2 == 0)
	{
		return false;
	}

	int reflectBorderSize = (winSize - 1) / 2;
	Mat scr_Y_ref;
	copyMakeBorder(scr_Y, scr_Y_ref, reflectBorderSize, reflectBorderSize, reflectBorderSize, reflectBorderSize, BORDER_REFLECT);

	for (int i = 0; i < scr_Y.rows; i++)
	{
		for (int j = 0; j < scr_Y.cols; j++)
		{
			Mat mean_tmp, var_tmp;
			Mat mean_uchar(1, 1, CV_8UC1);
			Mat var_inverse_uchar(1, 1, CV_8UC1);

			Mat LocalMatrix = scr_Y_ref(Rect(j, i, winSize, winSize));
			meanStdDev(LocalMatrix, mean_tmp, var_tmp);

			char *N = (char*)malloc(32);
			int rst = xf_Inverse(var_tmp.at<double>(0, 0) * 256.0, 16, N);
			int Nint = *N;
			double n = rst >> (Nint - 16);
			var_inverse_uchar.at<uchar>(0, 0) = rst >> (Nint - 16);
			localVarianceVarMatrix.at<uchar>(i, j) = var_inverse_uchar.at<uchar>(0, 0);

			mean_tmp.convertTo(mean_uchar, CV_8UC1, 1, 0);
			localMeanMatrix.at<uchar>(i, j) = mean_uchar.at<uchar>(0, 0);
		}
	}
}


bool adaptContrastEnhancement_Quantization(Mat &scr_Y, Mat &rst_Y, int winSize, int maxCg)
{
	Mat localMeanMatrix(Size(scr_Y.cols, scr_Y.rows), CV_8UC1), localVarVarianceMatrix(Size(scr_Y.cols, scr_Y.rows), CV_8UC1);
	
	if (!getVarianceMean_Quantization(scr_Y, localMeanMatrix, localVarVarianceMatrix, winSize))
	{
		cerr << "False";
		return false;
	}

	Mat mean_global, var_global;
	float alpha = 0.2;
	meanStdDev(scr_Y, mean_global, var_global);

	Mat cga(Size(scr_Y.cols, scr_Y.rows), CV_8UC1);
	Mat lVM_16U;
	localVarVarianceMatrix.convertTo(lVM_16U, CV_16UC1, 1, 0);
	Mat cg_tmp(Size(scr_Y.cols, scr_Y.rows), CV_16UC1);

	cg_tmp = ( alpha * mean_global.at<double>(0, 0) / 16 )* lVM_16U;
	cg_tmp.convertTo(cga, CV_8UC1, 1, 0);

	Mat cgb(Size(scr_Y.cols, scr_Y.rows), CV_8UC1);
	IplImage* cga_ip = &IplImage(cga);
	IplImage* cgb_ip = &IplImage(cgb);
	cvMinS(cga_ip, 16 * maxCg, cgb_ip);

	Mat cgc(Size(scr_Y.cols, scr_Y.rows), CV_8UC1);
	IplImage* cgc_ip = &IplImage(cgc);
	cvMaxS(cgb_ip, 16, cgc_ip);

	Mat cg = cvarrToMat(cgc_ip);
	Mat scr_Y_16S, lMM_16S, cg_16S;

	scr_Y.convertTo(scr_Y_16S, CV_16SC1, 1, 0);
	localMeanMatrix.convertTo(lMM_16S, CV_16SC1, 1, 0);
	cg.convertTo(cg_16S, CV_16SC1, 1, 0);

	Mat tmp1(Size(scr_Y.cols, scr_Y.rows), CV_16SC1);
	Mat tmp2(Size(scr_Y.cols, scr_Y.rows), CV_16SC1);
	Mat enhanceMatrix(Size(scr_Y.cols, scr_Y.rows), CV_16SC1);

	tmp1 = scr_Y_16S - lMM_16S;

	multiply(cg_16S, tmp1, tmp2, 1.0/16.0);
	
	enhanceMatrix = lMM_16S + tmp2;
	enhanceMatrix.convertTo(rst_Y, CV_8UC1, 1, 0);
}

bool ACE_test()
{
	String scr_PATH = "./";
	String scr_NAME = "hand3.png";
	Mat scr_RGB = imread("C:\\Users\\liuzh\\Downloads\\hand3.png", IMREAD_COLOR);

	Mat scr_YCC, scr_Y, rst_Y, rst_YCC, rst_RGB;
	int localSize = 217, maxCG = 10;
	vector<Mat> channels(3);

	if (!scr_RGB.data)
	{
		cout << "ACE_main函数中发生imread错误";
		system("pause");
		return false;
	}

	cvtColor(scr_RGB, scr_YCC, COLOR_RGB2YCrCb);
	split(scr_YCC, channels);
	scr_Y = channels[0];
	cout << "RGB转YCC" << endl;

	adaptContrastEnhancement_Quantization(scr_Y, rst_Y, localSize, maxCG);

	channels[0] = rst_Y;
	merge(channels, rst_YCC);
	cvtColor(rst_YCC, rst_RGB, COLOR_YCrCb2RGB);
	imwrite(scr_PATH + "hand3_ACE_Qu_LocalSize217_v0.44_16_test.png", rst_RGB);
}


int cc_test()
{
	string image_name = "ee03y.png";
	string path = "D:/work/isp-master/images/contrast/";
	Mat img = imread(path + image_name,0);

	//Mat gray_img;
	//cvtColor(img, gray_img, CV_BGR2GRAY);

	double m_coeffcient = 0.35;
	double E_coeffcient = 0.012;
	Mat out_contrast_auto = contrast_nlinear(img, m_coeffcient, E_coeffcient);
	imwrite("D:\\work\\isp-master\\images\\contrast\\ee03y_c.png", out_contrast_auto);
	return 0;
}

void vessel_enhance_test()
{
	Mat img = imread("D:\\project\\isp_modules\\isp_python\\images\\blood2.png");
	Mat img_rgb;
	cvtColor(img, img_rgb, COLOR_BGR2RGB);
	img_rgb = vessel_enhance(img_rgb);
	cvtColor(img_rgb, img, COLOR_RGB2BGR);
	imwrite("D:\\project\\isp_modules\\isp_python\\images\\enh_blood2.png", img);
}
