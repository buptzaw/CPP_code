#include <iostream>
#include <fstream>
using namespace std;

#include<opencv2/opencv.hpp>
#include <opencv2/core/core.hpp>
using namespace cv;

#include "./tools.h"

using BayerType = uchar;

Mat GreenMatch_ushort(Mat rawimge, float gm_thr = 0.95, float thr = 0.04)
{
	int margin = 3;
	int oj = 2;
	int oi = 3;
	Mat gm_rawimge;
	rawimge.convertTo(gm_rawimge, CV_16U);
	int h = rawimge.rows;
	int w = rawimge.cols;
	cout << "img_rows " << h << endl;
	cout << "img_cols " << w << endl;


	int i, j;
	double m1, m2, c1, c2;
	int o1_1, o1_2, o1_3, o1_4;
	int o2_1, o2_2, o2_3, o2_4;
	int maximum = 255;
	double f;
	for (j = oj; j < h - margin; j += 2)
		for (i = oi; i < w - margin; i += 2)
		{


			o1_1 = rawimge.at<uchar>(j - 1, i - 1);
			o1_2 = rawimge.at<uchar>(j - 1, i + 1);
			o1_3 = rawimge.at<uchar>(j + 1, i - 1);
			o1_4 = rawimge.at<uchar>(j + 1, i + 1);
			o2_1 = rawimge.at<uchar>(j - 2, i);
			o2_2 = rawimge.at<uchar>(j + 2, i);
			o2_3 = rawimge.at<uchar>(j, i - 2);
			o2_4 = rawimge.at<uchar>(j, i + 2);

			m1 = (o1_1 + o1_2 + o1_3 + o1_4) / 4.0;
			m2 = (o2_1 + o2_2 + o2_3 + o2_4) / 4.0;

			c1 = (abs(o1_1 - o1_2) + abs(o1_1 - o1_3) + abs(o1_1 - o1_4) +
				abs(o1_2 - o1_3) + abs(o1_3 - o1_4) + abs(o1_2 - o1_4)) /
				6.0;
			c2 = (abs(o2_1 - o2_2) + abs(o2_1 - o2_3) + abs(o2_1 - o2_4) +
				abs(o2_2 - o2_3) + abs(o2_3 - o2_4) + abs(o2_2 - o2_4)) /
				6.0;
			if ((rawimge.at<uchar>(j, i) < maximum * gm_thr) && (c1 < maximum * thr) &&
				(c2 < maximum * thr))
			{
				gm_rawimge.at<ushort>(j, i) = gm_rawimge.at<ushort>(j, i) * m1 / m2;
				if (gm_rawimge.at<ushort>(j, i) > 255)
				{
					gm_rawimge.at<ushort>(j, i) = 255;
				}
			}
		}
	gm_rawimge.convertTo(gm_rawimge, CV_8U);
	return gm_rawimge;
}

Mat GreenMatch_inverse(Mat rawimge, float gm_thr = 0.95, float thr = 0.04)
{
	int margin = 3;
	int oj = 2;
	int oi = 3;
	Mat gm_rawimge = rawimge.clone();
	//rawimge.convertTo(gm_rawimge, CV_16U);
	int h = rawimge.rows;
	int w = rawimge.cols;
	cout << "img_rows " << h << endl;
	cout << "img_cols " << w << endl;


	int i, j;
	//double m1, m2, c1, c2;
	int o1_1, o1_2, o1_3, o1_4;
	int o2_1, o2_2, o2_3, o2_4;
	int maximum = 255;
	//int max_gm_thr = maximum * gm_thr;
	//int max_thr = maximum * thr;
	//float max_gm_thr = maximum * gm_thr;
	//float max_thr = maximum * thr;
	ushort c1, c2;
	//ushort max_gm_thr = maximum * gm_thr;
	//ushort max_thr = maximum * thr * 6;
	float max_gm_thr = maximum * gm_thr;
	float max_thr = maximum * thr * 6;
	Mat m1_mat(h, w, CV_16UC1, Scalar(4));
	Mat m2_mat(h, w, CV_16UC1, Scalar(4));
	int inverse_scale = 0;
	Mat m2_inversed(h, w, CV_16UC1, Scalar(pow(2.0, 16 - inverse_scale)/4.0));
	
	//double f;
	for (j = oj; j < h - margin; j += 2)
		for (i = oi; i < w - margin; i += 2)
		{


			o1_1 = rawimge.at<uchar>(j - 1, i - 1);
			o1_2 = rawimge.at<uchar>(j - 1, i + 1);
			o1_3 = rawimge.at<uchar>(j + 1, i - 1);
			o1_4 = rawimge.at<uchar>(j + 1, i + 1);
			o2_1 = rawimge.at<uchar>(j - 2, i);
			o2_2 = rawimge.at<uchar>(j + 2, i);
			o2_3 = rawimge.at<uchar>(j, i - 2);
			o2_4 = rawimge.at<uchar>(j, i + 2);

			//m1 = (o1_1 + o1_2 + o1_3 + o1_4) / 4.0;
			//m2 = (o2_1 + o2_2 + o2_3 + o2_4) / 4.0;
			//m1_mat.at<ushort>(j, i) = 1;
			//m2_mat.at<ushort>(j, i) = 1;


			c1 = (abs(o1_1 - o1_2) + abs(o1_1 - o1_3) + abs(o1_1 - o1_4) +
				abs(o1_2 - o1_3) + abs(o1_3 - o1_4) + abs(o1_2 - o1_4));

			c2 = (abs(o2_1 - o2_2) + abs(o2_1 - o2_3) + abs(o2_1 - o2_4) +
				abs(o2_2 - o2_3) + abs(o2_3 - o2_4) + abs(o2_2 - o2_4));

			if ((rawimge.at<uchar>(j, i) < max_gm_thr) && (c1 < max_thr) &&
				(c2 < max_thr))
			{
				m1_mat.at<ushort>(j, i) = (o1_1 + o1_2 + o1_3 + o1_4);
				m2_mat.at<ushort>(j, i) = (o2_1 + o2_2 + o2_3 + o2_4);

				char N = 0;
				unsigned int inv = xf_Inverse(m2_mat.at<ushort>(j, i), 16, &N);
				//cout << inv << endl;

				//cout << (float)rawimge.at<uchar>(j, i) << endl;
				//cout << (float)m1_mat.at<ushort>(j, i) << endl;
				//cout << (float)m2_mat.at<ushort>(j, i) << endl;
				int shift = 16 - inverse_scale - int(N);
				//cout << shift << endl;
				if (shift > 0)
				{
					inv = inv << shift;
				}
				else
				{
					inv = inv >> (-shift);
				}
				//cout << inv << endl;
				m2_inversed.at<ushort>(j, i) = unsigned short(inv);

				//cout << m2_inversed.at<ushort>(j, i) << endl;

				//cout << rawimge.at<uchar>(j, i)* m1_mat.at<ushort>(j, i)*m2_inversed.at<ushort>(j, i) / pow(2.0, 16 - inverse_scale) << endl;
				//cout << j << endl;
				//cout << i << endl;
				//cout << m2_mat.at<ushort>(j, i) << endl;
				

			}
		}

	////Mat m2_inversed(m2_mat.size(), CV_16UC1);
	////int inverse_scale = 0;
	//for (int i = 0; i < h; ++i)
	//{
	//	for (int j = 0; j < w; ++j)
	//	{
	//		char N = 0;
	//		unsigned int inv = xf_Inverse(m2_mat.at<ushort>(i, j), 16, &N);
	//		// Q(inverse_scale).(16 - inverse_scale)
	//		int shift = 16 - inverse_scale - int(N);
	//		//cout << int(N) << endl;
	//		if (shift > 0)
	//		{
	//			inv = inv << shift;
	//		}
	//		else
	//		{
	//			inv = inv >> (-shift);
	//		}
	//		m2_inversed.at<ushort>(i, j) = unsigned short(inv);



	//	}
	//}
	Mat temp;
	//multiply(rawimge, m1_mat, temp, 1/4.0, CV_16U);
	//multiply(temp, m2_inversed, gm_rawimge, 4.0 / pow(2.0, 16 - inverse_scale), CV_8U);
	multiply(rawimge, m1_mat, temp, 1/4.0, CV_16U);
	multiply(temp, m2_inversed, gm_rawimge, 4.0 / pow(2.0, 16 - inverse_scale), CV_8U);
	//cout << (float)temp.at<ushort>(2, 3) << endl;
	//cout << (float)gm_rawimge.at<uchar>(2, 3) << endl;

	//multiply(w, Y_2(rect), temp, 1.0 / pow(2.0, 16 - w_scale), CV_16UC1);
	//gm_rawimge.convertTo(gm_rawimge, CV_8U);
	return gm_rawimge;
}
int main()
{

	Mat img = imread("C:\\Users\\25757\\Documents\\Visual Studio 2013\\Projects\\Project2\\Project2\\test_images\\gb\\bayer_color_raw2_np.bmp",0);
	Mat gm_ori_bgr;
	cvtColor(img, gm_ori_bgr, COLOR_BayerRG2BGR);
	//imwrite("D:/work/isp_python/gm_ori_c.png", gm_ori_bgr);

	//Mat blc_output = blc(img, bl_r, bl_gr, bl_gb, bl_b, scale_r, scale_gr, scale_gb, scale_b, clip_min, clip_max);
	Mat gb_output = GreenMatch_inverse(img);

	Mat gm_bgr;
	cvtColor(gb_output, gm_bgr, COLOR_BayerRG2BGR);
	imshow("gm_bgr", gm_bgr);
	imwrite("C:\\Users\\25757\\Documents\\Visual Studio 2013\\Projects\\Project2\\Project2\\test_images\\gb\\gm_c_inverse.png", gm_bgr);
	//imshow("gm_ori_bgr", gm_ori_bgr);
	waitKey(0);
	return 0;
}