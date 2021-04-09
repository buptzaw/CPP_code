#include <iostream>
#include <cmath>
#include <opencv2/opencv.hpp>
#include <string>
#include <sstream>
using namespace std;
using namespace cv;

const int ISO_Table[] = { 50, 100, 125, 160, 200, 250, 320, 400, 500, 640, 800, 1000, 1250, 1600, 3200, 4000, 5000, 6400 };
const uchar desired_lum = 128;
const uchar thred_ISO = 10;

const unsigned short desired_lum_bayer = 128;
const uchar BAYER_BIT_WIDTH = 10;

// Bayer
unsigned short cal_bayer_mean(Mat img){
	// input: bayer img (RGGB/BGGR)
	// output: bayer bits
	Mat img_g(Size(img.cols / 2, img.rows ), img.type(), Scalar(0));
	for (int i = 0; i < img.rows; i++){
		if (i % 2 == 0){
			for (int j = 1; j < img.cols; j = j + 2)
				img_g.at<uchar>(i, (j - 1) / 2) = img.at<uchar>(i, j);
		}
		else{
			for (int j = 0; j < img.cols; j = j + 2)
				img_g.at<uchar>(i, j/2) = img.at<uchar>(i, j);
		}
	}
	return (unsigned short)(mean(img_g).val[0]);

}

unsigned short cal_bayer_peak(Mat img, int top = 207360){
	// input: bayer img (RGGB/BGGR)
	// output: bayer bits
	Mat img_g(Size(img.cols / 2, img.rows ), img.type(), Scalar(0));
	for (int i = 0; i < img.rows; ++i){
		if (i % 2 == 0){
			for (int j = 1; j < img.cols; j = j + 2)
				img_g.at<uchar>(i, (j - 1) / 2) = img.at<uchar>(i, j);
		}
		else{
			for (int j = 0; j < img.cols; j = j + 2)
				img_g.at<uchar>(i, j / 2) = img.at<uchar>(i, j);
		}
	}

	int histSize = pow(2, BAYER_BIT_WIDTH);
	float range[] = { 0, pow(2, BAYER_BIT_WIDTH) };
	const float *histRanges = { range };
	Mat hist;
	calcHist(&img_g, 1, 0, Mat(), hist, 1, &histSize, &histRanges, true, false);

	float total = 0;
	float lum = 0;
	for (int i = range[1]-1; i >= 0; i--){
		total += hist.at<float>(i);
		lum += hist.at<float>(i)*i;
		if (total > top) break;
	}

	return (unsigned short)(lum / total);

}

unsigned short cal_bayer_block(Mat img, int index){
	// input: bayer img (RGGB/BGGR)  index:0~8
	// output: bayer bits
	Mat img_g(Size(img.cols / 2, img.rows ), img.type(), Scalar(0));
	for (int i = 0; i < img.rows; ++i){
		if (i % 2 == 0){
			for (int j = 1; j < img.cols; j = j + 2)
				img_g.at<uchar>(i, (j - 1) / 2) = img.at<uchar>(i, j);
		}
		else{
			for (int j = 0; j < img.cols; j = j + 2)
				img_g.at<uchar>(i, j / 2) = img.at<uchar>(i, j);
		}
	}
	int rows = img_g.size[0];
	int cols = img_g.size[1];
	int block_h = rows / 3;
	int block_w = cols / 3;
	int position[9][2] = { { 0, 0 }, { 0, 1 }, { 0, 2 },
	{ 1, 0 }, { 1, 1 }, { 1, 2 },
	{ 2, 0 }, { 2, 1 }, { 2, 2 } };
	//boundingbox(block_w*position[index][1], block_h*position[index][0], block_w, block_h)
	Mat block = img_g(Rect(block_w*position[index][1], block_h*position[index][0], block_w, block_h));
	return (unsigned short)(mean(block).val[0]);

}

float adjust_ISO(Mat img, float ISO_cur, uchar cur_luminance, int thred_ISO){
	if (desired_lum - thred_ISO <= cur_luminance && cur_luminance <= desired_lum + thred_ISO){
		return ISO_cur;
	}
	int ISO_next_quan = desired_lum*ISO_cur / cur_luminance;

	int min = INT_MAX;
	float result = 0;
	int len = sizeof(ISO_Table) / sizeof(ISO_Table[0]);
	for (int i = 0; i < len; i++){
		if (abs(ISO_next_quan - ISO_Table[i]) < min){
			min = abs(ISO_next_quan - ISO_Table[i]);
			result = ISO_Table[i];
		}
	}
	return result;
}

void AG_test()
//int main()
{
	//uchar (*lum_cal_method)(Mat img) = cal_luminance_peak;

	// init iso.
	int frame_count = 0;
	float iso = 50;

	while (true)
	{
		// Get a frame with new iso.
		ostringstream iis;
		iis << int(iso);
		cout << iis.str() << endl;
		//Mat src_bgr = imread("D:\\project\\endoscope\\test\\AE\\ISO_" + iis.str() + ".jpg");
		Mat src_bgr = imread("C:\\Users\\25757\\Documents\\Visual Studio 2013\\Projects\\Project2\\Project2\\AE\\AE_bayer\\ISO_" + iis.str() + ".jpg",0);
		Mat src;
		//cvtColor(src_bgr, src, COLOR_BGR2RGB);

		uchar lum = cal_bayer_peak(src, 1080 * 1920 * 0.1);
		float next_iso = adjust_ISO(src, iso, lum, thred_ISO);
		// Just for jumping out loop when runing this test code.
		if (iso == next_iso)
		{
			cout << "It takes " << frame_count << " frame(s) to be stable !" << endl;
			break;
		}
		frame_count += 1;
		iso = next_iso;
	}
	
}
int main(){

	AG_test();
	return 0;
}
