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

// RGB luminance method.
uchar cal_luminance_mean(Mat img){
	//input img: RGB
	Mat img_hsv;
	cvtColor(img, img_hsv, COLOR_RGB2HSV);
	Mat hsv[3];
	split(img_hsv, hsv);
	return uchar(mean(hsv[2]).val[0]);
}

uchar cal_luminance_peak(Mat img, int top = 207360){
	//input img: RGB
	Mat gray_img;
	cvtColor(img, gray_img, CV_RGB2GRAY);

	int histSize = 256;
	float range[] = { 0, 256 };
	const float *histRanges = { range };

	Mat hist;
	calcHist(&gray_img, 1, 0, Mat(), hist, 1, &histSize, &histRanges, true, false);
	float total = 0;
	float lum = 0;
	for (int i = 255; i >= 0; i--){
		total += hist.at<float>(i);
		lum += hist.at<float>(i)*i;
		if (total > top) break;
	}

	return uchar(lum / total);

}

uchar cal_luminance_block(Mat img, int index){
	//input img: RGB;  index:0~8
	//return: 8bit nums
	Mat img_hsv;
	cvtColor(img, img_hsv, COLOR_RGB2HSV);
	Mat hsv[3];
	split(img_hsv, hsv);
	int rows = img.size[0];
	int cols = img.size[1];
	int block_h = rows / 3;
	int block_w = cols / 3;
	int position[9][2] = { { 0, 0 }, { 0, 1 }, { 0, 2 },
	{ 1, 0 }, { 1, 1 }, { 1, 2 },
	{ 2, 0 }, { 2, 1 }, { 2, 2 } };
	//boundingbox(block_w*position[index][1], block_h*position[index][0], block_w, block_h)
	Mat block = hsv[2](Rect(block_w*position[index][1], block_h*position[index][0], block_w, block_h));
	return uchar(mean(block).val[0]);
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

void AG_test(){
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
		Mat src_bgr = imread("C:\\Users\\25757\\Documents\\Visual Studio 2013\\Projects\\Project2\\Project2\\AE\\AE\\ISO_" + iis.str() + ".jpg");
		Mat src;
		cvtColor(src_bgr, src, COLOR_BGR2RGB);

		uchar lum = cal_luminance_peak(src, 1080 * 1920 * 0.1);
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
	system("psuse");
	return 0;
}
