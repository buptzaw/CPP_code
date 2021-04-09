#include<iostream>
#include<opencv2/opencv.hpp>
#include <functional>
#include <vector>
using namespace std;
using namespace cv;
Mat get_matrix(int matrix_num)
{
	if (matrix_num == 1)//LED0:0
	{
		Mat color_matrix = (Mat_<float>(3, 3) << 1.37911625e+00, 2.28603929e-02, 1.36573547e-01, -3.98607436e-02, 1.43694523e+00, 1.74232788e-01, -1.35643591e-03, -9.51146297e-03, 1.66423218e+00);
		return color_matrix;
	}
	else if (matrix_num == 2)//LED1:1
	{
		Mat color_matrix = (Mat_<float>(3, 3) << 1.31664317, -0.18242148, -0.09962376, 0.86044813, 1.47838428, -0.09915282, 0.26059395, -0.04511439, 0.80458405);
		return color_matrix;
	}
	else
	{
		printf("%s\n", "matrix not found");
		system("pause");
		Mat color_matrix = (Mat_<float>(3, 3) << 0, 0, 0, 0, 0, 0, 0, 0, 0);
		return color_matrix;
	}
}
Mat apply_cmatrix_light(Mat data, Mat color_matrix)
{
	cout << "----------------------------------------------------" << endl;
	cout << "running color correction..." << endl;

	Mat cam2rgb = color_matrix;
	int width = data.rows;
	int height = data.cols;

	Mat cor1 = data.reshape(1, height * width);
	Mat k0 = (Mat_<float>(1, 3) << color_matrix.at<float>(0, 0), color_matrix.at<float>(1, 0), color_matrix.at<float>(2, 0));
	Mat k1 = (Mat_<float>(1, 3) << color_matrix.at<float>(0, 1), color_matrix.at<float>(1, 1), color_matrix.at<float>(2, 1));
	Mat k2 = (Mat_<float>(1, 3) << color_matrix.at<float>(0, 2), color_matrix.at<float>(1, 2), color_matrix.at<float>(2, 2));
	Mat color_corrected_r, color_corrected_g, color_corrected_b;
	filter2D(cor1, color_corrected_r, -1, k0);
	filter2D(cor1, color_corrected_g, -1, k1);
	filter2D(cor1, color_corrected_b, -1, k2);

	Mat color_corrected = cor1;
	for (int i = 0; i < width*height; i++)
	{
		color_corrected.at<uchar>(i, 0) = color_corrected_r.at<uchar>(i, 1);
		color_corrected.at<uchar>(i, 1) = color_corrected_g.at<uchar>(i, 1);
		color_corrected.at<uchar>(i, 2) = color_corrected_b.at<uchar>(i, 1);
	}
	Mat dst = color_corrected.reshape(3, width);
	return dst;
}

Mat light_mode(Mat rgb_img, int mode = 0)
{
	if (mode == 0)
		return rgb_img;
	Mat color_matrix = get_matrix(mode);
	Mat output_image = apply_cmatrix_light(rgb_img, color_matrix);
	return output_image;
}
#pragma endregion

void light()
{
	string image_name = "iphone_0";
	string path = "C:\\Users\\25757\\Documents\\Visual Studio 2013\\Projects\\Project2\\Project2\\test_images\\light\\";
	Mat Scr = imread(path + image_name + ".jpg");

	if (Scr.empty())
	{
		printf("%s\n", "data not found");
		system("pause");
		return;
	}

	Mat output_image;
	cvtColor(Scr, output_image, CV_BGR2RGB);
	int matrix_num = 0;
	output_image = light_mode(Scr, matrix_num);
	Mat data;
	cvtColor(output_image, data, CV_RGB2BGR);
	imshow("corrected_cpp", data);
	waitKey(0);
	imwrite(path + "corrected_cpp.jpg", data);

}
