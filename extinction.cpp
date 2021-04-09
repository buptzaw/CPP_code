#include<iostream>
#include<cmath>
#include<math.h>
#include<opencv2/opencv.hpp>
#include <opencv2/core/core.hpp>
using namespace std;
using namespace cv;
Mat TwoPassLabel(const Mat bwImg)
{
	Mat labImg;
	labImg = bwImg.clone();
	int rows = bwImg.rows;
	int cols = bwImg.cols;

	
	int label = 2;
	vector<int> labelSet;
	labelSet.push_back(0);
	labelSet.push_back(0);

	
	int data_prev = 0;
	int data_cur = 1;

	int left, up;
	int left_up, right_up;
	for (int i = 1; i < rows - 1; i++)
	{
		data_cur = i;
		data_prev = i - 1;

		for (int j = 1; j < cols - 1; j++)
		{

			if (labImg.at<uchar>(data_cur, j) == 0)
				continue;
			left = labImg.at<uchar>(data_cur, j - 1);
			up = labImg.at<uchar>(data_prev, j);
			left_up = labImg.at<uchar>(data_prev, j - 1);
			right_up = labImg.at<uchar>(data_prev, j + 1);

			int count = 0;
			int m = 0;
			int neighborLabels[4] = { 10000,10000,10000,10000 };
			for (int curLabel : {left, up, left_up, right_up})
			{
				if (curLabel > 1)
				{
					neighborLabels[m] = curLabel;
					count = count + 1;
				}
				m = m + 1;
			}
			//cout << count << endl;
			if (!count)//赋予一个新的label
			{
				labelSet.push_back(label);
				//cout << label << endl;
				labImg.at<uchar>(data_cur, j) = label;
				label++;
				continue;
			}
			//
			int smallestLabel = 10000;
			//if (count == 2 && neighborLabels[1] < smallestLabel)
			//	smallestLabel = neighborLabels[1];
			for (int k = 0; k < 4; k++)
			{
				if (neighborLabels[k] < smallestLabel)
				{
					smallestLabel = neighborLabels[k];

				}
			}

			//cout << smallestLabel << endl;
			labImg.at<uchar>(data_cur, j) = smallestLabel;
			//cout << smallestLabel << endl;
			//设置等价表，这里可能有点难理解
			//左点有可能比上点小，也有可能比上点大，两种情况都要考虑,例如
			//0 0 1 0 1 0       x x 2 x 3 x
			//1 1 1 1 1 1  ->   4 4 2 2 2 2
			//要将labelSet中3的位置设置为2

			for (int k = 0; k < 4; k++)
			{
				if (neighborLabels[k] < 10000)
				{
					int neiLabel = neighborLabels[k];
					int oldSmallestLabel = labelSet[neiLabel];
					if (oldSmallestLabel > smallestLabel)
					{
						labelSet[oldSmallestLabel] = smallestLabel;
					}
					else if (oldSmallestLabel < smallestLabel)
						labelSet[smallestLabel] = oldSmallestLabel;
				}
			}

		}

	}
	//上面一步中,有的labelSet的位置还未设为最小值，例如
	//0 0 1 0 1      x x 2 x 3
	//0 1 1 1 1  ->  x 4 2 2 2
	//1 1 1 0 1      5 4 2 x 2
	//上面这波操作中，把labelSet[4]设为2，但labelSet[5]仍为4
	//这里可以将labelSet[5]设为2
	for (size_t i = 2; i < labelSet.size(); i++)
	{
		int curLabel = labelSet[i];
		int prelabel = labelSet[curLabel];
		while (prelabel != curLabel)
		{
			curLabel = prelabel;
			prelabel = labelSet[prelabel];
		}
		labelSet[i] = curLabel;
	}

	//第二次扫描，用labelSet进行更新，最后一列
//data_cur = (int*)labImg.data;
	for (int i = 0; i < rows; i++)
	{
		for (int j = 0; j < cols; j++)
			labImg.at<uchar>(i, j) = labelSet[labImg.at<uchar>(i, j)];
	}


	return labImg;
}
Mat calc_module1_specular_mask(Mat cE, int T)
{
	Mat cE_t;
	threshold(cE, cE_t, T, 255, THRESH_BINARY);//记得换成class后改成img_clip
	return cE_t;
}
Mat filling_image_using_centroid_color(Mat mask, Mat img)
{
	int row = mask.rows;
	int col = mask.cols;
	/*Mat disk_mask2 = (Mat_<uchar>(5, 5) << 0, 0, 1, 0, 0, 0, 1, 1, 1, 0, 1, 1, 1, 1, 1, 0, 1, 1, 1, 0, 0, 0, 1, 0, 0);
	Mat disk_mask4 = (Mat_<uchar>(9, 9) << 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 0, 0, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 0, 0, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0);*/
	Mat disk_mask2 = (Mat_<uchar>(3, 3) << 0, 1, 0, 1, 1, 1, 0, 1, 0);
	Mat disk_mask4 = (Mat_<uchar>(7, 7) << 0, 0, 0, 1, 0, 0, 0, 0, 1, 1, 1, 1, 1, 0, 0, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 0, 0, 1, 1, 1, 1, 1, 0, 0, 0, 0, 1, 0, 0, 0);

	Mat dilated_mask_1;
	Mat dilated_mask_2;
	dilate(mask, dilated_mask_1, disk_mask2);
	dilate(mask, dilated_mask_2, disk_mask4);


	Mat centroid_color_area;
	subtract(dilated_mask_2, dilated_mask_1, centroid_color_area);

	Mat centroid_color_area1;
	threshold(centroid_color_area, centroid_color_area1, 10, 1, THRESH_BINARY);// 二值化
	Mat labeled_area = TwoPassLabel(centroid_color_area1);
	double max = 0.0, min = 0.0;
	minMaxIdx(labeled_area, &min, &max);

	//cv::imshow("labelImg", labeled_area*10);
	//cv::waitKey(0);

	int num_region = max;

	int row_index[1000] = { 0 };
	int col_index[1000] = { 0 };
	int R_mean[1000] = { 0 };
	int G_mean[1000] = { 0 };
	int B_mean[1000] = { 0 };

	Mat r(row, col, CV_16U, Scalar(0));
	Mat c(row, col, CV_16U, Scalar(0));
	for (int i = 0; i < row; i++)
	{
		for (int j = 0; j < col; j++)
		{
			r.at<ushort>(i, j) = i;
			c.at<ushort>(i, j) = j;
		}
	}

	Mat sum_row_mat;
	Mat sum_col_mat;
	Mat sum_R_mat, sum_G_mat, sum_B_mat;
	Mat num_mask(row, col, CV_8U, 0);

	int sum_num_mask, sum_row, sum_col, sum_R, sum_G, sum_B;
	int mean_row, mean_col, mean_R, mean_G, mean_B;

	Mat color_corrected[3];
	cv::split(img, color_corrected);
	Mat B = color_corrected[0];
	Mat G = color_corrected[1];
	Mat R = color_corrected[2];

	int num_region_true = 0;
	for (int i = 2; i <=num_region; i++)
	{
		inRange(labeled_area, i, i, num_mask);


		threshold(num_mask, num_mask, 0, 1, THRESH_BINARY);
		sum_num_mask = cv::sum(num_mask)[0];
		Mat tmp_m, tmp_sd;
		double m = 0;
		num_mask.convertTo(num_mask, CV_16U);
		if (sum_num_mask > 0)
		{
			num_region_true = num_region_true + 1;

			multiply(num_mask, r, sum_row_mat);
			sum_row = cv::sum(sum_row_mat)[0];
			mean_row = sum_row / sum_num_mask;

			multiply(num_mask, c, sum_col_mat);
			sum_col = cv::sum(sum_col_mat)[0];
			mean_col = sum_col / sum_num_mask;


			num_mask.convertTo(num_mask, CV_8U);
			multiply(num_mask, R, sum_R_mat);
			sum_R = cv::sum(sum_R_mat)[0];
			mean_R = sum_R / sum_num_mask;

			multiply(num_mask, G, sum_G_mat);
			sum_G = cv::sum(sum_G_mat)[0];
			mean_G = sum_G / sum_num_mask;

			multiply(num_mask, B, sum_B_mat);
			sum_B = cv::sum(sum_B_mat)[0];
			mean_B = sum_B / sum_num_mask;
			cout << sum_row << endl;



			if (mean_R + mean_G + mean_B > 100)
			{
				row_index[num_region_true] = mean_row;
				col_index[num_region_true] = mean_col;
				R_mean[num_region_true] = mean_R;
				G_mean[num_region_true] = mean_G;
				B_mean[num_region_true] = mean_B;

				cout << num_region_true << endl;
				cout << "mean_row " << mean_row << endl;
				cout << "mean_col " << mean_col << endl;
				cout << "mean_R " << mean_R << endl;
				cout << "mean_G " << mean_G << endl;
				cout << "mean_B " << mean_B << endl;
			}


		}
	}
	Mat filled_img = img;
	for (int i = 0; i < row; i++) {
		for (int j = 0; j < col; j++) {

			if (mask.at<uchar>(i, j) == 255)
			{
				int nearset_region_index = 0;
				int nearset_distance = 10000;

				for (int k = 0; k <= num_region_true; k++)
				{
					//int distance_to_centroid = sqrt((i - row_index[k])*(i - row_index[k]) + (j - col_index[k])*(j - col_index[k]));
					int distance_to_centroid = abs(i - row_index[k]) + abs(j - col_index[k]);

					////如果没有有abs
					//int i_d, j_d;
					//if (i > row_index[k])
					//	i_d = i - row_index[k];
					//else
					//	i_d = row_index[k] - i;
					//if (j > col_index[k])
					//	j_d = j - col_index[k];
					//else
					//	j_d = col_index[k] - j;
					//int distance_to_centroid = i_d + j_d;

					if (distance_to_centroid < nearset_distance)
					{
						nearset_distance = distance_to_centroid;
						nearset_region_index = k;

					}
				}
				filled_img.at<Vec3b>(i, j)[0] = B_mean[nearset_region_index];
				filled_img.at<Vec3b>(i, j)[1] = G_mean[nearset_region_index];
				filled_img.at<Vec3b>(i, j)[2] = R_mean[nearset_region_index];
			}
		}
	}

	return filled_img;

}

double contrast_coeffcient(Mat c,int scale)
{
	Mat tmp_m, tmp_sd;
	double mean_c = 0, std_c = 0;
	meanStdDev(c, tmp_m, tmp_sd);
	mean_c = tmp_m.at<double>(0, 0);
	std_c = tmp_sd.at<double>(0, 0);

	//int scale = 7;
	int t = pow(2.0, scale) / ((mean_c + std_c) / mean_c);
	//cout <<"mean_c " <<  mean_c << endl;
	//cout << "std_c " << std_c << endl;
	return t;
}
Mat calc_modul2_specular_mask(Mat filled_img, float T2_rel, Mat cR, Mat cG, Mat cB)
{
	Mat temp[3];
	cv::split(filled_img, temp);
	Mat cB1 = temp[0];
	Mat cG1 = temp[1];
	Mat cR1 = temp[2];
	Mat fB, fG, fR;
	medianBlur(cR1, fR, 9);
	medianBlur(cG1, fG, 9);
	medianBlur(cB1, fB, 9);

	int scale = 6;

	int tR = contrast_coeffcient(cR, scale);
	int tG = contrast_coeffcient(cG, scale);
	int tB = contrast_coeffcient(cB, scale);
	//cout << "tR " << tR << endl;
	//cout << "tG " << tG << endl;
	//cout << "tB " << tB << endl;
	int row = cR.rows;
	int col = cR.cols;
	Mat max_img = filled_img;
	Mat e_max = cR;
	for (int i = 0; i < row; i++) {
		for (int j = 0; j < col; j++) {
			max_img.at<Vec3b>(i, j)[0] = (tB * cB.at<uchar>(i, j)) / fB.at<uchar>(i, j);
			max_img.at<Vec3b>(i, j)[1] = (tG * cG.at<uchar>(i, j)) / fG.at<uchar>(i, j);
			max_img.at<Vec3b>(i, j)[2] = (tR * cR.at<uchar>(i, j)) / fR.at<uchar>(i, j);
			int MaxVal = 0;
			if (max_img.at<Vec3b>(i, j)[0] > max_img.at<Vec3b>(i, j)[1])
				MaxVal = max_img.at<Vec3b>(i, j)[0];
			else
				MaxVal = max_img.at<Vec3b>(i, j)[1];

			if (max_img.at<Vec3b>(i, j)[2] > MaxVal)
				MaxVal = max_img.at<Vec3b>(i, j)[2];
			e_max.at<uchar>(i, j) = MaxVal;
			//cout << MaxVal << endl;
		}
	}
	//cout << (float)e_max.at<uchar>(361, 71) << endl;
	Mat module2_specular_mask;
	threshold(e_max, module2_specular_mask, T2_rel*pow(2.0, scale), 255, THRESH_BINARY);
	return module2_specular_mask;

}
Mat SpecularDetection(Mat img, int img_clip = 255)
{
	Mat temp[3];
	cv::split(img, temp);
	Mat cB = temp[0];
	Mat cG = temp[1];
	Mat cR = temp[2];
	Mat cE;
	cvtColor(img, cE, CV_BGR2GRAY);
	int T1 = 240;
	int T2_abs = 190;
	float T2_rel = 1.2;
	Mat module1_specular_mask = calc_module1_specular_mask(cE, T1);
	//imshow("module1_specular_mask_c", module1_specular_mask);
	//waitKey(0);
	Mat specular_mask_T2_abs = calc_module1_specular_mask(cE, T2_abs);
	//imshow("specular_mask_T2_abs_c", specular_mask_T2_abs);
	//waitKey(0);
	Mat filled_img = filling_image_using_centroid_color(specular_mask_T2_abs, img);
	//imshow("filled_img_c", filled_img);
	//waitKey(0);

	Mat module2_specular_mask = calc_modul2_specular_mask(filled_img, T2_rel, cR, cG, cB);
	//imshow("module2_specular_mask_c", module2_specular_mask);
	//waitKey(0);
	Mat final_mask;
	bitwise_or(module1_specular_mask, module2_specular_mask, final_mask);
	//imshow("final_mask_c", final_mask);
	//waitKey(0);
	Mat dilate_mask(4, 4, CV_8UC1, 1);
	//Mat dilate_mask = (Mat_<uchar>(3, 3) << 0, 1, 0, 1, 1, 1, 0, 1, 0);
	Mat specular_mask;
	dilate(final_mask, specular_mask, dilate_mask);
	//imshow("specular_mask_c", specular_mask);
	//waitKey(0);
	return specular_mask;

}


Mat Inpaintting(Mat specular_mask, Mat img)
{

	Mat filled_img = filling_image_using_centroid_color(specular_mask, img);
	//imshow("filled_img", filled_img);
	//waitkey(0);
	int sig = 8;
	int window = (3 * sig) * 2 + 1;
	Mat gaussian_filtered_img;
	GaussianBlur(filled_img, gaussian_filtered_img, Size(window, window), sig);
	//imshow("gaussian_filtered_img", gaussian_filtered_img);
	//waitKey(0);
	Mat mx;
	GaussianBlur(specular_mask, mx, Size(11, 11), 5);
	addWeighted(mx, 3, specular_mask, 1, 0, mx);
	//imshow("mx", mx);
	//waitKey(0);
	int row = specular_mask.rows;
	int col = specular_mask.cols;
	Mat inpainted_img = img;
	for (int i = 0; i < row; i++) {
		for (int j = 0; j < col; j++) {
			inpainted_img.at<Vec3b>(i, j)[0] = (mx.at<uchar>(i, j) / 255.0)*gaussian_filtered_img.at<Vec3b>(i, j)[0] + (1 - (mx.at<uchar>(i, j) / 255.0))*img.at<Vec3b>(i, j)[0];//
			inpainted_img.at<Vec3b>(i, j)[1] = (mx.at<uchar>(i, j) / 255.0)*gaussian_filtered_img.at<Vec3b>(i, j)[1] + (1 - (mx.at<uchar>(i, j) / 255.0))*img.at<Vec3b>(i, j)[1];
			inpainted_img.at<Vec3b>(i, j)[2] = (mx.at<uchar>(i, j) / 255.0)*gaussian_filtered_img.at<Vec3b>(i, j)[2] + (1 - (mx.at<uchar>(i, j) / 255.0))*img.at<Vec3b>(i, j)[2];
		}
	}
	//imshow("inpainted_img", inpainted_img);
	//waitKey(0);
	return inpainted_img;
}

int main()
{
	string image_name = "fig5_a.bmp";
	string path = "C:\\Users\\25757\\Documents\\Visual Studio 2013\\Projects\\Project2\\Project2\\test_images\\extinction\\";
	//Mat img = imread(path + image_name);
	Mat img = imread("C:\\Users\\25757\\Documents\\Visual Studio 2013\\Projects\\Project2\\Project2\\test_images\\extinction\\fig5_a.bmp");
	Mat img1 = img.clone();
	if (img.empty())
	{
		//printf("read failed；");
		printf("read failure");
		system("pause");
		//system("pause");
		return 0;
	}
	//imshow("原始图像", img);
	//waitKey(0);
	int clip = 255;
	Mat specular_mask = SpecularDetection(img);
	imshow("specular_mask", specular_mask);
	//waitKey(0);
	Mat inpainted_img = Inpaintting(specular_mask, img1);
	imshow("inpainted_img", inpainted_img);
	//imwrite("D:\\work\\isp-master\\images\\ext\\fig5_a_inpainted_c.bmp", inpainted_img);

	waitKey(0);
	return 0;
}