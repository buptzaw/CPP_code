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
	Mat distortion_coefficientsa = (cv::Mat_<float >(1, 4) << -1.467, 1.036, 0, 0);
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


Mat distortion_correction111_old1(Mat img)
{
	Mat out;

	double k1 = -0.6, k2 = 0.1;//k1,k2 影响很大
	int p1 = 0, p2 = 0;
	short int fx = 1565, fy = 1562, cx = 722, cy = 541;

	cv::Mat image = img;
	int rows = image.rows, cols = image.cols;
	Mat mapx(rows, cols, CV_32F, 0.00);
	Mat mapy(rows, cols, CV_32F, 0.00);


	char N_x = 0;
	int rst_x = xf_Inverse(fx, 16, &N_x);
	int Nint_x = N_x;             //此时已有所有参数，可以算出结果
	cout << "Nint_x是：" << endl;
	cout << Nint_x << endl;
	cout << "rst_x是：" << endl;
	cout << rst_x << endl;
	int output_fx = rst_x >> (Nint_x - 16);//
	cout << "output_fx是：" << endl;//输出是41.87取整
	cout << output_fx << endl;

	char N_y = 0;
	int rst_y = xf_Inverse(fy, 16, &N_y);
	cout << "rst_y是：" << endl;
	cout << rst_y << endl;
	int Nint_y = N_y;
	cout << "Nint_x是：" << endl;
	cout << Nint_x << endl;
	cout << "output_fy是：" << endl;
	int output_fy = rst_y >> (Nint_y - 16);//
	cout << output_fy << endl;

	int x = cx * output_fx;//29602
	cout << x << endl;
	int y = cy * output_fy;//22181
	cout << y << endl;
	//cout << rows << endl;
	//cout << cols << endl;
	if (k1<0)
	{
		for (int v = 0; v < rows; v++) {
			for (int u = 0; u < cols; u++) {
				int x = (u - cx) *output_fx, y = (v - cy) * output_fy;//x,y放大了2的16次方倍
				int r1 = sqrt(x * x + y * y); //r放大了2的16次方
				int r = r1 >> 8;//移位8次之后，还有2的8次方
				cout << "输出变换之后的x是：" << endl;
				cout << x << endl;
				cout << "输出变换之后的y是：" << endl;
				cout << y << endl;
				cout << "输出变换之后的r是：" << endl;
				cout << r << endl;

				//float x_distorted = x * (1 + k1 * r * r + k2 * r * r * r * r) + 2 * p1 * x * y + p2 * (r * r + 2 * x * x);
				//float y_distorted = y * (1 + k1 * r * r + k2 * r * r * r * r) + p1 * (r * r + 2 * y * y) + 2 * p2 * x * y;
				//float u_distorted = fx * x_distorted + cx;
				//float v_distorted = fy * y_distorted + cy;
				int r_2 = ((r * r) >> 16);
				int temp = 1 + r_2 * (k1 + k2 * r_2) + 2 * p1 * x * y + p2 * (r * r + 2 * x * x);
				int temp1 = x * temp;
				int x_distorted = temp1 >> 16;
				// int x_distorted_r = x * (1 + k1 * r * r + k2 * r * r * r * r) + 2 * p1 * x * y + p2 * (r * r + 2 * x * x);
				// cout << " x_distorted_r是："<< endl;
				// cout << x_distorted_r << endl;
				// int x_distorted = x_distorted_r >> 8;
				// cout << " x_distorted是：" << endl;
				//cout << x_distorted << endl;
				// int y_distorted_r = y * (1 + k1 * r * r + k2 * r * r * r * r) + p1 * (r * r + 2 * y * y) + 2 * p2 * x * y;
				int u_distorted = fx * x_distorted + cx;
				//int u_distorted = u_distorted_r >> 8;
				// int r_2 = (r * r) >> 16;
				int temp_y = 1 + r_2 * (k1 + k2 * r_2) + 2 * p1 * x * y + p2 * (r * r + 2 * x * x);
				int temp1_y = y * temp_y;
				int y_distorted = temp1 >> 16;
				//cout << u_distorted << endl;
				// int y_distorted = y_distorted_r >> 8;

				int v_distorted = fy * y_distorted + cy;


				mapx.at<float>(v, u) = u_distorted;//x映射到u
				mapy.at<float>(v, u) = v_distorted;//y映射到v

				//float rx = cx * output_fx;
				//float ry = cy * output_fy;
				int rx = cx * output_fx;//rx放大2的16次方倍
				int ry = cy * output_fy;
				//float div;
				int div = 0;
				if (rx < ry){

					int m = (1 + k1 * rx * ry + k2 * ry * ry * ry * ry);
					char N_m = 0;
					int rst_m = xf_Inverse(fy, 16, &N_m);
					int Nint_m = N_m;
					int div = rst_m >> (Nint_m - 16);

					//div = 1 / (1 + k1 * rx*rx + k2 * rx * rx * rx * rx);
				}
				else{
					int n = (1 + k1 * ry*ry + k2 * ry * ry * ry * ry);
					char N_n = 0;
					int rst_n = xf_Inverse(fy, 16, &N_n);

					int Nint_n = N_n;

					int div = rst_n >> (Nint_n - 16);

					// div = 1 / (1 + k1 * ry*ry + k2 * ry * ry * ry * ry);
				}
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

	return out;
}

//Mat distortion_correction111(Mat img,Mat &mapx_new,Mat &mapy_new)//思考这里究竟是引用还是指针变量；定义函数类型究竟是Mat还还是无返回值void类型
void distortion_correction_q(Mat img, Mat& mapx_new, Mat& mapy_new)//思考这里究竟是引用还是指针变量；定义函数类型究竟是Mat还还是无返回值void类型
{
	//Mat out;


	double k1 = -0.6, k2 = 0.1, p1 = 0.0, p2 = 0.0;              //k1,k2 影响很大
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
	//cout << mapx.cols << endl;//1920
	//cout << mapy.cols << endl;//1920
	cout << mapx.at<float>(1, 1) << endl;
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
	cout << "mapx的最小值和最大值是：" << endl;//41.1198和1602.79
	cout << min << endl;
	cout << max << endl;

	//求出矩阵mapy所包含的数据范围即(min,max)

	cout << mapy.at<float>(1, 1) << endl;
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
	cout << "mapy的最小值和最大值是：" << endl;//-4.73654e-005和1079.52
	cout << min_y << endl;
	cout << max_y << endl;

	//int mapx_1 = (mapx.at<float>(1, 1) * 256);
	//cout << mapx_1 << endl;
	//mapx.at<float>(1, 1) = mapx_1;
	//cout << mapx.at<float>(1, 1) << endl;
	//cout << mapx.at<int>(1, 1) << endl;//?很大且不是我们期待的数字
	//cout << mapy.rows << endl;//1080
	//cout << mapy.cols << endl;//1920

	for (int i = 0; i < rows; i++){
		for (int j = 0; j < cols; j++){
			//mapx.at<float>(i,j) = (mapx.at<float>(i, j)*256*256);
			int mapx_1 = (mapx.at<float>(i, j) * 256);
			//cout << mapx_1 << endl;
			mapx.at<float>(i, j) = mapx_1;
		}
	}


	//int mapx_3 = (mapy.at<float>(1, 1) * 256*256);
	//cout << mapx_3 << endl;
	//mapy.at<float>(1, 1) = mapx_3;
	cout << mapy.at<float>(1, 1) << endl;
	for (int i = 0; i < rows; i++){
		for (int j = 0; j < cols; j++){
			//mapy.at<float>(i, j) = (mapy.at<float>(i, j)*pow(2, 16));
			int mapx_2 = (mapy.at<float>(i, j) * 256 * 256);
			mapy.at<float>(i, j) = mapx_2;
		}
	}
	cout << mapy.at<float>(1, 1) << endl;

	//求出矩阵mapy<int>所包含的数据范围即(min,max)
	//mapx = mapx * 256;
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
	cout << "mapx_int的最小值和最大值是：" << endl;//10526和410315
	cout << min_x_int << endl;
	cout << max_x_int << endl;

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
	cout << "mapy_int的最小值和最大值是：" << endl;//-3和5.92806e+007
	cout << min_y_int << endl;
	cout << max_y_int << endl;

	mapx.convertTo(mapx_new, CV_32S, 1, 0);
	mapy.convertTo(mapy_new, CV_32S, 1, 0);

	//mapx = mapx / 256;
	//mapy = mapy / 65536;
	//mapx = mapx/256;
	//mapy = mapy/256/256;
	//cv::remap(img, out, mapx, mapy, cv::INTER_LINEAR);

	//return out;

}

Mat renew_mapx(Mat X){
	Mat Z;
	for (int i = 0; i < X.rows; i++){
		for (int j = 0; j < X.cols; j++){
			char N = 0;
			int rst = xf_Inverse(X.at<int>(i, j), 32, &N);
			int Nint = N;
			X.at<float>(i, j) = rst >> Nint;
		}
	}
	return X;
}

Mat renew_mapy(Mat Y){
	Mat Z;
	for (int i = 0; i < Y.rows; i++){
		for (int j = 0; j < Y.cols; j++){
			char N = 0;
			int rst = xf_Inverse(Y.at<int>(i, j), 32, &N);
			int Nint = N;
			Z.at<float>(i, j) = rst >> Nint;
		}
	}
	return Z;
}

float sum(Mat& a){
	float sum = 0.0;
	for (int i = 0; i < a.rows; i++){
		for (int j = 0; j < a.cols; j++){
			sum = sum + a.at<int>(i, j);
		}
	}
	return sum;
}

//int test_ldc()
//int main()
int cpoy()
{
	string path = "C:\\Users\\25757\\Documents\\Visual Studio 2013\\Projects\\Project2\\Project2\\10xgain21.png";
	Mat img = imread(path);

	int img_rows = img.rows;
	int img_cols = img.cols;
	cout << img_rows << endl;
	cout << img_cols << endl;

	//cout << "img的（1，1）元素是：" << endl;

	//cout << img.at<float>(1, 1) << endl;

	if (img.empty())
	{
		printf("%s\n", "File not be found!");
		system("pause");
		return 0;
	}
	else{
		//imshow("origin", img);
		//waitKey(0);
		//Mat out_LD = distortion_correction111(img);
		//imwrite("C:\\Users\\25757\\Documents\\Visual Studio 2013\\Projects\\Project2\\Project2\\ldc_new_1.png", out_LD);
		//imshow("after ldc", out_LD);
		Mat mapx_new, mapy_new;
		//Mat out_LD_Quant = distortion_correction111(img,mapx_new,mapy_new);
		distortion_correction_q(img, mapx_new, mapy_new);
		mapx_new.convertTo(mapx_new, CV_32FC1, 1.0 / 256, 0);
		mapy_new.convertTo(mapy_new, CV_32FC1, 1.0 / 65536, 0);
		Mat out;
		Mat mapx_renew, mapy_renew;
		//mapx_renew = renew_mapx(mapx_new);
		//mapy_renew = renew_mapy(mapy_new);//写函数求和求差，别再原来的主函数里写，不容易检查

		//cout << mapx_renew.at<float>(1, 1) << endl;

		//cv::remap(img, out, mapx_renew, mapy_renew, cv::INTER_LINEAR);
		cv::remap(img, out, mapx_new, mapy_new, cv::INTER_LINEAR);

		//Mat a_sum;//求和out矩阵
		//a_sum = sum(out);
		//cout << a_sum << endl;

		Mat out_LD_Quant = out;

		//cout << "out_LD_Quant的（1，1）元素是：" << endl;
		//cout << out_LD_Quant.at<float>(1, 1) << endl;


		//Mat a_sum; //调用函数求和
		//a_sum = sum(out_LD_Quant);
		//cout << a_sum << endl;

		imwrite("C:\\Users\\25757\\Documents\\Visual Studio 2013\\Projects\\Project2\\Project2\\ldc_Quant_1.png", out_LD_Quant);
		imshow("after ldc", out_LD_Quant);

		waitKey(0);
	}
	return 0;
}