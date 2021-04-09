?/*
* 注意：
*  1、copymakeborder在自适应中值滤波、联合双边滤波、NLM中使用，xfopencv可能没有此函数
*  2、引导滤波中使用了CV_64F类型用于浮点乘法，在转为HLS时需要修改
*/
#include <cmath>
#include <iostream>
#include <omp.h>
#include <time.h>
#include <opencv2/opencv.hpp>
using namespace cv;
using namespace std;

//Evaluation index: PSNR
double getPSNR(const Mat& I1, const Mat& I2)
{
    Mat s1;
    absdiff(I1, I2, s1);       // |I1 - I2|
    try {
        s1.convertTo(s1, CV_32F);  // cannot make a square on 8 bits
    }
    catch (exception ex)
    {
        cout << ex.what() << endl;
    }
    //s1.convertTo(s1, CV_32F);  // cannot make a square on 8 bits
    s1 = s1.mul(s1);           // |I1 - I2|^2

    Scalar s = sum(s1);         // sum elements per channel

    double sse = s.val[0] + s.val[1] + s.val[2]; // sum channels

    if (sse <= 1e-10) // for small values return zero
        return 0;
    else
    {
        double  mse = sse / (double)(I1.channels() * I1.total());
        double psnr = 10.0 * log10((255 * 255) / mse);
        return psnr;
    }
}

//盐噪声
void saltNoise(Mat img, int n)
{
    int x, y;
    for (int i = 0; i < n / 2; i++)
    {
        x = std::rand() % img.cols;
        y = std::rand() % img.rows;
        if (img.type() == CV_8UC1)
        {
            img.at<uchar>(y, x) = 255;
        }
        else if (img.type() == CV_8UC3)
        {
            img.at<Vec3b>(y, x)[0] = 255;
            img.at<Vec3b>(y, x)[1] = 255;
            img.at<Vec3b>(y, x)[2] = 255;
        }
    }
}

//椒噪声
void pepperNoise(Mat img, int n)
{
    int x, y;
    for (int i = 0; i < n / 2; i++)
    {
        x = std::rand() % img.cols;
        y = std::rand() % img.rows;
        if (img.type() == CV_8UC1)
        {
            img.at<uchar>(y, x) = 0;
        }
        else if (img.type() == CV_8UC3)
        {
            img.at<Vec3b>(y, x)[0] = 0;
            img.at<Vec3b>(y, x)[1] = 0;
            img.at<Vec3b>(y, x)[2] = 0;
        }
    }
}

// 中值滤波器
void medianFilter(Mat& img, int kernelSize)
{
    medianBlur(img,img,kernelSize);
}

// 自适应中值滤波器
uchar adaptiveMedianFilter(Mat img, int row, int col, int kernelSize, int maxSize)
{
    std::vector<uchar> pixels;
    for (int y = -kernelSize / 2; y <= kernelSize / 2; y++)
    {
        for (int x = -kernelSize / 2; x <= kernelSize / 2; x++)
        {
            pixels.push_back(img.at<uchar>(row + y, col + x));
        }
    }

    sort(pixels.begin(), pixels.end());

    auto min = pixels[0];
    auto max = pixels[kernelSize * kernelSize - 1];
    auto med = pixels[kernelSize * kernelSize / 2];
    auto zxy = img.at<uchar>(row, col);
    if (med > min && med < max)
    {
        // to B
        if (zxy > min && zxy < max)
            return zxy;
        else
            return med;
    }
    else
    {
        kernelSize += 2;
        if (kernelSize <= maxSize)
            return adaptiveMedianFilter(img, row, col, kernelSize, maxSize);// 增大窗口尺寸，继续A过程。
        else
            return med;
    }
}

/*均值滤波*/
Mat meanFilter(Mat src, Size size)
{
    Mat dst;
    blur(src, dst, size);
    return dst;
}

Mat guassianFilter(Mat src, Size size, double sigmaX, double sigmaY)
{
    Mat dst;
    GaussianBlur(src, dst, size, sigmaX, sigmaY);
    return dst;
}
//引导滤波器  
Mat guidedFilter(Mat& srcMat, Mat& guidedMat, int radius, double eps)
{
    //------------【0】转换源图像信息，将输入扩展为64位浮点型，以便以后做乘法------------  
    srcMat.convertTo(srcMat, CV_64FC1);
    guidedMat.convertTo(guidedMat, CV_64FC1);
    //--------------【1】各种均值计算----------------------------------  
    Mat mean_p, mean_I, mean_Ip, mean_II;
    boxFilter(srcMat, mean_p, CV_64FC1, Size(radius, radius));//生成待滤波图像均值mean_p   
    boxFilter(guidedMat, mean_I, CV_64FC1, Size(radius, radius));//生成引导图像均值mean_I     
    boxFilter(srcMat.mul(guidedMat), mean_Ip, CV_64FC1, Size(radius, radius));//生成互相关均值mean_Ip  
    boxFilter(guidedMat.mul(guidedMat), mean_II, CV_64FC1, Size(radius, radius));//生成引导图像自相关均值mean_II  
    //--------------【2】计算相关系数，计算Ip的协方差cov和I的方差var------------------  
    Mat cov_Ip = mean_Ip - mean_I.mul(mean_p);
    Mat var_I = mean_II - mean_I.mul(mean_I);
    //---------------【3】计算参数系数a、b-------------------  
    Mat a = cov_Ip / (var_I + eps);
    Mat b = mean_p - a.mul(mean_I);
    //--------------【4】计算系数a、b的均值-----------------  
    Mat mean_a, mean_b;
    boxFilter(a, mean_a, CV_64FC1, Size(radius, radius));
    boxFilter(b, mean_b, CV_64FC1, Size(radius, radius));
    //---------------【5】生成输出矩阵------------------  
    Mat dstImage = mean_a.mul(srcMat) + mean_b;

    return dstImage;
}


Mat NLM(Mat Y, int h_ = 10, int ds = 5, int Ds = 11, int img_clip = 255)
{
    Y.convertTo(Y, CV_32F);
    int f = ds / 2;
    int t = Ds / 2;
    int img_rows = Y.rows;//height
    int img_cols = Y.cols;//width

    cout << "img_rows " << img_rows << endl;
    cout << "img_cols " << img_cols << endl;

    int padLength = t + f;
    float h = h_ * h_;

    cout << "t " << t << endl;
    cout << "f " << f << endl;
    Mat pad(img_rows + 2 * padLength, img_cols + 2 * padLength, CV_32F, 0.00);
    int a, b;
    for (int i = 0; i < img_rows + 2 * padLength; i++)
    {
        for (int j = 0; j < img_cols + 2 * padLength; j++)
        {

            a = i - padLength;
            b = j - padLength;
            if (i < padLength)
                a = padLength - i - 1;
            if (j < padLength)
                b = padLength - j - 1;
            if (i >= padLength + img_rows)
                a = padLength - i + 2 * img_rows - 1;
            if (j >= padLength + img_cols)
                b = padLength - j + 2 * img_cols - 1;
            pad.at<float>(i, j) = Y.at<float>(a, b);
        }
    }

    cout << "pad " << (float)pad.at<float>(0, 0) << endl;

    cout << "pad " << (float)pad.at<float>(img_rows + 2 * padLength - 1, img_cols + 2 * padLength - 1) << endl;
    Mat Y_1(img_rows + 2 * f, img_cols + 2 * f, CV_32F, 0.00);
    for (int i = padLength - f; i < padLength + f + img_rows; i++)
    {
        for (int j = padLength - f; j < padLength + f + img_cols; j++)
        {

            Y_1.at<float>(i - (padLength - f), j - (padLength - f)) = pad.at<float>(i, j);
        }
    }
    cout << "Y_1 " << (float)Y_1.at<float>(0, 0) << endl;

    Mat kernel(2 * f + 1, 2 * f + 1, CV_32F, 0.00);
    for (int d = 1; d < f + 1; d++)
    {
        for (int m = f - d; m < f + d + 1; m++)//上下
        {
            for (int n = f - d; n < f + d + 1; n++)
            {
                kernel.at<float>(m, n) = kernel.at<float>(m, n) + (1.0 / ((2 * d + 1) * (2 * d + 1)));
            }
        }
    }
    float kernel_sum = sum(kernel)[0];
    for (int m = 0; m < 2 * f + 1; m++)//上下
    {
        for (int n = 0; n < 2 * f + 1; n++)
        {
            kernel.at<float>(m, n) = kernel.at<float>(m, n) / kernel_sum;
        }
    }
    cout << "kernel " << (float)kernel.at<float>(0, 0) << endl;


    Mat average(img_rows, img_cols, CV_32F, 0.00);
    Mat sweight(img_rows, img_cols, CV_32F, 0.00);
    Mat wmax(img_rows, img_cols, CV_32F, 0.00);


    Mat w(img_rows, img_cols, CV_32F, 0.00);
    int c, d;
    Mat Y_2(img_rows + 2 * f, img_cols + 2 * f, CV_32F, 0.00);
    Mat Y_1_2_square(img_rows + 2 * f, img_cols + 2 * f, CV_32F, 0.00);
    Mat Y_1_2_square_16S(img_rows + 2 * f, img_cols + 2 * f, CV_16S, 0.00);
    //Mat Y_filter(img_rows + 2 * f, img_cols + 2 * f, CV_32F, 0.00);
    Mat Y_filter(img_rows + 2 * f, img_cols + 2 * f, CV_32F, 0);
    Mat Y_filter_16S(img_rows + 2 * f, img_cols + 2 * f, CV_16S, 0);
    for (int i = -t; i < t + 1; i++)//上下
    {
        for (int j = -t; j < t + 1; j++)
        {
            //cout << "i j " << i << " " << j << endl;
            if (i == 0 && j == 0)
                continue;
            for (int m = padLength + i - f; m < padLength + i + f + img_rows; m++)
            {
                for (int n = padLength + j - f; n < padLength + j + f + img_cols; n++)
                {
                    Y_2.at<float>(m - (padLength + i - f), n - (padLength + j - f)) = pad.at<float>(m, n);
                }
            }
            //cout << "Y_2 " << (float)Y_2.at<float>(0, 0) << endl;


            for (int m = 0; m < img_rows + 2 * f; m++)//上下
            {
                for (int n = 0; n < img_cols + 2 * f; n++)
                {
                    Y_1_2_square.at<float>(m, n) = (Y_2.at<float>(m, n) - Y_1.at<float>(m, n)) * (Y_2.at<float>(m, n) - Y_1.at<float>(m, n));
                }
            }
            //filter2D(Y_1_2_square, Y_filter, CV_32F, kernel);

            Y_1_2_square.convertTo(Y_1_2_square_16S, CV_16S);
            filter2D(Y_1_2_square_16S, Y_filter_16S, CV_16S, kernel);
            Y_filter_16S.convertTo(Y_filter, CV_32F);

            //cout << "Y_filter " << (float)Y_filter.at<float>(0, 0) << endl;
            //cout << "h " << h << endl;

            for (int m = f; m < f + img_rows; m++)//上下
            {
                for (int n = f; n < f + img_cols; n++)
                {
                    w.at<float>(m - f, n - f) = std::exp(-(Y_filter.at<float>(m, n) / h));
                }
            }
            //cout << "w " << (float)w.at<float>(0, 0) << endl;

            for (int m = 0; m < img_rows; m++)
            {
                for (int n = 0; n < img_cols; n++)
                {
                    sweight.at<float>(m, n) = sweight.at<float>(m, n) + w.at<float>(m, n);
                }
            }
            //cout << "sweight " << (float)sweight.at<float>(0, 0) << endl;
            for (int m = 0; m < img_rows; m++)
            {
                for (int n = 0; n < img_cols; n++)
                {
                    if (wmax.at<float>(m, n) < w.at<float>(m, n))
                        wmax.at<float>(m, n) = w.at<float>(m, n);
                }
            }
            //cout << "wmax " << (float)wmax.at<float>(0, 0) << endl;


            for (int m = 0; m < img_rows; m++)
            {
                for (int n = 0; n < img_cols; n++)
                {
                    average.at<float>(m, n) = average.at<float>(m, n) + (w.at<float>(m, n) * Y_2.at<float>(m + f, n + f));
                }
            }



            //cout << "average " << (float)average.at<float>(0, 0) << endl;
        }
    }

    Mat out(img_rows, img_cols, CV_32F, 0.00);
    for (int m = 0; m < img_rows; m++)
    {
        for (int n = 0; n < img_cols; n++)
        {
            out.at<float>(m, n) = (average.at<float>(m, n) + wmax.at<float>(m, n) * Y.at<float>(m, n)) / (sweight.at<float>(m, n) + wmax.at<float>(m, n));
        }
    }
    //cout << "out " << (float)out.at<float>(0, 0) << endl;
    out.convertTo(out, CV_8U);

    return out;
}

Mat get_weights_table_16U(int table_size, float h, float dist_multiplier, int scale)
{
    //Mat table();
    unsigned short* T = new unsigned short[table_size];
    // Create table.
    for (int i = 0; i < table_size; ++i)
    {
        float temp = std::exp(-(i * dist_multiplier / h)) / pow(2.0, scale) * pow(2.0, 16);
        T[i] = (unsigned short) temp;
        cout << T[i] << ", ";
    }
    Mat lut_data(1, table_size, CV_16UC1, T);

    return lut_data;
}

Mat quant_16S(Mat data, int scale)
{
    // 1bit for sign.
    data = data / pow(2.0, scale - 15);
    data.convertTo(data, CV_16U);
    return data;
}

#include "tools.h"
Mat NLM_16bit(Mat Y, int h_ = 10, int ds = 5, int Ds = 11, int img_clip = 255)
{
    Y.convertTo(Y, CV_16UC1);
    int f = ds / 2;
    int t = Ds / 2;
    int img_rows = Y.rows;//height
    int img_cols = Y.cols;//width

    int padLength = t + f;
    float h = h_ * h_;

    Mat pad(img_rows + 2 * padLength, img_cols + 2 * padLength, CV_16UC1, 0.00);
    int a, b;
    for (int i = 0; i < img_rows + 2 * padLength; i++)
    {
        for (int j = 0; j < img_cols + 2 * padLength; j++)
        {

            a = i - padLength;
            b = j - padLength;
            if (i < padLength)
                a = padLength - i - 1;
            if (j < padLength)
                b = padLength - j - 1;
            if (i >= padLength + img_rows)
                a = padLength - i + 2 * img_rows - 1;
            if (j >= padLength + img_cols)
                b = padLength - j + 2 * img_cols - 1;
            pad.at<unsigned short>(i, j) = Y.at<unsigned short>(a, b);
        }
    }

    // boundingbox(padLength - f, padLength - f, img_rows + 2 * f, img_cols + 2 * f)
    Mat Y_1 = pad(Rect(padLength - f, padLength - f, img_cols + 2 * f, img_rows + 2 * f));

    // Calculate FP32 kernel.
    //Mat kernel(2 * f + 1, 2 * f + 1, CV_32F, 0.00);
    //for (int d = 1; d < f + 1; d++)
    //{
    //    for (int m = f - d; m < f + d + 1; m++)//????
    //    {
    //        for (int n = f - d; n < f + d + 1; n++)
    //        {
    //            kernel.at<float>(m, n) = kernel.at<float>(m, n) + (1.0 / ((2 * d + 1) * (2 * d + 1)));
    //        }
    //    }
    //}
    //float kernel_sum = sum(kernel)[0];
    //kernel = kernel / kernel_sum;
    
    // FP32 -> Q(1+kernel_scale).(15-kernel_scale)
    int kernel_scale = 1;
    //kernel = quant_16S(kernel, kernel_scale);

    // Save the quantized kernel from quant_16S().
    Mat kernel = (Mat_<short>(2 * f + 1, 2 * f + 1) << 655, 655, 655, 655, 655, 655, 2476, 2476, 2476, 655, 655, 2476, 2476, 2476, 655, 655, 2476, 2476, 2476, 655, 655, 655, 655, 655, 655);
    
    // Q(1+kernel_scale).(15-kernel_scale) --> FP32.
    kernel.convertTo(kernel, CV_32FC1, 1 / pow(2.0, 16 - kernel_scale));

    Mat average(img_rows, img_cols, CV_16UC1, Scalar(0));
    Mat sweight(img_rows, img_cols, CV_16UC1, Scalar(0));

    Mat w(img_rows, img_cols, CV_16UC1, Scalar(0));
    Mat Y_2(img_rows + 2 * f, img_cols + 2 * f, CV_16UC1, Scalar(0));
    Mat Y_filter_16(img_rows + 2 * f, img_cols + 2 * f, CV_16UC1, Scalar(0));
    Mat Y_1_2_square_16(img_rows + 2 * f, img_cols + 2 * f, CV_16UC1, Scalar(0));

    int table_size = 2048;
    int w_scale = 7;
    // Pre calculate weights: exp(-x/h) and quantize to Qw_scale.(16-w_scale).
    //Mat weights_table = get_weights_table_16U(table_size, h*4, 1.28, w_scale);

    // save results from get_weights_table_16U().
    // Q(w_scale).(16-w_scale)
    Mat weights_table = (Mat_<unsigned short>(1, table_size) << 512, 510, 508, 507, 505, 503, 502, 500, 499, 497, 495, 494, 492, 491, 489, 488, 486, 484, 483, 481, 480, 478, 477, 475, 474, 472, 471, 469, 468, 466, 465, 463, 462, 460, 459, 457, 456, 454, 453, 451, 450, 449, 447, 446, 444, 443, 441, 440, 439, 437, 436, 434, 433, 432, 430, 429, 428, 426, 425, 423, 422, 421, 419, 418, 417, 415, 414, 413, 411, 410, 409, 407, 406, 405, 404, 402, 401, 400, 398, 397, 396, 395, 393, 392, 391, 390, 388, 387, 386, 385, 383, 382, 381, 380, 378, 377, 376, 375, 374, 372, 371, 370, 369, 368, 367, 365, 364, 363, 362, 361, 360, 358, 357, 356, 355, 354, 353, 352, 350, 349, 348, 347, 346, 345, 344, 343, 342, 341, 339, 338, 337, 336, 335, 334, 333, 332, 331, 330, 329, 328, 327, 326, 325, 323, 322, 321, 320, 319, 318, 317, 316, 315, 314, 313, 312, 311, 310, 309, 308, 307, 306, 305, 304, 303, 302, 301, 301, 300, 299, 298, 297, 296, 295, 294, 293, 292, 291, 290, 289, 288, 287, 286, 285, 285, 284, 283, 282, 281, 280, 279, 278, 277, 276, 276, 275, 274, 273, 272, 271, 270, 269, 269, 268, 267, 266, 265, 264, 263, 263, 262, 261, 260, 259, 258, 258, 257, 256, 255, 254, 254, 253, 252, 251, 250, 250, 249, 248, 247, 246, 246, 245, 244, 243, 242, 242, 241, 240, 239, 239, 238, 237, 236, 236, 235, 234, 233, 233, 232, 231, 230, 230, 229, 228, 227, 227, 226, 225, 224, 224, 223, 222, 222, 221, 220, 219, 219, 218, 217, 217, 216, 215, 215, 214, 213, 213, 212, 211, 211, 210, 209, 208, 208, 207, 207, 206, 205, 205, 204, 203, 203, 202, 201, 201, 200, 199, 199, 198, 197, 197, 196, 196, 195, 194, 194, 193, 192, 192, 191, 191, 190, 189, 189, 188, 188, 187, 186, 186, 185, 185, 184, 183, 183, 182, 182, 181, 180, 180, 179, 179, 178, 178, 177, 176, 176, 175, 175, 174, 174, 173, 173, 172, 171, 171, 170, 170, 169, 169, 168, 168, 167, 167, 166, 165, 165, 164, 164, 163, 163, 162, 162, 161, 161, 160, 160, 159, 159, 158, 158, 157, 157, 156, 156, 155, 155, 154, 154, 153, 153, 152, 152, 151, 151, 150, 150, 149, 149, 148, 148, 147, 147, 146, 146, 146, 145, 145, 144, 144, 143, 143, 142, 142, 141, 141, 140, 140, 140, 139, 139, 138, 138, 137, 137, 136, 136, 136, 135, 135, 134, 134, 133, 133, 133, 132, 132, 131, 131, 130, 130, 130, 129, 129, 128, 128, 128, 127, 127, 126, 126, 126, 125, 125, 124, 124, 124, 123, 123, 122, 122, 122, 121, 121, 120, 120, 120, 119, 119, 119, 118, 118, 117, 117, 117, 116, 116, 115, 115, 115, 114, 114, 114, 113, 113, 113, 112, 112, 111, 111, 111, 110, 110, 110, 109, 109, 109, 108, 108, 108, 107, 107, 107, 106, 106, 106, 105, 105, 105, 104, 104, 104, 103, 103, 103, 102, 102, 102, 101, 101, 101, 100, 100, 100, 99, 99, 99, 98, 98, 98, 97, 97, 97, 96, 96, 96, 96, 95, 95, 95, 94, 94, 94, 93, 93, 93, 93, 92, 92, 92, 91, 91, 91, 90, 90, 90, 90, 89, 89, 89, 88, 88, 88, 88, 87, 87, 87, 86, 86, 86, 86, 85, 85, 85, 85, 84, 84, 84, 83, 83, 83, 83, 82, 82, 82, 82, 81, 81, 81, 81, 80, 80, 80, 80, 79, 79, 79, 79, 78, 78, 78, 78, 77, 77, 77, 77, 76, 76, 76, 76, 75, 75, 75, 75, 74, 74, 74, 74, 73, 73, 73, 73, 72, 72, 72, 72, 72, 71, 71, 71, 71, 70, 70, 70, 70, 69, 69, 69, 69, 69, 68, 68, 68, 68, 67, 67, 67, 67, 67, 66, 66, 66, 66, 66, 65, 65, 65, 65, 64, 64, 64, 64, 64, 63, 63, 63, 63, 63, 62, 62, 62, 62, 62, 61, 61, 61, 61, 61, 60, 60, 60, 60, 60, 59, 59, 59, 59, 59, 59, 58, 58, 58, 58, 58, 57, 57, 57, 57, 57, 57, 56, 56, 56, 56, 56, 55, 55, 55, 55, 55, 55, 54, 54, 54, 54, 54, 53, 53, 53, 53, 53, 53, 52, 52, 52, 52, 52, 52, 51, 51, 51, 51, 51, 51, 50, 50, 50, 50, 50, 50, 49, 49, 49, 49, 49, 49, 49, 48, 48, 48, 48, 48, 48, 47, 47, 47, 47, 47, 47, 47, 46, 46, 46, 46, 46, 46, 46, 45, 45, 45, 45, 45, 45, 44, 44, 44, 44, 44, 44, 44, 43, 43, 43, 43, 43, 43, 43, 43, 42, 42, 42, 42, 42, 42, 42, 41, 41, 41, 41, 41, 41, 41, 40, 40, 40, 40, 40, 40, 40, 40, 39, 39, 39, 39, 39, 39, 39, 39, 38, 38, 38, 38, 38, 38, 38, 38, 37, 37, 37, 37, 37, 37, 37, 37, 37, 36, 36, 36, 36, 36, 36, 36, 36, 35, 35, 35, 35, 35, 35, 35, 35, 35, 34, 34, 34, 34, 34, 34, 34, 34, 34, 33, 33, 33, 33, 33, 33, 33, 33, 33, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 31, 31, 31, 31, 31, 31, 31, 31, 31, 31, 30, 30, 30, 30, 30, 30, 30, 30, 30, 30, 29, 29, 29, 29, 29, 29, 29, 29, 29, 29, 29, 28, 28, 28, 28, 28, 28, 28, 28, 28, 28, 28, 27, 27, 27, 27, 27, 27, 27, 27, 27, 27, 27, 26, 26, 26, 26, 26, 26, 26, 26, 26, 26, 26, 26, 25, 25, 25, 25, 25, 25, 25, 25, 25, 25, 25, 25, 24, 24, 24, 24, 24, 24, 24, 24, 24, 24, 24, 24, 24, 23, 23, 23, 23, 23, 23, 23, 23, 23, 23, 23, 23, 23, 22, 22, 22, 22, 22, 22, 22, 22, 22, 22, 22, 22, 22, 22, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 19, 19, 19, 19, 19, 19, 19, 19, 19, 19, 19, 19, 19, 19, 19, 19, 18, 18, 18, 18, 18, 18, 18, 18, 18, 18, 18, 18, 18, 18, 18, 18, 18, 17, 17, 17, 17, 17, 17, 17, 17, 17, 17, 17, 17, 17, 17, 17, 17, 17, 17, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 14, 14, 14, 14, 14, 14, 14, 14, 14, 14, 14, 14, 14, 14, 14, 14, 14, 14, 14, 14, 14, 13, 13, 13, 13, 13, 13, 13, 13, 13, 13, 13, 13, 13, 13, 13, 13, 13, 13, 13, 13, 13, 13, 13, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0);
    
    unsigned short* exp_table = (unsigned short*)weights_table.data;

    for (int i = -t; i < t + 1; i++)
    {
        for (int j = -t; j < t + 1; j++)
        {
            // boundingbox(padLength + i - f, padLength + j - f, img_cols + 2 * f, img_rows + 2 * f)
            Mat Y_2 = pad(Rect(padLength + i - f, padLength + j - f, img_cols + 2 * f, img_rows + 2 * f));

            Mat temp;
            absdiff(Y_2, Y_1, temp);
            multiply(temp, temp, Y_1_2_square_16, 1.0, CV_16UC1);

            filter2D(Y_1_2_square_16, Y_filter_16, CV_16UC1, kernel);
            min(Y_filter_16, table_size - 1, Y_filter_16);
            for (int m = f; m < f + img_rows; m++)
            {
                for (int n = f; n < f + img_cols; n++)
                {
                    unsigned short v = exp_table[Y_filter_16.at<unsigned short>(m, n)];
                    w.at<unsigned short>(m - f, n - f) = v;
                }
            }

            add(w, sweight, sweight);

            //boundingbox(f, f, img_cols, img_rows)
            Rect rect(f, f, img_cols, img_rows);
            // w: Q(w_scale).(16-w_scale)
            // Y_2: Q16.0
            // out: Q16.0
            multiply(w, Y_2(rect), temp, 1.0 / pow(2.0, 16 - w_scale), CV_16UC1);
            add(average, temp, average);
        }
    }

    // 1 / sweight
    Mat sweight_inversed(sweight.size(), CV_16UC1);
    int inverse_scale = 5;
    for (int i = 0; i < img_rows; ++i)
    {
        for (int j = 0; j < img_cols; ++j)
        {
            char N = 0;
            unsigned int inv = xf_Inverse(sweight.at<unsigned short>(i, j), w_scale, &N);
            // Q(inverse_scale).(16 - inverse_scale)
            int shift = 16 - inverse_scale - int(N);
            if (shift > 0)
            {
                inv = inv << shift;
            }
            else
            {
                inv = inv >> (-shift);
            }
            sweight_inversed.at<unsigned short>(i, j) = unsigned short(inv);
        }
    }
    Mat out(img_rows, img_cols, CV_16UC1, Scalar(0));
    // sweight_inversed:    Qinverse_scale.(16-inverse_scale)
    // average:             Q16.0
    // out:                 Q16.0
    multiply(average, sweight_inversed, out, 1.0 / pow(2.0, 16 - inverse_scale), CV_16UC1);
    out.convertTo(out, CV_8U);

    return out;
}

void jointBilateralFilter(const Mat& src, Mat& dst, int d, double sigma_color, double sigma_space, Mat& joint, int borderType)
{
    Size size = src.size();
    if (dst.empty())
        dst = Mat::zeros(src.size(), src.type());

    CV_Assert(
        (src.type() == CV_8UC1 || src.type() == CV_8UC3)
        && src.type() == dst.type() && src.size() == dst.size()
        && src.data != dst.data);
    if (sigma_color <= 0)
        sigma_color = 1;
    if (sigma_space <= 0)
        sigma_space = 1;

    double gauss_color_coeff = -0.5 / (sigma_color * sigma_color);
    double gauss_space_coeff = -0.5 / (sigma_space * sigma_space);

    if (joint.empty())
        src.copyTo(joint);

    const int cn = src.channels();
    const int cnj = joint.channels();

    int radius;
    if (d <= 0)
        radius = cvRound(sigma_space * 1.5);	// 根据 sigma_space 计算 radius
    else
        radius = d / 2;
    radius = MAX(radius, 1);
    d = radius * 2 + 1;	// 重新计算 像素“矩形”邻域的直径d，确保是奇数

    // 扩展 src 和 joint 长宽各2*radius
    Mat jim;
    Mat sim;
    copyMakeBorder(joint, jim, radius, radius, radius, radius, borderType);
    copyMakeBorder(src, sim, radius, radius, radius, radius, borderType);

    // cnj: joint的通道数
    vector<float> _color_weight(cnj * 256);
    vector<float> _space_weight(d * d);	 // (2*radius + 1)^2
    vector<int> _space_ofs_jnt(d * d);
    vector<int> _space_ofs_src(d * d);
    float* color_weight = &_color_weight[0];
    float* space_weight = &_space_weight[0];
    int* space_ofs_jnt = &_space_ofs_jnt[0];
    int* space_ofs_src = &_space_ofs_src[0];

    // initialize color-related bilateral filter coefficients
    // 色差的高斯权重
    for (int i = 0; i < 256 * cnj; i++)
        color_weight[i] = (float)std::exp(i * i * gauss_color_coeff);

    int maxk = 0;	// 0 - (2*radius + 1)^2
    // initialize space-related bilateral filter coefficients
    for (int i = -radius; i <= radius; i++)
    {
        for (int j = -radius; j <= radius; j++)
        {
            double r = std::sqrt((double)i * i + (double)j * j);
            if (r > radius)
                continue;
            space_weight[maxk] = (float)std::exp(r * r * gauss_space_coeff);
            space_ofs_jnt[maxk] = (int)(i * jim.step + j * cnj);			// joint 邻域内的相对坐标 (i, j)【偏移量】， 左上角为(-radius, -radius)，右下角为(radius, radius)
            space_ofs_src[maxk++] = (int)(i * sim.step + j * cn);		// src 邻域内的相对坐标 (i, j)
        }
    }
#pragma omp parallel for
    for (int i = 0; i < size.height; i++)
    {
        const uchar* jptr = jim.data + (i + radius) * jim.step + radius * cnj;	// &jim.ptr(i+radius)[radius]
        const uchar* sptr = sim.data + (i + radius) * sim.step + radius * cn; // &sim.ptr(i+radius)[radius]
        uchar* dptr = dst.data + i * dst.step;												// dst.ptr(i)

        // src 和 joint 通道数不同的四种情况
        if (cn == 1 && cnj == 1)
        {
            for (int j = 0; j < size.width; j++)
            {
                float sum = 0, wsum = 0;
                int val0 = jptr[j];	// jim.ptr(i + radius)[j + radius]

                for (int k = 0; k < maxk; k++)
                {
                    int val = jptr[j + space_ofs_src[k]];	 // jim.ptr(i + radius + offset_x)[j + radius + offset_y]
                    int val2 = sptr[j + space_ofs_src[k]];	// sim.ptr(i + radius + offset_x)[j + radius + offset_y]

                    // 根据joint当前像素和邻域像素的 距离权重 和 色差权重，计算综合的权重
                    float w = space_weight[k]
                        * color_weight[std::abs(val - val0)];
                    sum += val2 * w;	// 统计 src 邻域内的像素带权和
                    wsum += w;			// 统计权重和
                }
                // overflow is not possible here => there is no need to use CV_CAST_8U
                // 归一化 src 邻域内的像素带权和，并赋给 dst对应的像素
                dptr[j] = (uchar)cvRound(sum / wsum);
            }
        }
        else if (cn == 3 && cnj == 3)
        {
            for (int j = 0; j < size.width * 3; j += 3)
            {
                float sum_b = 0, sum_g = 0, sum_r = 0, wsum = 0;
                int b0 = jptr[j], g0 = jptr[j + 1], r0 = jptr[j + 2];	// jim.ptr(i + radius)[j + radius][0...2]
                for (int k = 0; k < maxk; k++)
                {
                    const uchar* sptr_k = jptr + j + space_ofs_src[k];
                    const uchar* sptr_k2 = sptr + j + space_ofs_src[k];

                    int b = sptr_k[0], g = sptr_k[1], r = sptr_k[2];	 // jim.ptr(i + radius + offset_x)[j + radius + offset_y][0...2]
                    float w = space_weight[k]
                        * color_weight[std::abs(b - b0) + std::abs(g - g0)
                        + std::abs(r - r0)];
                    sum_b += sptr_k2[0] * w;	// sim.ptr(i + radius + offset_x)[j + radius + offset_y][0...2]
                    sum_g += sptr_k2[1] * w;
                    sum_r += sptr_k2[2] * w;
                    wsum += w;
                }
                wsum = 1.f / wsum;
                b0 = cvRound(sum_b * wsum);
                g0 = cvRound(sum_g * wsum);
                r0 = cvRound(sum_r * wsum);
                dptr[j] = (uchar)b0;
                dptr[j + 1] = (uchar)g0;
                dptr[j + 2] = (uchar)r0;
            }
        }
        else if (cn == 1 && cnj == 3)
        {
            for (int j = 0, l = 0; j < size.width * 3; j += 3, l++)
            {
                float sum_b = 0, wsum = 0;

                int b0 = jptr[j], g0 = jptr[j + 1], r0 = jptr[j + 2];	// jim.ptr(i + radius)[j + radius][0...2]
                for (int k = 0; k < maxk; k++)
                {
                    int val = *(sptr + l + space_ofs_src[k]);	// sim.ptr(i + radius + offset_x)[l + radius + offset_y]

                    const uchar* sptr_k = jptr + j + space_ofs_jnt[k];
                    int b = sptr_k[0], g = sptr_k[1], r = sptr_k[2];		// jim.ptr(i + radius + offset_x)[j + radius + offset_y][0...2]

                    float w = space_weight[k]
                        * color_weight[std::abs(b - b0) + std::abs(g - g0)
                        + std::abs(r - r0)];
                    sum_b += val * w;
                    wsum += w;
                }
                wsum = 1.f / wsum;
                b0 = cvRound(sum_b * wsum);
                dptr[l] = (uchar)b0;
            }
        }
        else if (cn == 3 && cnj == 1)
        {
            for (int j = 0, l = 0; j < size.width * 3; j += 3, l++)
            {
                float sum_b = 0, sum_g = 0, sum_r = 0, wsum = 0;
                int val0 = jptr[l];	// jim.ptr(i + radius)[l + radius]
                for (int k = 0; k < maxk; k++)
                {
                    int val = jptr[l + space_ofs_jnt[k]];	// jim.ptr(i + radius + offset_x)[l + radius + offset_y]

                    const uchar* sptr_k = sptr + j + space_ofs_src[k];		// sim.ptr(i + radius + offset_x)[j + radius + offset_y] 

                    float w = space_weight[k]
                        * color_weight[std::abs(val - val0)];

                    sum_b += sptr_k[0] * w;	// sim.ptr(i + radius + offset_x)[j + radius + offset_y] [0...2]
                    sum_g += sptr_k[1] * w;
                    sum_r += sptr_k[2] * w;
                    wsum += w;
                }

                // overflow is not possible here => there is no need to use CV_CAST_8U
                wsum = 1.f / wsum;
                dptr[j] = (uchar)cvRound(sum_b * wsum);
                dptr[j + 1] = (uchar)cvRound(sum_g * wsum);
                dptr[j + 2] = (uchar)cvRound(sum_r * wsum);
            }
        }
    }
}

Mat BF(Mat data, int d, int sigma_color, int sigma_space, bool isEdge)
{
    Mat dst = Mat();
    if (isEdge)
    {
        Mat edge;
        /*注：ddepth = -1时，代表输出图像与输入图像相同的深度。
            int dx：int类型dx，x 方向上的差分阶数，1或0
            int dy：int类型dy，y 方向上的差分阶数，1或0
            其中，dx = 1，dy = 0，表示计算X方向的导数，检测出的是垂直方向上的边缘；dx = 0，dy = 1，表示计算Y方向的导数，检测出的是水平方向上的边缘。
            int ksize：为进行边缘检测时的模板大小为ksize * ksize，取值为1、3、5和7，其中默认值为3。特殊情况：ksize = 1时，采用的模板为3 * 1或1 * 3。
            当ksize = 3时，Sobel内核可能产生比较明显的误差，此时，可以使用 Scharr 函数，该函数仅作用于大小为3的内核。具有跟sobel一样的速度，但结果更精确，其内核为：*/
        Sobel(data, edge, -1, 1, 1, 3, CV_SCHARR, 0, 4);
        jointBilateralFilter(data, dst, d, sigma_color, sigma_space, edge, BORDER_REFLECT);
    }
    else
    {
        bilateralFilter(data, dst, d, sigma_color, sigma_space);
    }
    return dst;
}

//void chroma_denoising(Mat& U, Mat& V, Size size = Size(7, 7), double sigmaX = 40.0, double sigmaY = 40.0)
void chroma_denoising(Mat& U,Mat& V, Size size = Size(7, 7), double sigmaX = 40.0, double sigmaY=40.0)
{
    GaussianBlur(U, U, size, sigmaX, sigmaY);
    GaussianBlur(V, V, size, sigmaX, sigmaY);
}




//测试代码
void mean_test()
{
    clock_t start, end;

   // Mat img = imread("../test_data/nr/24.jpg", IMREAD_GRAYSCALE);
	Mat img = imread("C:\\Users\\25757\\Documents\\Visual Studio 2013\\Projects\\Project2\\Project2\\ct-awb_bpc_nr\\24.jpg", IMREAD_GRAYSCALE);

    Mat src = imread("C:\\Users\\25757\\Documents\\Visual Studio 2013\\Projects\\Project2\\Project2\\ct-awb_bpc_nr\\noisy.png",IMREAD_GRAYSCALE);
    if (!src.data)
    {
        cout << "读取图片错误，请重新输入正确路径！\n";
        return;
    }
    start = clock();
    Mat dst = meanFilter(src,Size(3,3));
    end = clock();
    cout << "耗时：" << (end - start) / 1000.0 << " s"<< endl;
    cout << "noisy PSNR:" << getPSNR(img, src) << endl;
    cout << "denoised PSNR:" << getPSNR(img, dst) << endl;
    imshow("噪声图", src);
    imshow("均值滤波降噪图", dst);
    imwrite("C:\\Users\\25757\\Documents\\Visual Studio 2013\\Projects\\Project2\\Project2\\ct-awb_bpc_nr\\denoisedMean.png", dst);
    waitKey(0);
}

void guassian_test()
{
    clock_t start, end;
    Mat img = imread("C:\\Users\\25757\\Documents\\Visual Studio 2013\\Projects\\Project2\\Project2\\ct-awb_bpc_nr\\24.jpg", IMREAD_GRAYSCALE);
    Mat src = imread("C:\\Users\\25757\\Documents\\Visual Studio 2013\\Projects\\Project2\\Project2\\ct-awb_bpc_nr\\noisy.png", IMREAD_GRAYSCALE);
    if (!src.data)
    {
        cout << "读取图片错误，请重新输入正确路径！\n";
        system("pause");
        return;
    }
    start = clock();
    Mat dst = guassianFilter(src, Size(3, 3),0.65,0.65);
    end = clock();
    cout << "耗时：" << (end - start) / 1000.0 << " s" << endl;
    cout << "noisy PSNR:" << getPSNR(img, src) << endl;
    cout << "denoised PSNR:" << getPSNR(img, dst) << endl;
    imshow("噪声图", src);
    imshow("高斯滤波降噪图", dst);
    imwrite("C:\\Users\\25757\\Documents\\Visual Studio 2013\\Projects\\Project2\\Project2\\ct-awb_bpc_nr\\denoisedGuassian.png", dst);
    waitKey(0);
}

void guide_test()
{
 
    clock_t start, end;
    Mat img = imread("C:\\Users\\25757\\Documents\\Visual Studio 2013\\Projects\\Project2\\Project2\\ct-awb_bpc_nr\\24.jpg", IMREAD_GRAYSCALE);
    Mat srcImage = imread("C:\\Users\\25757\\Documents\\Visual Studio 2013\\Projects\\Project2\\Project2\\ct-awb_bpc_nr\\noisy.png",IMREAD_GRAYSCALE);
    if (!srcImage.data)
    {
        cout << "读取图片错误，请重新输入正确路径！\n";
        system("pause");
        return;
    }
       
    Mat tempImage;
    srcImage.convertTo(tempImage, CV_64FC1, 1.0 / 255.0);//将分通道转换成浮点型数据  
    Mat cloneImage = tempImage.clone(); //将tempImage复制一份到cloneImage  
    start = clock();
    Mat resultImage = guidedFilter(tempImage, cloneImage,5,0.04);//对分通道分别进行导向滤波，半径为1、3、5...等奇数  
    end = clock();
    Mat dstImage;
    normalize(resultImage, dstImage, 0x00, 0xFF, cv::NORM_MINMAX, CV_8U);
    //cout << dstImage << endl;
    cout << "耗时：" << (end - start) / 1000.0 << " s"<< endl;
    cout << "noisy PSNR:" << getPSNR(img, srcImage) << endl;
    cout << "denoised PSNR:" << getPSNR(img, dstImage) << endl;
    imshow("【源图像】", srcImage);
    imwrite("C:\\Users\\25757\\Documents\\Visual Studio 2013\\Projects\\Project2\\Project2\\ct-awb_bpc_nr\\denoisedGuide.png",dstImage);
    imshow("【引导滤波/导向滤波】", dstImage);
    waitKey(0);
}

void median_test()
{
    clock_t start, end;
    Mat img = imread("C:\\Users\\25757\\Documents\\Visual Studio 2013\\Projects\\Project2\\Project2\\ct-awb_bpc_nr\\24.jpg", IMREAD_GRAYSCALE);
    int minSize = 3;
    int maxSize = 7;
    Mat src;
    src = imread("C:\\Users\\25757\\Documents\\Visual Studio 2013\\Projects\\Project2\\Project2\\ct-awb_bpc_nr\\24.jpg",0);
    ////    cvtColor(img, img, COLOR_BGR2GRAY);
    //imshow("src", img);
    saltNoise(src, 200000);
    pepperNoise(src, 200000);
    imwrite("C:\\Users\\25757\\Documents\\Visual Studio 2013\\Projects\\Project2\\Project2\\ct-awb_bpc_nr\\saltpepperNoisy.png",src);
    Mat temp = src.clone();
    start = clock();
    copyMakeBorder(temp, temp, maxSize / 2, maxSize / 2, maxSize / 2, maxSize / 2, BORDER_REFLECT);

    //自适应中值滤波测试
    for (int j = maxSize / 2; j < temp.rows - maxSize / 2; j++)
    {
        for (int i = maxSize / 2; i < temp.cols - maxSize / 2; i++)
        {
            temp.at<uchar>(j, i) = adaptiveMedianFilter(temp, j, i, minSize, maxSize);            
        }
    }
    cout << temp.size << endl;
    end = clock();
    imwrite("C:\\Users\\25757\\Documents\\Visual Studio 2013\\Projects\\Project2\\Project2\\ct-awb_bpc_nr\\denoisedAdaptMedium.png", temp);
    imshow("sdapt_color_dst", temp);
    cout << "耗时：" << (end - start) / 1000.0 << " s"<< endl;
    cout << "noisy PSNR:" << getPSNR(img, src) << endl;
    copyMakeBorder(img, img, maxSize / 2, maxSize / 2, maxSize / 2, maxSize / 2, BORDER_REFLECT);
    cout << "denoised PSNR:" << getPSNR(img, temp) << endl;

    waitKey(0);

    // 中值滤波
    Mat img2, media_dst = src.clone();
    start = clock();
    medianFilter(media_dst, 3);
    end = clock();
    imwrite("C:\\Users\\25757\\Documents\\Visual Studio 2013\\Projects\\Project2\\Project2\\ct-awb_bpc_nr\\denoisedMedium.png", media_dst);
    cout << "耗时：" << (end - start) / 1000.0 << " s" << endl;
    cout << "noisy PSNR:" << getPSNR(img, src) << endl;
    cout << "denoised PSNR:" << getPSNR(img, media_dst) << endl;
    imshow("media_dst", media_dst);
    waitKey(0);
    destroyAllWindows();

}

void bilateral_test()
{
    clock_t start, end;
    Mat img = imread("C:\\Users\\25757\\Documents\\Visual Studio 2013\\Projects\\Project2\\Project2\\ct-awb_bpc_nr\\24.jpg", IMREAD_GRAYSCALE);

    Mat src = imread("C:\\Users\\25757\\Documents\\Visual Studio 2013\\Projects\\Project2\\Project2\\ct-awb_bpc_nr\\noisy.png",IMREAD_GRAYSCALE);
    double max = 0.0, min = 0.0;
    minMaxIdx(src, &min, &max);
    cout << max <<endl;

    //bilateral test
    start = clock();
    Mat dst = BF(src, -1, 60,60,false);
    end = clock();
    cout << "Bilateral 耗时：" << (end - start) << "ms" << endl;
    cout << "noisy PSNR:" << getPSNR(img, src) << endl;
    cout << "denoised PSNR:" << getPSNR(img, dst) << endl;
    //imwrite("bilateral.png", dst);
    imshow("src",src);
    imshow("dst",dst);
    waitKey(0);
}

void test_NLM()
{
    string image_name = "noise3";
    //string path = "D:\\project\\endoscope\\test\\rs_0_100\\rs-50.jpg";
    string path = "C:\\Users\\25757\\Documents\\Visual Studio 2013\\Projects\\Project2\\Project2\\ct-awb_bpc_nr\\noise3.jpg";
    Mat img = imread(path);
    if (img.empty())
    {
        printf("%s\n", "读入图片失败；");
        system("pause");
        return;
    }
    imshow("原始图像", img);

   // waitKey(0);

    Mat gray_img;
    Mat yuv_img;
    cvtColor(img, yuv_img, CV_BGR2YUV);
    Mat color_corrected[3];
    cv::split(yuv_img, color_corrected);
    Mat Y = color_corrected[0];//修改灰度图像


    int clip = 255;
    int ds = 1;
    int Ds = 2;
    int h = 10;

    Mat out_yuv;

    Mat out_Y, out_Y_16bit, out_Y_opencv;
    Mat color_U_V[2];


    out_Y_16bit = NLM_16bit(Y);
    color_corrected[0] = out_Y_16bit;
    merge(color_corrected, 3, out_yuv);
    cvtColor(out_yuv, out_yuv, CV_YUV2BGR);
    imwrite("C:\\Users\\25757\\Documents\\Visual Studio 2013\\Projects\\Project2\\Project2\\ct-awb_bpc_nr\\after_nlm_8bit.png", out_yuv);

    out_Y = NLM(Y);
    color_corrected[0] = out_Y;
    merge(color_corrected, 3, out_yuv);
    cvtColor(out_yuv, out_yuv, CV_YUV2BGR);
    imwrite("C:\\Users\\25757\\Documents\\Visual Studio 2013\\Projects\\Project2\\Project2\\ct-awb_bpc_nr\\after_nlm.png", out_yuv);

    fastNlMeansDenoising(Y, out_Y_opencv, 10, 5, 11);
    color_corrected[0] = out_Y;
    merge(color_corrected, 3, out_yuv);
    cvtColor(out_yuv, out_yuv, CV_YUV2BGR);
    imwrite("C:\\Users\\25757\\Documents\\Visual Studio 2013\\Projects\\Project2\\Project2\\ct-awb_bpc_nr\\after_nlm_opencv.png", out_yuv);
}


int Chroma_test()
	{
		clock_t start, end;
		Mat img = imread("C:\\Users\\25757\\Documents\\Visual Studio 2013\\Projects\\Project2\\Project2\\ct-awb_bpc_nr\\24.jpg", IMREAD_COLOR);
		Mat src = imread("C:\\Users\\25757\\Documents\\Visual Studio 2013\\Projects\\Project2\\Project2\\ct-awb_bpc_nr\\UVNoise.bmp", IMREAD_COLOR);
		Mat noisy = imread("C:\\Users\\25757\\Documents\\Visual Studio 2013\\Projects\\Project2\\Project2\\ct-awb_bpc_nr\\BGR_UVNoise.png", IMREAD_COLOR);
		Mat yuv_noise;
		cout << img.rows<< endl;
		cout << img.cols << endl;
		cout << noisy.rows << endl;
		cout << noisy.cols << endl;
		//cvtColor(src,yuv_noise, COLOR_BGR2YUV);
		vector<Mat> yuv;
		split(src, yuv);
		if (!src.data)
		{
			cout << "读取图片错误，请重新输入正确路径！\n";
			system("pause");
			return -1;
		}
		start = clock();
		chroma_denoising(yuv[1], yuv[2], Size(7, 7), 40.0, 40.0);
		Mat dst;
		merge(yuv, dst);
		cvtColor(dst, dst, COLOR_YUV2BGR);
		cout << img.rows << endl;
		cout << img.cols << endl;
		end = clock();
		cout << "耗时：" << (end - start) / 1000.0 << " s" << endl;
		cout << noisy.rows << endl;
		cout << noisy.cols << endl;
		cout << "noisy PSNR:" << getPSNR(img, noisy) << endl;
		cout << "denoised PSNR:" << getPSNR(img, dst) << endl;
		imshow("噪声图", noisy);
		imshow("UV降噪图", dst);
		imwrite("C:\\Users\\25757\\Documents\\Visual Studio 2013\\Projects\\Project2\\Project2\\ct-awb_bpc_nr\\denoisedMean.png", dst);
		waitKey(0);
	}
