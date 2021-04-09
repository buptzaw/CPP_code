#include<iostream>
#include<cmath>
#include<opencv2/opencv.hpp>
#include <time.h>
#include "tools.h"
using namespace std;
using namespace cv;



Mat bpc_interval(Mat bayerData, int neighborhood_size = 3)
{
    if ((neighborhood_size % 2) == 0)
    {
        cout << "neighborhood_size shoud be odd number, recommended value 3" << endl;
        return bayerData;
    }

    Mat img = bayerData;
    int w = img.cols, h = img.rows;
    // Separate out the quarter resolution images
    Mat D[4];// Empty dictionary
    getRGGB(img, D);

    // number of pixels to be padded at the borders
    int no_of_pixel_pad = floor(neighborhood_size / 2.);

    for (int idx = 0; idx < 4; idx++) // perform same operation for each quarter
    {
        // display progress
        cout << "bad pixel correction: Quarter " << (idx + 1) << " of 4" << endl;

        Mat data = D[idx];
        int width = data.cols, height = data.rows;

        // pad pixels at the borders
        // reflect would not repeat the border value
        copyMakeBorder(data, data, no_of_pixel_pad, no_of_pixel_pad, no_of_pixel_pad, no_of_pixel_pad, BORDER_REFLECT);
        for (int i = no_of_pixel_pad; i + no_of_pixel_pad + 1 < height; i++)
        {
            for (int j = no_of_pixel_pad; j + no_of_pixel_pad + 1 < width; j++)
            {
                // save the middle pixel value
                uchar mid_pixel_val = data.at<uchar>(i, j);

                // extract the neighborhood
                Mat neighborhood = data(Rect(j - no_of_pixel_pad, i - no_of_pixel_pad, 2 * no_of_pixel_pad + 1, 2 * no_of_pixel_pad + 1)); // [i - no_of_pixel_pad:i + no_of_pixel_pad + 1, j - no_of_pixel_pad : j + no_of_pixel_pad + 1]

                // set the center pixels value same as the left pixel
                // Does not matter replace with right or left pixel
                // is used to replace the center pixels value
                neighborhood.at<uchar>(no_of_pixel_pad, no_of_pixel_pad) = neighborhood.at<uchar>(no_of_pixel_pad, no_of_pixel_pad - 1);

                double min_neighborhood = 0;
                double max_neighborhood = 0;
                minMaxIdx(neighborhood, &min_neighborhood, &max_neighborhood);

                if (mid_pixel_val < min_neighborhood)
                {
                    data.at<uchar>(i, j) = min_neighborhood;
                }
                else if (mid_pixel_val > max_neighborhood)
                {
                    data.at<uchar>(i, j) = max_neighborhood;
                }
                else
                    data.at<uchar>(i, j) = mid_pixel_val;
            }
            //cout << i << " " << height << endl;

        }

        // Put the corrected image to the dictionary
        D[idx] = data(Rect(no_of_pixel_pad, no_of_pixel_pad, width, height));
    }

    getBayer(D, img);
    img.convertTo(img, CV_8UC1);

    return img;

}

Mat bpc_interval_detect(Mat bayerData, int neighborhood_size = 3)
{
    if ((neighborhood_size % 2) == 0)
    {
        cout << "neighborhood_size shoud be odd number, recommended value 3" << endl;
        return bayerData;
    }

    // convert to float32 in case they were not
    // Being consistent in data format to be float32
    Mat img = bayerData;
    int w = img.cols, h = img.rows;
    //bayerData.convertTo(img, CV_32FC1);

    // Separate out the quarter resolution images
    Mat D[4];// Empty dictionary
    getRGGB(bayerData, D);
    // number of pixels to be padded at the borders
    int no_of_pixel_pad = floor(neighborhood_size / 2.);
    Mat mask = Mat::zeros(img.size(), CV_8UC1);
    for (int idx = 0; idx < 4; idx++) // perform same operation for each quarter
    {
        // display progress
        cout << "bad pixel correction: Quarter " << (idx + 1) << " of 4" << endl;

        Mat data = D[idx];
        int width = data.cols, height = data.rows;

        // pad pixels at the borders
        // reflect would not repeat the border value
        copyMakeBorder(data, data, no_of_pixel_pad, no_of_pixel_pad, no_of_pixel_pad, no_of_pixel_pad, BORDER_REFLECT);
        for (int i = no_of_pixel_pad; i + no_of_pixel_pad + 1 < height; i++)
        {
            for (int j = no_of_pixel_pad; j + no_of_pixel_pad + 1 < width; j++)
            {
                // save the middle pixel value
                uchar mid_pixel_val = data.at<uchar>(i, j);

                // extract the neighborhood
                Mat neighborhood = data(Rect(j - no_of_pixel_pad, i - no_of_pixel_pad, 2 * no_of_pixel_pad + 1, 2 * no_of_pixel_pad + 1)); // [i - no_of_pixel_pad:i + no_of_pixel_pad + 1, j - no_of_pixel_pad : j + no_of_pixel_pad + 1]

                // set the center pixels value same as the left pixel
                // Does not matter replace with right or left pixel
                // is used to replace the center pixels value
                neighborhood.at<uchar>(no_of_pixel_pad, no_of_pixel_pad) = neighborhood.at<uchar>(no_of_pixel_pad, no_of_pixel_pad - 1);

                double min_neighborhood = 0;
                double max_neighborhood = 0;
                minMaxIdx(neighborhood, &min_neighborhood, &max_neighborhood);

                if (mid_pixel_val < min_neighborhood || mid_pixel_val > max_neighborhood)
                {
                    mask.at<uchar>(i, j) = 1;
                }

            }
        }
    }

    return mask;
}

Mat bpc_interval_correct(Mat bayerData, Mat mask, int neighborhood_size)
{
    if ((neighborhood_size % 2) == 0)
    {
        cout << "neighborhood_size shoud be odd number, recommended value 3" << endl;
        return bayerData;
    }

    // convert to float32 in case they were not
    // Being consistent in data format to be float32

    // Separate out the quarter resolution images
    Mat D[4];// Empty dictionary
    getRGGB(bayerData, D);
    Mat med = bayerData;
    for (int idx = 0; idx < 4; idx++)
    {
        medianBlur(D[idx], D[idx], neighborhood_size);
    }
    // Regrouping the data    
    getBayer(D, med);
    med.copyTo(bayerData, mask);
    return bayerData;
}


//rawpy enhance


void _find_bad_pixel_candidates_bayer2x2(Mat* D, Mat *Mask)
{
    // assert raw.raw_pattern.shape[0] == 2

    // optimized code path for common 2x2 pattern
    // create a view for each color, do 3x3 median on it, find bad pixels, correct coordinates
    // This shortcut allows to do median filtering without using a mask, which means
    // that OpenCV's extremely fast median filter algorithm can be used.
    int r = 3;

    double thresh = 245;
    Mat coords;

    // we have 4 colors(two greens are always seen as two colors)
    for (int i = 0; i < 4; i++)
    {
        Mat rawslice = D[i];
        //rawslice = np.require(rawslice, rawslice.dtype, 'C')
        Mat med;
        medianBlur(rawslice, med, r);

        absdiff(rawslice, med, med);
        // detect possible bad pixels
        Mask[i] = med > thresh;

        // convert to coordinates and correct for slicing
    }
}

Mat median_bpc(Mat rawimg, Mat* Mask, bool isDetect = true)
{
    cout << "----------------------------------------------------" << endl;
    cout << "Running bad pixel correction..." << endl;

    Mat D[4];
    clock_t start, end;
    start = clock();
    getRGGB(rawimg, D);
    end = clock();
    cout << "getrggb timeout:" << (end - start) / 1000. << " s" << endl;

    if (isDetect || Mask == NULL)
    {
        Mat M[4];
        _find_bad_pixel_candidates_bayer2x2(D,M);
        Mask = M;
    }
    int r = 3;
   

    for (int i = 0; i < 4; i++)
    {
        Mat smooth;
        medianBlur(D[i], smooth, r);
        smooth.copyTo(D[i], Mask[i]);
    }

    getBayer(D, rawimg);
    return rawimg;
}

void test()
{
   // Mat raw = imread("test_data/wb/bayer/color1_raw.png", 0);
	Mat raw = imread("C:\\Users\\25757\\Documents\\Visual Studio 2013\\Projects\\Project2\\Project2\\ct-awb_bpc_nr\\color1_raw.png", 0);
	//"C:\\Users\\25757\\Documents\\Visual Studio 2013\\Projects\\Project2\\Project2\\ct-awb_bpc_nr\\test.png"
    raw = median_bpc(raw, NULL);
    Mat bgr;
    cvtColor(raw,bgr, COLOR_BayerBG2BGR);
    imwrite("C:\\Users\\25757\\Documents\\Visual Studio 2013\\Projects\\Project2\\Project2\\ct-awb_bpc_nr\\bgr.png", bgr);
}
