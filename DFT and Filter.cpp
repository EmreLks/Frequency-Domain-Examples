#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui.hpp>
#include <iostream>
#include <cmath>
#include <fstream>

using namespace cv;
using namespace std;

void PrepareImageForConv(Mat src, Mat* dest, int kernelWidth, int kernelHeight);
void MyDft(Mat inputImg, Mat& outputImg, Mat& realMat, Mat& imagMat);
void InMyDft(Mat realMat, Mat imagMat, Mat& outputImg);
void MultiplyComplexNumber(Mat realPart1, Mat imagPart1, Mat realPart2, Mat imagPart2, Mat& realResult, Mat& imagResult);
void MultiplyComplexNumberWithRealNumber(Mat realPart1, Mat imagPart1, Mat realPart2, Mat& realResult, Mat& imagResult);
void PrepareForDFT(Mat& input);
void CreateLowPassFilter(Mat& lowPassFilter, int radius);
void CreateBHPFFilter(Mat& bhpfFilter, int order, int Do);
void CenterImage(Mat& src);
int main()

    int question = 0;
    cout << "Chose questions[1, 4]: " << endl;
    cin >> question;

    switch (question)
    {
        case 1:
        {
            string image_path = "images/question_1.tif";
            Mat inputImg = imread(image_path, IMREAD_GRAYSCALE);
            Mat floatInputImg;
            inputImg.convertTo(floatInputImg, CV_32FC1);

            Mat inverseDftImg = Mat::zeros(floatInputImg.rows, floatInputImg.cols, CV_32FC1);
            Mat spectrumImg = Mat::zeros(floatInputImg.rows, floatInputImg.cols, CV_32FC1);
            Mat realPart = Mat::zeros(floatInputImg.rows, floatInputImg.cols, CV_32FC1);
            Mat imaginaryPart = Mat::zeros(floatInputImg.rows, floatInputImg.cols, CV_32FC1);

            if (inputImg.empty())
            {
                cout << "Could not read the image: " << image_path << endl;
                return 1;
            }
            // else do nothing.

            MyDft(floatInputImg, spectrumImg, realPart, imaginaryPart);
            InMyDft(realPart, imaginaryPart, inverseDftImg);

            spectrumImg += Scalar::all(1);
            log(spectrumImg, spectrumImg);
            normalize(spectrumImg, spectrumImg, 0, 1, NORM_MINMAX);

            normalize(inverseDftImg, inverseDftImg, 0, 255, NORM_MINMAX);
            inverseDftImg.convertTo(inverseDftImg, CV_8UC1);

            CenterImage(spectrumImg);

            namedWindow("Input Image", WINDOW_AUTOSIZE);
            imshow("Input Image", inputImg);

            namedWindow("Spectrum Image", WINDOW_AUTOSIZE);
            imshow("Spectrum Image", spectrumImg);

            namedWindow("Inverse DFT Image", WINDOW_AUTOSIZE);
            imshow("Inverse DFT Image", inverseDftImg);

            break;
        }

        case 2:
        {
            string image_path = "images/question_2.tif";
            Mat inputImg = imread(image_path, IMREAD_GRAYSCALE);
            Mat floatInputImg;
            inputImg.convertTo(floatInputImg, CV_32FC1);
            if (floatInputImg.empty()) {
                cout << "Could not read the image: " << image_path << endl;
                return 1;
            }
            // Fill sobel mask.
            Mat sobelMask = Mat::zeros(3, 3, CV_32FC1);
            sobelMask.at<float>(0, 0) = -1; sobelMask.at<float>(0, 1) = 0; sobelMask.at<float>(0, 2) = 1;
            sobelMask.at<float>(1, 0) = -2; sobelMask.at<float>(1, 1) = 0; sobelMask.at<float>(1, 2) = 2;
            sobelMask.at<float>(2, 0) = -1; sobelMask.at<float>(2, 1) = 0; sobelMask.at<float>(2, 2) = 1;

            // Zero padding operations.
            Mat sobelMaskPadding, inputImgPadding;
            PrepareImageForConv(sobelMask, &sobelMaskPadding, floatInputImg.rows, floatInputImg.cols);
            PrepareImageForConv(floatInputImg, &inputImgPadding, sobelMask.rows, sobelMask.cols);
            // DFT datas.
            // Input.
            Mat dftInputImg = Mat::zeros(inputImgPadding.rows, inputImgPadding.cols, CV_32FC1);
            Mat inputRealPart = Mat::zeros(inputImgPadding.rows, inputImgPadding.cols, CV_32FC1);
            Mat inputImagPart = Mat::zeros(inputImgPadding.rows, inputImgPadding.cols, CV_32FC1);
            // Sobel.
            Mat dftSobelMask = Mat::zeros(sobelMaskPadding.rows, sobelMaskPadding.cols, CV_32FC1);
            Mat sobelRealPart = Mat::zeros(sobelMaskPadding.rows, sobelMaskPadding.cols, CV_32FC1);
            Mat sobelImagPart = Mat::zeros(sobelMaskPadding.rows, sobelMaskPadding.cols, CV_32FC1);
            // Result:
            Mat resultRealPart = Mat::zeros(inputImgPadding.rows, inputImgPadding.cols, CV_32FC1);
            Mat resultImagPart = Mat::zeros(inputImgPadding.rows, inputImgPadding.cols, CV_32FC1);

            // (-1)^(x + y)
            PrepareForDFT(inputImgPadding);
            PrepareForDFT(sobelMaskPadding);

            // Input Img.
            MyDft(inputImgPadding, dftInputImg, inputRealPart, inputImagPart);

            // Sobel mask.
            MyDft(sobelMaskPadding, dftSobelMask, sobelRealPart, sobelImagPart);

            MultiplyComplexNumber(inputRealPart, inputImagPart, sobelRealPart, sobelImagPart, resultRealPart, resultImagPart);

            Mat inverserDft = Mat::zeros(resultRealPart.rows, resultRealPart.cols, CV_32FC1);

            InMyDft(resultRealPart, resultImagPart, inverserDft);

            PrepareForDFT(inverserDft);

            normalize(inverserDft, inverserDft, 0, 255, NORM_MINMAX);

            inverserDft.convertTo(inverserDft, CV_8UC1);

            CenterImage(inverserDft);

            namedWindow("Input Image", WINDOW_AUTOSIZE);
            imshow("Input Image", inputImg);

            namedWindow("Inverse DFT Image", WINDOW_AUTOSIZE);
            imshow("Inverse DFT Image", inverserDft);

            break;
        }

        case 3:
        {
            string image_path = "images/question_3.tif";
            Mat inputImg = imread(image_path, IMREAD_GRAYSCALE);
            Mat floatInputImg;
            inputImg.convertTo(floatInputImg, CV_32FC1);
            if (floatInputImg.empty()) {
                cout << "Could not read the image: " << image_path << endl;
                return 1;
            }

            // Create low pass filter.
            Mat lowPassFilter = Mat::zeros(floatInputImg.rows, floatInputImg.cols, CV_32FC1);
            // radius 30.
            CreateLowPassFilter(lowPassFilter, 30);

            // Apply DFT.
            // Input.
            Mat dftInputImg = Mat::zeros(floatInputImg.rows, floatInputImg.cols, CV_32FC1);
            Mat inputRealPart = Mat::zeros(floatInputImg.rows, floatInputImg.cols, CV_32FC1);
            Mat inputImagPart = Mat::zeros(floatInputImg.rows, floatInputImg.cols, CV_32FC1);
            // Low Pass Filter.
            Mat dftLowPass = Mat::zeros(lowPassFilter.rows, lowPassFilter.cols, CV_32FC1);
            Mat lowPassRealPart = Mat::zeros(lowPassFilter.rows, lowPassFilter.cols, CV_32FC1);
            Mat lowPassImagPart = Mat::zeros(lowPassFilter.rows, lowPassFilter.cols, CV_32FC1);
            // Result:
            Mat resultRealPart = Mat::zeros(floatInputImg.rows, floatInputImg.cols, CV_32FC1);
            Mat resultImagPart = Mat::zeros(floatInputImg.rows, floatInputImg.cols, CV_32FC1);

            // Input Img.
            MyDft(floatInputImg, dftInputImg, inputRealPart, inputImagPart);

            // Low Pass Filter.
            MyDft(lowPassFilter, dftLowPass, lowPassRealPart, lowPassImagPart);

            MultiplyComplexNumber(inputRealPart, inputImagPart, lowPassRealPart, lowPassImagPart, resultRealPart, resultImagPart);
            //MultiplyComplexNumberWithRealNumber(inputRealPart, inputImagPart, lowPassFilter, resultRealPart, resultImagPart);

            // Inverse DFT.
            Mat inverserDft = Mat::zeros(resultRealPart.rows, resultRealPart.cols, CV_32FC1);

            InMyDft(resultRealPart, resultImagPart, inverserDft);

            normalize(inverserDft, inverserDft, 0, 255, NORM_MINMAX);

            inverserDft.convertTo(inverserDft, CV_8UC1);

            CenterImage(inverserDft);

            namedWindow("Input Image", WINDOW_AUTOSIZE);
            imshow("Input Image", inputImg);

            namedWindow("Inverse DFT Image", WINDOW_AUTOSIZE);
            imshow("Inverse DFT Image", inverserDft);

            break;
        }

        case 4:
        {
            string image_path = "images/question_4.tif";
            Mat inputImg = imread(image_path, IMREAD_GRAYSCALE);
            Mat floatInputImg;
            inputImg.convertTo(floatInputImg, CV_32FC1);
            if (floatInputImg.empty()) {
                cout << "Could not read the image: " << image_path << endl;
                return 1;
            }

            // Create BHFP Filter.
            Mat bhpfFilter = Mat::zeros(floatInputImg.rows, floatInputImg.cols, CV_32FC1);
            
            // Order:2, Do:60.
            CreateBHPFFilter(bhpfFilter, 2, 60);

            // Apply DFT.
            // Input.
            Mat dftInputImg = Mat::zeros(floatInputImg.rows, floatInputImg.cols, CV_32FC1);
            Mat inputRealPart = Mat::zeros(floatInputImg.rows, floatInputImg.cols, CV_32FC1);
            Mat inputImagPart = Mat::zeros(floatInputImg.rows, floatInputImg.cols, CV_32FC1);

            // Result:
            Mat resultRealPart = Mat::zeros(floatInputImg.rows, floatInputImg.cols, CV_32FC1);
            Mat resultImagPart = Mat::zeros(floatInputImg.rows, floatInputImg.cols, CV_32FC1);

            // Input Img.
            MyDft(floatInputImg, dftInputImg, inputRealPart, inputImagPart);

            MultiplyComplexNumberWithRealNumber(inputRealPart, inputImagPart, bhpfFilter, resultRealPart, resultImagPart);

            // Inverse DFT.
            Mat inverserDft = Mat::zeros(resultRealPart.rows, resultRealPart.cols, CV_32FC1);

            InMyDft(resultRealPart, resultImagPart, inverserDft);

            normalize(inverserDft, inverserDft, 0, 255, NORM_MINMAX);

            inverserDft.convertTo(inverserDft, CV_8UC1);

            namedWindow("Input Image", WINDOW_AUTOSIZE);
            imshow("Input Image", inputImg);

            namedWindow("Inverse DFT Image", WINDOW_AUTOSIZE);
            imshow("Inverse DFT Image", inverserDft);

            break;
        }

        default:
        {
            break;
        }
    }
    // End of the switch case.

    int k = waitKey(0);
    if (k == 'e')
    {
        exit(0);
    }
    return 0;
}

void PrepareImageForConv(Mat src, Mat* dest, int kernelWidth, int kernelHeight)
{
    int destImgWidth = src.cols, destImgHeight = src.rows;
    int rowStartIndex = 0, colStartIndex = 0;

    if (kernelWidth > 1)
    {
        destImgWidth += (kernelWidth - 1);

        rowStartIndex = floor((kernelWidth - 1) / 2);
    }
    else
    {
        rowStartIndex = 0;
    }

    if (kernelHeight > 1)
    {
        destImgHeight += (kernelHeight - 1);

        colStartIndex = floor((kernelHeight - 1) / 2);
    }
    else
    {
        colStartIndex = 0;
    }

    *dest = Mat::zeros(destImgHeight, destImgWidth, CV_32FC1);

    for (int i = 0; i < src.rows; i++)
    {
        for (int j = 0; j < src.cols; j++)
        {
            dest->at<float>(i + rowStartIndex, j + colStartIndex) = src.at<float>(i, j);
        }
    }
}

void MyDft(Mat inputImg, Mat& outputImg, Mat& realMat, Mat& imagMat)
{
    int HEIGHT = inputImg.rows, WIDTH = inputImg.cols;

    Mat realHorImg = Mat::zeros(HEIGHT, WIDTH, CV_32FC1);
    Mat imagHorImg = Mat::zeros(HEIGHT, WIDTH, CV_32FC1);

    double real = 0, imaginary = 0, sum = 0, teta = 0;

    for (int u = 0; u < HEIGHT; u++)
    {
        for (int v = 0; v < WIDTH; v++)
        {
            real = 0.0;
            imaginary = 0.0;

            for (int y = 0; y < WIDTH; y++)
            {
                teta = ((-2) * CV_PI * v * y) / WIDTH;

                real += inputImg.at<float>(u, y) * cos(teta);
                imaginary += inputImg.at<float>(u, y) * sin(teta);
            }
            // end of the loop.

            realHorImg.at<float>(u, v) = real;
            imagHorImg.at<float>(u, v) = imaginary;
        }
        // end of the loop.
    }
    // end of the loop.
    for (int v = 0; v < WIDTH; v++)
    {
        for (int u = 0; u < HEIGHT; u++)
        {
            real = 0.0;
            imaginary = 0.0;
            sum = 0.0;

            for (int x = 0; x < HEIGHT; x++)
            {
                teta = ((-2) * CV_PI * u * x) / HEIGHT;

                real += realHorImg.at<float>(x, v) * cos(teta)
                    - imagHorImg.at<float>(x, v) * sin(teta);

                imaginary += realHorImg.at<float>(x, v) * sin(teta)
                    + imagHorImg.at<float>(x, v) * cos(teta);
            }
            // end of the loop.

            realMat.at<float>(u, v) = real;
            imagMat.at<float>(u, v) = imaginary;

            sum = sqrt(real * real + imaginary * imaginary);

            outputImg.at<float>(u, v) = sum;
        }
        // end of the loop.
    }
    // end of the loop.
}
void InMyDft(Mat realMat, Mat imagMat, Mat& outputImg)
{
    int HEIGHT = realMat.rows, WIDTH = realMat.cols;

    Mat realHorImg = Mat::zeros(HEIGHT, WIDTH, CV_32FC1);
    Mat imagHorImg = Mat::zeros(HEIGHT, WIDTH, CV_32FC1);

    double real = 0.0, imaginary = 0.0, teta = 0.0, sum = 0.0;

    for (int u = 0; u < HEIGHT; u++)
    {
        for (int v = 0; v < WIDTH; v++)
        {
            real = 0.0;
            imaginary = 0.0;

            for (int y = 0; y < WIDTH; y++)
            {
                teta = (2 * CV_PI * v * y) / WIDTH;

                real += realMat.at<float>(u, y) * cos(teta) - imagMat.at<float>(u, y) * sin(teta);
                imaginary += realMat.at<float>(u, y) * sin(teta) + imagMat.at<float>(u, y) * cos(teta);
            }
            // end of the loop.

            realHorImg.at<float>(u, v) = real;
            imagHorImg.at<float>(u, v) = imaginary;
        }
        // end of the loop.
    }
    // end of the loop.
    for (int v = 0; v < WIDTH; v++)
    {
        for (int u = 0; u < HEIGHT; u++)
        {
            real = 0.0;
            imaginary = 0.0;
            sum = 0.0;

            for (int x = 0; x < HEIGHT; x++)
            {
                teta = (2 * CV_PI * u * x) / HEIGHT;

                real += realHorImg.at<float>(x, v) * cos(teta)
                    - imagHorImg.at<float>(x, v) * sin(teta);

                imaginary += realHorImg.at<float>(x, v) * sin(teta)
                    + imagHorImg.at<float>(x, v) * cos(teta);
            }
            // end of the loop.

            real = real / (WIDTH * HEIGHT);
            imaginary = imaginary / (WIDTH * HEIGHT);

            sum = sqrt(real * real + imaginary * imaginary);

            outputImg.at<float>(u, v) = sum;
        }
        // end of the loop.
    }
    // end of the loop.
}

void MultiplyComplexNumber(Mat realPart1, Mat imagPart1, Mat realPart2, Mat imagPart2, Mat& realResult, Mat& imagResult)
{
    for (int i = 0; i < realPart1.rows; i++)
    {
        for (int j = 0; j < realPart1.cols; j++)
        {
            realResult.at<float>(i, j) = realPart1.at<float>(i, j) * realPart2.at<float>(i, j)
                - imagPart1.at<float>(i, j) * imagPart2.at<float>(i, j);

            imagResult.at<float>(i, j) = realPart1.at<float>(i, j) * imagPart2.at<float>(i, j)
                + imagPart1.at<float>(i, j) * realPart2.at<float>(i, j);
        }
        // end of the loop.
    }
    // end of the loop.
}

void MultiplyComplexNumberWithRealNumber(Mat realPart1, Mat imagPart1, Mat realPart2,Mat& realResult, Mat& imagResult)
{
    for (int i = 0; i < realPart1.rows; i++)
    {
        for (int j = 0; j < realPart1.cols; j++)
        {
            realResult.at<float>(i, j) = realPart1.at<float>(i, j) * realPart2.at<float>(i, j);

            imagResult.at<float>(i, j) = imagPart1.at<float>(i, j) * realPart2.at<float>(i, j);
        }
        // end of the loop.
    }
    // end of the loop.
}

void PrepareForDFT(Mat& input)
{
    for (int i = 0; i < input.rows; i++)
    {
        for (int j = 0; j < input.cols; j++)
        {
            input.at<float>(i, j) = input.at<float>(i, j) * pow(-1, (i + j));
        }
        // end of the loop.
    }
    // end of the loop.
}

void CreateLowPassFilter(Mat& lowPassFilter, int radius)
{
    double dUV = 0;

    for (int i = 0; i < lowPassFilter.rows; i++)
    {
        for (int j = 0; j < lowPassFilter.cols; j++)
        {
            dUV = sqrt(pow((i - lowPassFilter.rows / 2), 2) + pow((j - lowPassFilter.cols / 2), 2));

            if (dUV <= radius)
            {
                lowPassFilter.at<float>(i, j) = 1;
            }
            else
            {
                lowPassFilter.at<float>(i, j) = 0;
            }
        }
        // end of the loop.
    }
    // end of the loop.
}

void CreateBHPFFilter(Mat& bhfpFilter, int order, int Do)
{
    double dUV = 0;
    double hUV = 0;

    for (int i = 0; i < bhfpFilter.rows; i++)
    {
        for (int j = 0; j < bhfpFilter.cols; j++)
        {
            dUV = sqrt(pow(i, 2) + pow(j, 2));

            hUV = 1 / ( 1 + pow((Do / dUV), 2 * order) );

            bhfpFilter.at<float>(i, j) = hUV;
        }
        // end of the loop.
    }
    // end of the loop.
}

void CenterImage(Mat& src)
{
    int cx = src.cols / 2; int cy = src.rows / 2;

    Mat part1_r(src, Rect(0, 0, cx, cy)); //Element coordinates are expressed as (cx,cy)
    Mat part2_r(src, Rect(cx, 0, cx, cy));
    Mat part3_r(src, Rect(0, cy, cx, cy));
    Mat part4_r(src, Rect(cx, cy, cx, cy));

    Mat temp;
    part1_r.copyTo(temp); //Upper left and lower right exchange position
    part4_r.copyTo(part1_r);
    temp.copyTo(part4_r);

    part2_r.copyTo(temp); //Upper right and bottom left exchange position
    part3_r.copyTo(part2_r);
    temp.copyTo(part3_r);
}