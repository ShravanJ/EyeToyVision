/*
*	EyeToy Vision
*	This program uses software from OpenCV (http://opencv.org)
*	Author: Shravan Jambukesan (shravan.j97@gmail.com)
*	Date: 7/4/2017
*/


#include "opencv2\highgui.hpp"
#include "opencv2\imgproc.hpp"
#include "opencv2\objdetect\objdetect.hpp"
#include "opencv2\videoio\videoio.hpp"
#include "opencv2\imgcodecs\imgcodecs.hpp"
#include "opencv2\face.hpp"

#include <vector>
#include <stdio.h>
#include <Windows.h>
#include <iostream>

using namespace cv;
using namespace std;
using namespace cv::face;

void train(Mat img);

int num_components = 10;
Ptr<FaceRecognizer> model = createFisherFaceRecognizer(6, 10.0);

vector<Mat> images;
vector<int> labels;

string personName = "";

bool trainStarted = false;
bool trainingFinished = false;

int main(int argc, const char** argv)
{
	string input;
	cout << "EyeToy Vision - Powered by OpenCV" << endl;
	cout << "Type in 'train' to start training or 'rec' to test recognition: ";

	cin >> input;

	if (input == "train")
	{
		trainStarted = true;
		while (!trainingFinished)
		{
			VideoCapture cap(0);

			if (!cap.isOpened())
			{
				cout << "Could not open device" << endl;
				break;
			}
			else
			{
				Mat img;
				cap >> img;
				cout << "Training started" << endl;
				train(img);
			}
		}
	}

	if (input == "rec")
	{
		model->load("shravan_trained.xml");
		cout << "Model loaded" << endl;
		VideoCapture cap(0);
		while (true)
		{

			if (!cap.isOpened())
			{
				cout << "Could not start video capture" << endl;
				break;
			}
			else
			{
				Mat img;
				cap >> img;
				Mat gray;
				cvtColor(img, gray, CV_BGR2GRAY);
				namedWindow("EyeToy Vision", WINDOW_AUTOSIZE);
				imshow("EyeToy Vision", img);
				if (model->predict(gray) != -1)
				{
					cout << "Recognized with ID: " << model->predict(gray) << endl;
				}
				int key = waitKey(20);

			}

		}

	}

	return 0;
}

void train(Mat img)
{
	Mat gray_img;
	cvtColor(img, gray_img, CV_BGR2GRAY);
	for (int num = 1; num <= 6; num++)
	{
		stringstream ss;
		string jpeg = ".png";
		string filename = "shravan_";
		ss << filename << num << jpeg;
		filename = ss.str();
		cout << "Training..." << endl;
		imwrite(filename, gray_img);
		if (num == 6)
		{
			try
			{
				images.push_back(imread("shravan_1.png", CV_LOAD_IMAGE_GRAYSCALE)); labels.push_back(1);
				images.push_back(imread("shravan_2.png", CV_LOAD_IMAGE_GRAYSCALE)); labels.push_back(1);
				images.push_back(imread("shravan_3.png", CV_LOAD_IMAGE_GRAYSCALE)); labels.push_back(1);
				images.push_back(imread("shravan_4.png", CV_LOAD_IMAGE_GRAYSCALE)); labels.push_back(1);
				images.push_back(imread("shravan_5.png", CV_LOAD_IMAGE_GRAYSCALE)); labels.push_back(1);
				images.push_back(imread("shravan_6.png", CV_LOAD_IMAGE_GRAYSCALE)); labels.push_back(1);
				images.push_back(imread("smile_1.png", CV_LOAD_IMAGE_GRAYSCALE)); labels.push_back(2);
				images.push_back(imread("smile_2.png", CV_LOAD_IMAGE_GRAYSCALE)); labels.push_back(2);
				images.push_back(imread("smile_3.png", CV_LOAD_IMAGE_GRAYSCALE)); labels.push_back(2);
				images.push_back(imread("smile_4.png", CV_LOAD_IMAGE_GRAYSCALE)); labels.push_back(2);
				images.push_back(imread("smile_5.png", CV_LOAD_IMAGE_GRAYSCALE)); labels.push_back(2);
				images.push_back(imread("smile_6.png", CV_LOAD_IMAGE_GRAYSCALE)); labels.push_back(2);
				model->train(images, labels);
				trainingFinished = true;
				model->save("shravan_trained.xml");
				cout << "Training finished" << endl;
			}
			catch (cv::Exception e)
			{
				cout << e.msg << endl;
			}

		}
	}



}



