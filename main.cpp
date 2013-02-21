#include  "MCSS.h"
#include  <opencv2/opencv.hpp>

using namespace cv;

void lgcLabel_mouse_call_back(int event, int x, int y, int flags, void* userdata)
{
	Mat *mat = (Mat *)userdata;
	if( event == CV_EVENT_LBUTTONDOWN )
	{
		int label = *mat->ptr(y, x)-1;
		cerr << "(" << x << " " << y << ") Label = " << label << endl;
	}
}

void lumRatio_mouse_call_back(int event, int x, int y, int flags, void* userdata)
{
	Mat *mat = (Mat *)userdata;
	float b, g, r;
	if( event == CV_EVENT_LBUTTONDOWN )
	{
		b = mat->at<Vec3f>(y, x)[0];
		g = mat->at<Vec3f>(y, x)[1];
		r = mat->at<Vec3f>(y, x)[2];
		cerr << "(" << x << " " << y << ") = " << b << " " << g << " " << r << endl;
	}
}

int main(int argc, char *argv[])
{
	Mat frame, bg, mask, dst, tmp;
	VideoCapture cap, cap2, cap3;
	MCSS fgr;
	MCSS_Param p;
	int n = 0, key;
	bool resizeFrame = false;

	if( argc != 4 )
	{
		cerr << "Usage: " << argv[0] << " <video-path> <mask-path> <bg-path>" << endl;
		return -1;
	}

	cap.open(std::string(argv[1]));
	if( !cap.isOpened() )
	{
		cout << endl << "Can not open " << argv[1] << endl;
		return -1;
	}
	cap >> frame;
	if(!frame.data)
	{
		cout << "can not read data from " << argv[1] << endl;
		return -1;
	}
	if( frame.cols*frame.rows > 600*600 )
		resizeFrame = true;

	cap2.open(std::string(argv[2]));
	if( !cap2.isOpened() )
	{
		cout << endl << "Can not open " << argv[2] << endl;
		return -1;
	}
	cap2 >> tmp;
	if(!tmp.data)
	{
		cout << "can not read data from " << argv[2] << endl;
		return -1;
	}

	cap3.open(std::string(argv[3]));
	if( !cap3.isOpened() )
	{
		cout << endl << "Can not open " << argv[3] << endl;
		return -1;
	}
	cap3 >> tmp;
	if(!tmp.data)
	{
		cout << "can not read data from " << argv[3] << endl;
		return -1;
	}

	namedWindow("lumRatio");
	namedWindow("lgcImg");
	setMouseCallback("lumRatio", lumRatio_mouse_call_back, &fgr.lumRatio);
	setMouseCallback("lgcImg", lgcLabel_mouse_call_back, &fgr.lgcLabel);

	p = fgr.getParameters();
	p.alpha = 0.000621;
	p.threshold1 = 1.7;
	p.threshold2 = 12.5;
	p.isMGThrFixed = true;
	p.hasFringe = true;
	p.mgThrFixed = Vec3f(0.23, 0.23, 0.23);
	// p.alpha = 0.00021;
	fgr.setParameters(p);
	while( (key = waitKey(30)) != 27 )
	{
		n++;
		if( key == ' ' )
			while( (key = waitKey(0)) != ' ' );
		cap >> frame;
		if( !frame.data )
			break;

		cap2 >> tmp;
		cvtColor(tmp, mask, CV_BGR2GRAY);
		threshold(mask, mask, 128, 255, CV_THRESH_BINARY);
		if( n < 10 )
		{
			cap3 >> bg;
			// resize(bg, bg, tmp.size(), 1, 0);
		}
		// bg = imread(argv[3]);

		if( resizeFrame )
		{
			resize(frame, frame, Size(frame.cols/2, frame.rows/2));
			resize(mask, mask, frame.size());
			resize(bg, bg, frame.size());
		}

		imshow("image", frame);
		imshow("mask", mask);
		imshow("background", bg);
		fgr(frame, bg, mask, dst);
		// threshold(dst, dst, 128, 255, CV_THRESH_BINARY);
		fgr.lumRatio.convertTo(tmp, CV_8UC3, 30, 0);
		imshow("dst", dst);
		imshow("lumRatio", tmp);

#if 0
		stringstream stream;
		stream << "pic/pic_" << n++ << ".jpg";
		imwrite(stream.str(), dst);
#endif
	}

	return 0;
}

