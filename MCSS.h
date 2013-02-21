#ifndef  __MCSS_H__
#define  __MCSS_H__

#include  <iostream>
#include  <string>
#include  <opencv2/opencv.hpp>

using namespace cv;
using namespace std;


/* some default parameters */
#define  MAX_OBJ_NUM		100
#define  MIN_OBJ_AREA		200
#define  STACK_SIZE			200000
#define  MAX_LGC_NUM		900
#define  MIN_LGC_AREA		5
#define  SMALL_LGC_LABEL	65535

#define  LOW_LUMINANCE_RATIO_THRESHOLD	1
#define  HIGH_LUMINANCE_RATIO_THRESHOLD	6
#define  LOW_RELATIVE_SIZE_THRESHOLD	0.04
#define  HIGH_RELATIVE_SIZE_THRESHOLD	1
#define  EXTRINSIC_TERMINAL_POINT_WEIGHT_THRESHOLD	0.22
#define  ALPHA	0.000321
#define  FRAME_WINDOW 60
#define  HISTORY	30
#define  DEFAULT_MGTHR_FIXED	(Vec3f(0.22, 0.22, 0.22))

/* some parameters of the model */
struct MCSS_Param
{
	float threshold1, threshold2;
	float lambda_low, lambda_high;
	float tao;
	float alpha;
	Vec3f mgThrFixed;
	bool isMGThrFixed;
	bool hasFringe;
};

/* moving cast shadow suppression */
/*
 * the class implements the following algorithm:
 * "Accurate moving cast shadow suppression based on local color constancy detection"
 * Ariel Amato, Mikhail G. Mozerov, Andrew D. Bagdanov, and Jordi Gonz` lez
*/
class MCSS
{
	public:
		MCSS();
		/* update the model */
		void operator()(Mat current, Mat background, Mat mask, OutputArray output);
		/* get some current parameters */
		MCSS_Param getParameters();
		/* set parameters */
		void setParameters(MCSS_Param parameters);

		/* luminance ratio */
		Mat lumRatio;
		/* label each pixel which LGC they belong */
		Mat lgcLabel;
		/* result mask */
		Mat dst;

	private:
		int nframes;
		Size frameSize;
		int frameType;
		/* the threshold to limit the LGC area */
		float threshold1, threshold2;
		/* needed when calculate the luminance ratio */
		float v;
		/* relative size threshold lambda */
		float lambda_low, lambda_high;
		/* Extrinsic terminal point weight threshold tao */
		float tao;
		/* an argument needed when computing minimum gradient threshold */
		float alpha;
		/* we can use a fixed value or calculate it in each frame */
		bool isMGThrFixed;
		Vec3f mgThrFixed;
		/* indicate if the narrow bright fringe exist */
		bool hasFringe;

		/* check if each lgc belongs to shadow */
		vector<bool> lgcIsShadow;
		/* sum of each pixel in mean_bg */
		vector<Vec3f> mean_average_bg;
		/* sum of each pixel in variance_bg */
		vector<Vec3f> standard_deviation_average_bg;
		/* Minimum gradient threshold in RGB space */
		vector<Vec3f> mgThr;
		/* mean value in the lgc */
		vector<Vec3f> meanLGC;
		/* terminal pixel weight */
		vector<float> tpw;

		Mat lgcImg;
		/* label each pixel which object they belong */
		Mat objLabel;
		/* area of each object */
		vector<int> objArea;
		/* area of each LGC */
		vector<int> lgcArea;
		/* record the object index of each lgc */
		vector<int> lgcToObj;
		/* mean value and standard deviation of background
		 * cont, mean, std
		 * */
		Mat cont, mean, STD;

		int getNearestVal(Point pos);
};


#endif  /*__MCSS_H__*/
