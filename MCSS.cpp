/* 
 * Implementation of the moving cast shadow suppression model from:
 * "Accurate moving cast shadow suppression based on local color constancy detection"
 * Ariel Amato, Mikhail G.Mozerov, Andrew D.Bagdanov, and Jordi
 *
 * Author: 刘弘也 (nicklhy@gmail.com)
 * Date: 1-Aug-2012
 *
 * */

#include  "MCSS.h"
#define  debug

static Point stack[STACK_SIZE];

/**
 * @brief getNearestVal: check the pixel, in narrow bright fringe, if it belongs to shadow or foreground
 *						 ( F. Edge Noise Correction )
 *
 * @param mat: a matrix represent background(0), foreground(255) and shadow(127)
 * @param pos: the position of this pixel
 *
 * @return: return 127 if it belongs to shadow or 255 to foreground
 */
int MCSS::getNearestVal(Point pos)
{
	int a = 5, i1, i2, j1, j2, i, j, m = 0, n = 0;
	assert(dst.type() == CV_8UC1);
	while(1)
	{
		i1 = MAX(0, pos.x-a);
		i2 = MIN(dst.rows-1, i+a);
		j1 = MAX(0, pos.y-a);
		j2 = MIN(dst.cols-1, j+a);
		for( i = i1 ; i <= i2 ; ++i )
		{
			for( j = j1 ; j <= j2 ; ++j )
			{
				if( dst.at<uchar>(i, j) == 255 )
					++n;
				if( dst.at<uchar>(i, j) == 127 )
					++m;
			}
		}

		if( m == 0 && n == 0 )
			++a;
		if( m>0 && m>=n )
			return 127;
		if( n>0 && n>m )
			return 255;
	}
}

/**
 * @brief cleanRegionU8: use DFS to clean a region with val1 start from point p
 *
 * @param mat: the matrix you want to clean
 * @param p: the position you want to start from
 * @param val1: the value you want to clean
 * @param val2: the new value you want to set
 *
 * @return: number of points we cleaned
 */
static int cleanRegionU8(Mat mat, Point p, int val1, int val2 = 0)
{
	CV_Assert(val1 != val2);
	CV_Assert(mat.type() == CV_8UC1);
	int topIndex = 0, num = 0;
	Mat visited = Mat::zeros(mat.size(), CV_8UC1);

	if( mat.at<uchar>(p.x, p.y) != val1 )
		return 0;

	visited.at<uchar>(p.x, p.y) = 255;
	mat.at<uchar>(p.x, p.y) = val2;
	stack[topIndex++] = p;
	while( topIndex )
	{
		Point p1 = stack[topIndex-1];
		visited.at<uchar>(p1.x, p1.y) = 255;
		mat.at<uchar>(p1.x, p1.y) = val2;
		++num;
		if( (p1.x>0) && (mat.at<uchar>(p1.x-1, p1.y)==val1) && !visited.at<uchar>(p1.x-1, p1.y) )
		{
			stack[topIndex++] = Point(p1.x-1, p1.y);
			continue;
		}
		else if( (p1.y>0) && (mat.at<uchar>(p1.x, p1.y-1)==val1) && !visited.at<uchar>(p1.x, p1.y-1) )
		{
			stack[topIndex++] = Point(p1.x, p1.y-1);
			continue;
		}
		else if( (p1.x<mat.rows-1) && (mat.at<uchar>(p1.x+1, p1.y)==val1) && !visited.at<uchar>(p1.x+1, p1.y) )
		{
			stack[topIndex++] = Point(p1.x+1, p1.y);
			continue;
		}
		else if( (p1.y<mat.cols-1) && (mat.at<uchar>(p1.x, p1.y+1)==val1) && !visited.at<uchar>(p1.x, p1.y+1) )
		{
			stack[topIndex++] = Point(p1.x, p1.y+1);
			continue;
		}
		else if( (p1.x>0 && p1.y>0) && (mat.at<uchar>(p1.x-1, p1.y-1)==val1) && !visited.at<uchar>(p1.x-1, p1.y-1) )
		{
			stack[topIndex++] = Point(p1.x-1, p1.y-1);
			continue;
		}
		else if( (p1.x>0 && p1.y<mat.cols-1) && (mat.at<uchar>(p1.x-1, p1.y+1)==val1) && !visited.at<uchar>(p1.x-1, p1.y+1) )
		{
			stack[topIndex++] = Point(p1.x-1, p1.y+1);
			continue;
		}
		else if( (p1.x<mat.rows-1 && p1.y>0) && (mat.at<uchar>(p1.x+1, p1.y-1)==val1) && !visited.at<uchar>(p1.x+1, p1.y-1) )
		{
			stack[topIndex++] = Point(p1.x+1, p1.y-1);
			continue;
		}
		else if( (p1.x<mat.rows-1 && p1.y<mat.cols-1) && (mat.at<uchar>(p1.x+1, p1.y+1)==val1) && !visited.at<uchar>(p1.x+1, p1.y+1) )
		{
			stack[topIndex++] = Point(p1.x+1, p1.y+1);
			continue;
		}
		else
			--topIndex;
	}
	return (num+1)/2;
}

/**
 * @brief cleanRegionU16: use DFS to clean a region with val1 start from point p
 *
 * @param mat: the matrix you want to clean
 * @param p: the position you want to start from
 * @param val1: the value you want to clean
 * @param val2: the new value you want to set
 *
 * @return: number of points we cleaned
 */
static int cleanRegionU16(Mat mat, Point p, int val1, int val2 = 0)
{
	CV_Assert(val1 != val2);
	CV_Assert(mat.type() == CV_16UC1);
	int topIndex = 0, num = 0;
	Mat visited = Mat::zeros(mat.size(), CV_8UC1);

	if( mat.at<ushort>(p.x, p.y) != val1 )
		return 0;

	visited.at<uchar>(p.x, p.y) = 255;
	mat.at<ushort>(p.x, p.y) = val2;
	stack[topIndex++] = p;
	while( topIndex )
	{
		Point p1 = stack[topIndex-1];
		visited.at<uchar>(p1.x, p1.y) = 255;
		mat.at<ushort>(p1.x, p1.y) = val2;
		++num;
		if( (p1.x>0) && (mat.at<ushort>(p1.x-1, p1.y)==val1) && !visited.at<uchar>(p1.x-1, p1.y) )
		{
			stack[topIndex++] = Point(p1.x-1, p1.y);
			continue;
		}
		else if( (p1.y>0) && (mat.at<ushort>(p1.x, p1.y-1)==val1) && !visited.at<uchar>(p1.x, p1.y-1) )
		{
			stack[topIndex++] = Point(p1.x, p1.y-1);
			continue;
		}
		else if( (p1.x<mat.rows-1) && (mat.at<ushort>(p1.x+1, p1.y)==val1) && !visited.at<uchar>(p1.x+1, p1.y) )
		{
			stack[topIndex++] = Point(p1.x+1, p1.y);
			continue;
		}
		else if( (p1.y<mat.cols-1) && (mat.at<ushort>(p1.x, p1.y+1)==val1) && !visited.at<uchar>(p1.x, p1.y+1) )
		{
			stack[topIndex++] = Point(p1.x, p1.y+1);
			continue;
		}
		else if( (p1.x>0 && p1.y>0) && (mat.at<ushort>(p1.x-1, p1.y-1)==val1) && !visited.at<uchar>(p1.x-1, p1.y-1) )
		{
			stack[topIndex++] = Point(p1.x-1, p1.y-1);
			continue;
		}
		else if( (p1.x>0 && p1.y<mat.cols-1) && (mat.at<ushort>(p1.x-1, p1.y+1)==val1) && !visited.at<uchar>(p1.x-1, p1.y+1) )
		{
			stack[topIndex++] = Point(p1.x-1, p1.y+1);
			continue;
		}
		else if( (p1.x<mat.rows-1 && p1.y>0) && (mat.at<ushort>(p1.x+1, p1.y-1)==val1) && !visited.at<uchar>(p1.x+1, p1.y-1) )
		{
			stack[topIndex++] = Point(p1.x+1, p1.y-1);
			continue;
		}
		else if( (p1.x<mat.rows-1 && p1.y<mat.cols-1) && (mat.at<ushort>(p1.x+1, p1.y+1)==val1) && !visited.at<uchar>(p1.x+1, p1.y+1) )
		{
			stack[topIndex++] = Point(p1.x+1, p1.y+1);
			continue;
		}
		else
			--topIndex;
	}
	return (num+1)/2;
}

/**
 * @brief findRegion: use DFS to label a foreground object with a unique index
 *
 * @param mask: the binary image of foreground
 * @param label: the matrix to store each object's label
 * @param p: we start searching at this point
 * @param objIndex: the index of this object
 *
 * @return: number of points in this foreground object
 */
static long findRegion(Mat mask,
		Mat label,
		Point p,
		int objIndex)
{
	int topIndex = 0;
	long num = 0;
	Mat visited = Mat::zeros(mask.size(), CV_8UC1);

	CV_Assert(mask.type() == CV_8UC1);

	if( mask.at<uchar>(p.x, p.y) == 0 )
		return 0;

	visited.at<uchar>(p.x, p.y) = 255;
	label.at<uchar>(p.x, p.y) = objIndex;
	stack[topIndex++] = p;
	while( topIndex )
	{
		Point p1 = stack[topIndex-1];
		visited.at<uchar>(p1.x, p1.y) = 255;
		label.at<uchar>(p1.x, p1.y) = objIndex;
		++num;
		// cerr << "topIndex = " << topIndex << ", (" << p1.x << "," << p1.y << ")" << endl;
		if( (p1.x>0) && mask.at<uchar>(p1.x-1, p1.y) && !visited.at<uchar>(p1.x-1, p1.y) )
		{
			stack[topIndex++] = Point(p1.x-1, p1.y);
			continue;
		}
		else if( (p1.y>0) && mask.at<uchar>(p1.x, p1.y-1) && !visited.at<uchar>(p1.x, p1.y-1) )
		{
			stack[topIndex++] = Point(p1.x, p1.y-1);
			continue;
		}
		else if( (p1.x<mask.rows-1) && mask.at<uchar>(p1.x+1, p1.y) && !visited.at<uchar>(p1.x+1, p1.y) )
		{
			stack[topIndex++] = Point(p1.x+1, p1.y);
			continue;
		}
		else if( (p1.y<mask.cols-1) && mask.at<uchar>(p1.x, p1.y+1) && !visited.at<uchar>(p1.x, p1.y+1) )
		{
			stack[topIndex++] = Point(p1.x, p1.y+1);
			continue;
		}
		else if( (p1.x>0 && p1.y>0) && mask.at<uchar>(p1.x-1, p1.y-1) && !visited.at<uchar>(p1.x-1, p1.y-1) )
		{
			stack[topIndex++] = Point(p1.x-1, p1.y-1);
			continue;
		}
		else if( (p1.x>0 && p1.y<mask.cols-1) && mask.at<uchar>(p1.x-1, p1.y+1) && !visited.at<uchar>(p1.x-1, p1.y+1) )
		{
			stack[topIndex++] = Point(p1.x-1, p1.y+1);
			continue;
		}
		else if( (p1.x<mask.rows-1 && p1.y>0) && mask.at<uchar>(p1.x+1, p1.y-1) && !visited.at<uchar>(p1.x+1, p1.y-1) )
		{
			stack[topIndex++] = Point(p1.x+1, p1.y-1);
			continue;
		}
		else if( (p1.x<mask.rows-1 && p1.y<mask.cols-1) && mask.at<uchar>(p1.x+1, p1.y+1) && !visited.at<uchar>(p1.x+1, p1.y+1) )
		{
			stack[topIndex++] = Point(p1.x+1, p1.y+1);
			continue;
		}
		else
			--topIndex;
	}
	return (num+1)/2;
}

/**
 * @brief findLGC: find a local gradient constancy
 *
 * @param objLabel: objects mask
 * @param lumRatio: luminance ratio matrix
 * @param lgcLabel: local gradient constancy matrix
 * @param p: start point
 * @param lgcIndex: lgc index
 * @param mgThr: the minimum gradient threshold
 * @param threshold1: the low threshold of the pixel in luminance ratio
 * @param threshold2: the high threshold of the pixel in luminance ratio
 *
 * @return: number of points in this lgc
 */
static long findLGC(Mat objLabel,
		Mat lumRatio,
		Mat lgcLabel,
		Point p,
		int lgcIndex,
		Vec3f mgThr,
		float threshold1 = 1,
		float threshold2 = 5)
{
	int topIndex = 0;
	long num = 0;
	Mat visited = Mat::zeros(objLabel.size(), CV_8UC1);

	if( objLabel.at<uchar>(p.x, p.y) == 0 )
		return 0;

	visited.at<uchar>(p.x, p.y) = 255;
	lgcLabel.at<ushort>(p.x, p.y) = lgcIndex;
	stack[topIndex++] = p;
	while( topIndex )
	{
		Point p1 = stack[topIndex-1];
		visited.at<uchar>(p1.x, p1.y) = 255;
		lgcLabel.at<ushort>(p1.x, p1.y) = lgcIndex;
		++num;
		if( (p1.x>0) && objLabel.at<uchar>(p1.x-1, p1.y) && !visited.at<uchar>(p1.x-1, p1.y) )
		{
			float b1, g1, r1, b2, g2, r2;
			b1 = lumRatio.at<Vec3f>(p1.x, p1.y)[0];
			b2 = lumRatio.at<Vec3f>(p1.x-1, p1.y)[0];
			g1 = lumRatio.at<Vec3f>(p1.x, p1.y)[1];
			g2 = lumRatio.at<Vec3f>(p1.x-1, p1.y)[1];
			r1 = lumRatio.at<Vec3f>(p1.x, p1.y)[2];
			r2 = lumRatio.at<Vec3f>(p1.x-1, p1.y)[2];
			if( mgThr[0] > abs(b1-b2) &&
					mgThr[1] > abs(g1-g2) &&
					mgThr[2] > abs(r1-r2) &&
					b2 < threshold2 && g2 < threshold2 && r2 < threshold2 &&
					b2 > threshold1 && g2 > threshold1 && r2 > threshold1 )
			{
				stack[topIndex++] = Point(p1.x-1, p1.y);
				continue;
			}
		}
		if( (p1.y>0) && objLabel.at<uchar>(p1.x, p1.y-1) && !visited.at<uchar>(p1.x, p1.y-1) )
		{
			float b1, g1, r1, b2, g2, r2;
			b1 = lumRatio.at<Vec3f>(p1.x, p1.y)[0];
			b2 = lumRatio.at<Vec3f>(p1.x, p1.y-1)[0];
			g1 = lumRatio.at<Vec3f>(p1.x, p1.y)[1];
			g2 = lumRatio.at<Vec3f>(p1.x, p1.y-1)[1];
			r1 = lumRatio.at<Vec3f>(p1.x, p1.y)[2];
			r2 = lumRatio.at<Vec3f>(p1.x, p1.y-1)[2];
			if(mgThr[0] > abs(b1-b2) &&
					mgThr[1] > abs(g1-g2) &&
					mgThr[2] > abs(r1-r2) &&
					b2 < threshold2 && g2 < threshold2 && r2 < threshold2 &&
					b2 > threshold1 && g2 > threshold1 && r2 > threshold1 )
			{
				stack[topIndex++] = Point(p1.x, p1.y-1);
				continue;
			}
		}
		if( (p1.x<objLabel.rows-1) && objLabel.at<uchar>(p1.x+1, p1.y) && !visited.at<uchar>(p1.x+1, p1.y) )
		{
			float b1, g1, r1, b2, g2, r2;
			b1 = lumRatio.at<Vec3f>(p1.x, p1.y)[0];
			b2 = lumRatio.at<Vec3f>(p1.x+1, p1.y)[0];
			g1 = lumRatio.at<Vec3f>(p1.x, p1.y)[1];
			g2 = lumRatio.at<Vec3f>(p1.x+1, p1.y)[1];
			r1 = lumRatio.at<Vec3f>(p1.x, p1.y)[2];
			r2 = lumRatio.at<Vec3f>(p1.x+1, p1.y)[2];
			if(mgThr[0] > abs(b1-b2) &&
					mgThr[1] > abs(g1-g2) &&
					mgThr[2] > abs(r1-r2) &&
					b2 < threshold2 && g2 < threshold2 && r2 < threshold2 &&
					b2 > threshold1 && g2 > threshold1 && r2 > threshold1 )
			{
				stack[topIndex++] = Point(p1.x+1, p1.y);
				continue;
			}
		}
		if( (p1.y<objLabel.cols-1) && objLabel.at<uchar>(p1.x, p1.y+1) && !visited.at<uchar>(p1.x, p1.y+1) )
		{
			float b1, g1, r1, b2, g2, r2;
			b1 = lumRatio.at<Vec3f>(p1.x, p1.y)[0];
			b2 = lumRatio.at<Vec3f>(p1.x, p1.y+1)[0];
			g1 = lumRatio.at<Vec3f>(p1.x, p1.y)[1];
			g2 = lumRatio.at<Vec3f>(p1.x, p1.y+1)[1];
			r1 = lumRatio.at<Vec3f>(p1.x, p1.y)[2];
			r2 = lumRatio.at<Vec3f>(p1.x, p1.y+1)[2];
			if(mgThr[0] > abs(b1-b2) &&
					mgThr[1] > abs(g1-g2) &&
					mgThr[2] > abs(r1-r2) &&
					b2 < threshold2 && g2 < threshold2 && r2 < threshold2 &&
					b2 > threshold1 && g2 > threshold1 && r2 > threshold1 )
			{
				stack[topIndex++] = Point(p1.x, p1.y+1);
				continue;
			}
		}
#if 0
		if( (p1.x>0 && p1.y>0) && objLabel.at<uchar>(p1.x-1, p1.y-1) && !visited.at<uchar>(p1.x-1, p1.y-1) )
		{
			float b1, g1, r1, b2, g2, r2;
			b1 = lumRatio.at<Vec3f>(p1.x, p1.y)[0];
			b2 = lumRatio.at<Vec3f>(p1.x-1, p1.y-1)[0];
			g1 = lumRatio.at<Vec3f>(p1.x, p1.y)[1];
			g2 = lumRatio.at<Vec3f>(p1.x-1, p1.y-1)[1];
			r1 = lumRatio.at<Vec3f>(p1.x, p1.y)[2];
			r2 = lumRatio.at<Vec3f>(p1.x-1, p1.y-1)[2];
			if(mgThr[0] > abs(b1-b2) &&
					mgThr[1] > abs(g1-g2) &&
					mgThr[2] > abs(r1-r2) &&
					b2 < threshold2 && g2 < threshold2 && r2 < threshold2 &&
					b2 > threshold1 && g2 > threshold1 && r2 > threshold1 )
			{
				stack[topIndex++] = Point(p1.x-1, p1.y-1);
				continue;
			}
		}
		if( (p1.x>0 && (p1.y<objLabel.cols-1)) && objLabel.at<uchar>(p1.x-1, p1.y+1) && !visited.at<uchar>(p1.x-1, p1.y+1) )
		{
			float b1, g1, r1, b2, g2, r2;
			b1 = lumRatio.at<Vec3f>(p1.x, p1.y)[0];
			b2 = lumRatio.at<Vec3f>(p1.x-1, p1.y+1)[0];
			g1 = lumRatio.at<Vec3f>(p1.x, p1.y)[1];
			g2 = lumRatio.at<Vec3f>(p1.x-1, p1.y+1)[1];
			r1 = lumRatio.at<Vec3f>(p1.x, p1.y)[2];
			r2 = lumRatio.at<Vec3f>(p1.x-1, p1.y+1)[2];
			if(mgThr[0] > abs(b1-b2) &&
					mgThr[1] > abs(g1-g2) &&
					mgThr[2] > abs(r1-r2) &&
					b2 < threshold2 && g2 < threshold2 && r2 < threshold2 &&
					b2 > threshold1 && g2 > threshold1 && r2 > threshold1 )
			{
				stack[topIndex++] = Point(p1.x-1, p1.y+1);
				continue;
			}
		}
		if( ((p1.x<objLabel.rows-1) && p1.y>0) && objLabel.at<uchar>(p1.x+1, p1.y-1) && !visited.at<uchar>(p1.x+1, p1.y-1) )
		{
			float b1, g1, r1, b2, g2, r2;
			b1 = lumRatio.at<Vec3f>(p1.x, p1.y)[0];
			b2 = lumRatio.at<Vec3f>(p1.x+1, p1.y-1)[0];
			g1 = lumRatio.at<Vec3f>(p1.x, p1.y)[1];
			g2 = lumRatio.at<Vec3f>(p1.x+1, p1.y-1)[1];
			r1 = lumRatio.at<Vec3f>(p1.x, p1.y)[2];
			r2 = lumRatio.at<Vec3f>(p1.x+1, p1.y-1)[2];
			if(mgThr[0] > abs(b1-b2) &&
					mgThr[1] > abs(g1-g2) &&
					mgThr[2] > abs(r1-r2) &&
					b2 < threshold2 && g2 < threshold2 && r2 < threshold2 &&
					b2 > threshold1 && g2 > threshold1 && r2 > threshold1 )
			{
				stack[topIndex++] = Point(p1.x+1, p1.y-1);
				continue;
			}
		}
		if( ((p1.x<objLabel.rows-1) && (p1.y<objLabel.cols-1)) && objLabel.at<uchar>(p1.x+1, p1.y+1) && !visited.at<uchar>(p1.x+1, p1.y+1) )
		{
			float b1, g1, r1, b2, g2, r2;
			b1 = lumRatio.at<Vec3f>(p1.x, p1.y)[0];
			b2 = lumRatio.at<Vec3f>(p1.x+1, p1.y+1)[0];
			g1 = lumRatio.at<Vec3f>(p1.x, p1.y)[1];
			g2 = lumRatio.at<Vec3f>(p1.x+1, p1.y+1)[1];
			r1 = lumRatio.at<Vec3f>(p1.x, p1.y)[2];
			r2 = lumRatio.at<Vec3f>(p1.x+1, p1.y+1)[2];
			if(mgThr[0] > abs(b1-b2) &&
					mgThr[1] > abs(g1-g2) &&
					mgThr[2] > abs(r1-r2) &&
					b2 < threshold2 && g2 < threshold2 && r2 < threshold2 &&
					b2 > threshold1 && g2 > threshold1 && r2 > threshold1 )
			{
				stack[topIndex++] = Point(p1.x+1, p1.y+1);
				continue;
			}
		}
#endif
		--topIndex;
	}
	return (num+1)/2;
}

MCSS::MCSS()
{
	nframes = 0;
	frameSize = Size(0, 0);
	frameType = CV_8UC3;

	threshold1 = LOW_LUMINANCE_RATIO_THRESHOLD;
	threshold2 = HIGH_LUMINANCE_RATIO_THRESHOLD;

	lambda_low = LOW_RELATIVE_SIZE_THRESHOLD;
	lambda_high = HIGH_RELATIVE_SIZE_THRESHOLD;
	tao = EXTRINSIC_TERMINAL_POINT_WEIGHT_THRESHOLD;
	alpha = ALPHA;
	v = 0.1;
	mgThrFixed = DEFAULT_MGTHR_FIXED;
	isMGThrFixed = true;
	hasFringe = false;
}

/**
 * @brief getParameters: get the current parameters of this model
 *
 * @return: a struct contains the current parameters
 */
MCSS_Param MCSS::getParameters()
{
	MCSS_Param p;

	p.lambda_low = lambda_low;
	p.lambda_high = lambda_high;
	p.threshold1 = threshold1;
	p.threshold2 = threshold2;
	p.tao = tao;
	p.alpha = alpha;
	p.mgThrFixed = mgThrFixed;
	p.isMGThrFixed = isMGThrFixed;
	p.hasFringe = hasFringe;

	return p;
}

/**
 * @brief setParameters: change some internal parameters in this model
 *
 * @param p: parameters you want to change
 */
void MCSS::setParameters(MCSS_Param p)
{
	lambda_low = p.lambda_low;
	lambda_high = p.lambda_high;
	threshold1 = p.threshold1;
	threshold2 = p.threshold2;
	tao = p.tao;
	alpha = p.alpha;
	mgThrFixed = p.mgThrFixed;
	isMGThrFixed = p.isMGThrFixed;
	hasFringe = p.hasFringe;
}

/**
 * @brief operator(): update the model
 *
 * @param current: the current frame
 * @param background: the background image
 * @param mask: mask image from any BS model
 * @param output: the final result mask image
 */
void MCSS::operator()(Mat current, Mat background, Mat mask, OutputArray output)
{
	Mat post_mask, tmp;
	int objNum = 0, lgcNum = 0;
	int i, j;

	CV_Assert(current.data != NULL);
	CV_Assert(background.data != NULL);
	CV_Assert(mask.data != NULL);
	CV_Assert(current.type() == CV_8UC3);
	CV_Assert(background.type() == CV_8UC3);
	CV_Assert(mask.type() == CV_8UC1);

	++nframes;
	if( nframes == 1 )
	{
		frameSize = mask.size();
		frameType = mask.type();

		mean = Mat::zeros(mask.size(), CV_32FC3);
		STD = Mat::zeros(mask.size(), CV_32FC3);
		cont = Mat::zeros(mask.size(), CV_8UC1);
		objLabel = Mat::zeros(mask.size(), CV_8UC1);
		lgcLabel = Mat::zeros(mask.size(), CV_16UC1);
		lumRatio = Mat::zeros(mask.size(), CV_32FC3);
		lgcImg = Mat::zeros(mask.size(), CV_8UC3);
		dst = Mat::zeros(mask.size(), CV_8UC1);
	}
	/* initialize everything */
	objLabel *= 0;
	lgcLabel *= 0;
	lumRatio *= 0;
	dst *= 0;
	lgcImg *= 0;
	mean_average_bg.clear();
	standard_deviation_average_bg.clear();
	mgThr.clear();
	objArea.clear();
	lgcArea.clear();
	tpw.clear();
	lgcIsShadow.clear();
	lgcToObj.clear();
	meanLGC.clear();

	cerr << endl << "frame " << nframes << endl;

	/* detect foreground objects */
	// threshold(mask, mask, 30, 255, CV_THRESH_BINARY);
	bool detectObj = true;
	tmp.create(mask.size(), CV_8UC1);
	tmp *= 0;
	for( i = 0 ; i < mask.rows && detectObj ; ++i )
	{
		for( j = 0 ; j < mask.cols && detectObj ; ++j )
		{
			/* 
			 * if point (i, j) belongs to background
			 * or has been labeled already, just
			 * ignore it
			 * */
			if( mask.at<uchar>(i, j) == 0 || objLabel.at<uchar>(i, j) != 0 || tmp.at<uchar>(i, j) != 0 )
				continue;
			else
				tmp.at<uchar>(i, j) = 1;
			if( objNum >= MAX_OBJ_NUM )
				detectObj = false;
			int area;
			area = findRegion(mask, objLabel, Point(i, j), objNum+1);
			/* eliminate small regions */
			if( area < MIN_OBJ_AREA )
			{
				cleanRegionU8(objLabel, Point(i, j), objNum+1, 0);
				continue;
			}
			++objNum;
			// cerr << "Get one object :" << objNum << endl << "Area: " << area << endl;
			objArea.push_back(area);
			mean_average_bg.push_back(0);
			standard_deviation_average_bg.push_back(0);
			mgThr.push_back(Vec3f(0, 0, 0));
		}
	}
	if( objNum == 0 )
	{
		mask.copyTo(output);
		return;
	}
	if( hasFringe )
	{
		/* 
		 * if the narrow bright fringe exist, try to avoid them 
		 * (F. Edge Noise Correction)
		 * */
		threshold(objLabel*2, mask, 1, 255, CV_THRESH_BINARY);
		// GaussianBlur(mask, mask, Size(5, 5), 0, 0);
		threshold(mask, mask, 10, 255, CV_THRESH_BINARY);
		erode(mask, post_mask, getStructuringElement(MORPH_RECT, Size(5, 5)), Point(-1, -1), 1);
		threshold(post_mask, post_mask, 254, 255, CV_THRESH_BINARY);
	}
	else
		mask.copyTo(post_mask);
	objLabel &= post_mask;
	// cerr << "Get " << objNum << " foreground objects" << endl;


	/* 
	 * we don't need to calculate the Minimum gradient threshold
	 * in most occasions, but we provide the implementation in paper
	 * (Part IV. EXPERIMENTAL RESULTS, A. Parameter Analysis, (1)Minimum gradient threshold)
	 * */
	if( !isMGThrFixed )
	{
		/* calculate the mean value(time sequence), variance of each object */
		for( i = 0 ; i < objLabel.rows ; ++i )
		{
			for( j = 0 ; j < objLabel.cols ; ++j )
			{
				if( objLabel.at<uchar>(i, j) != 0 )
					continue;
				uchar b, g, r;
				int n = cont.at<uchar>(i, j)+1;
				if( n < HISTORY )
					cont.at<uchar>(i, j) += 1;
				b = current.at<Vec3b>(i, j)[0];
				g = current.at<Vec3b>(i, j)[1];
				r = current.at<Vec3b>(i, j)[2];
				mean.at<Vec3f>(i, j)[0] = (n-1)*(mean.at<Vec3f>(i, j)[0])/n+b/n;
				mean.at<Vec3f>(i, j)[1] = (n-1)*(mean.at<Vec3f>(i, j)[1])/n+g/n;
				mean.at<Vec3f>(i, j)[2] = (n-1)*(mean.at<Vec3f>(i, j)[2])/n+r/n;
				STD.at<Vec3f>(i, j)[0] = (n-1)*STD.at<Vec3f>(i, j)[0]/n+abs(b-mean.at<Vec3f>(i, j)[0])/n;
				STD.at<Vec3f>(i, j)[1] = (n-1)*STD.at<Vec3f>(i, j)[1]/n+abs(g-mean.at<Vec3f>(i, j)[1])/n;
				STD.at<Vec3f>(i, j)[2] = (n-1)*STD.at<Vec3f>(i, j)[2]/n+abs(r-mean.at<Vec3f>(i, j)[2])/n;
			}
		}


		/* calculate the sum of each pixel's mean and standard deviation */
		for( i = 0 ; i < post_mask.rows ; ++i )
		{
			for( j = 0 ; j < post_mask.cols ; ++j )
			{
				if( objLabel.at<uchar>(i, j) == 0 )
					continue;
				mean_average_bg[objLabel.at<uchar>(i, j)-1][0] += mean.at<Vec3f>(i, j)[0];
				mean_average_bg[objLabel.at<uchar>(i, j)-1][1] += mean.at<Vec3f>(i, j)[1];
				mean_average_bg[objLabel.at<uchar>(i, j)-1][2] += mean.at<Vec3f>(i, j)[2];
				standard_deviation_average_bg[objLabel.at<uchar>(i, j)-1][0] += STD.at<Vec3f>(i, j)[0];
				standard_deviation_average_bg[objLabel.at<uchar>(i, j)-1][1] += STD.at<Vec3f>(i, j)[1];
				standard_deviation_average_bg[objLabel.at<uchar>(i, j)-1][2] += STD.at<Vec3f>(i, j)[2];
			}
		}
		for( i = 0 ; i < (int)mean_average_bg.size() ; ++i )
		{
			mean_average_bg[i][0] /= objArea[i];
			mean_average_bg[i][1] /= objArea[i];
			mean_average_bg[i][2] /= objArea[i];
			standard_deviation_average_bg[i][0] /= objArea[i];
			standard_deviation_average_bg[i][1] /= objArea[i];
			standard_deviation_average_bg[i][2] /= objArea[i];
		}
	}


	/* get the minimum gradient threshold for each object */
	for( i = 0 ; i < (int)mgThr.size() ; ++i )
	{
		if( !isMGThrFixed )
		{
			mgThr[i][0] = alpha*mean_average_bg[i][0]*standard_deviation_average_bg[i][0];
			mgThr[i][1] = alpha*mean_average_bg[i][1]*standard_deviation_average_bg[i][1];
			mgThr[i][2] = alpha*mean_average_bg[i][2]*standard_deviation_average_bg[i][2];
		}
		else
			mgThr[i] = mgThrFixed;
	}

#ifdef debug
	for( i = 0 ; i < (int)mgThr.size() ; ++i )
	{
		cerr << "mean_average_bg[" << i << "] = " << mean_average_bg[i][0] << " " << mean_average_bg[i][1] << " " << mean_average_bg[i][2] << endl;
		cerr << "standard_deviation_average_bg[" << i << "] = " << standard_deviation_average_bg[i][0] << " " << standard_deviation_average_bg[i][1] << " " << standard_deviation_average_bg[i][2] << endl;
		cerr << "mgThr[" << i << "] = " << mgThr[i][0] << " " << mgThr[i][1] << " " << mgThr[i][2] << endl;
	}
#endif

	/* 
	 * get the luminance ratio
	 * (Part III. MOVING SHADOW DETECTION ,C. Regions with Local Color Constancy)
	 * */
	post_mask.copyTo(dst);
	for( i = 0 ; i < post_mask.rows ; ++i )
	{
		for( j = 0 ; j < post_mask.cols ; ++j )
		{
			if( post_mask.at<uchar>(i, j) != 255 )
			{
				// dst.at<uchar>(i, j) = 1;
				continue;
			}
			lumRatio.at<Vec3f>(i, j)[0] = (background.at<Vec3b>(i, j)[0]+v)/(current.at<Vec3b>(i, j)[0]+v);
			lumRatio.at<Vec3f>(i, j)[1] = (background.at<Vec3b>(i, j)[1]+v)/(current.at<Vec3b>(i, j)[1]+v);
			lumRatio.at<Vec3f>(i, j)[2] = (background.at<Vec3b>(i, j)[2]+v)/(current.at<Vec3b>(i, j)[2]+v);
			/* Shadow like area */
			if( ((lumRatio.at<Vec3f>(i, j))[0] >= 1) &&
					((lumRatio.at<Vec3f>(i, j))[1] >= 1) &&
					((lumRatio.at<Vec3f>(i, j))[2] >= 1))
				dst.at<uchar>(i, j) = 127;
			/* foreground pixel */
			else
			{
				lumRatio.at<Vec3f>(i, j)[0] = 0;
				lumRatio.at<Vec3f>(i, j)[1] = 0;
				lumRatio.at<Vec3f>(i, j)[2] = 0;
				dst.at<uchar>(i, j) = 255;
			}
		}
	}
	// threshold(lumRatio, lumRatio, 50, 0, CV_THRESH_TOZERO_INV);

#ifdef debug
	tmp = lumRatio;
	dilate(tmp, tmp, Mat(), Point(-1, -1), 1);
	erode(tmp, tmp, Mat(), Point(-1, -1), 1);
	tmp.convertTo(tmp, CV_8UC3, 1, 0);
	imshow("lumRatio", tmp*30);
#endif
	/* search the local gradient constancy */
	bool detectLGC = true;
	for( i = 0 ; i < post_mask.rows && detectLGC ; ++i )
	{
		for( j = 0 ; j < post_mask.cols && detectLGC ; ++j )
		{
			if( objLabel.at<uchar>(i, j) == 0 || lgcLabel.at<ushort>(i, j) != 0 || lumRatio.at<Vec3f>(i, j) == Vec3f(0, 0, 0) )
				continue;
			if( mgThr[objLabel.at<uchar>(i, j)-1][0] == 0 &&  mgThr[objLabel.at<uchar>(i, j)-1][1] == 0 && mgThr[objLabel.at<uchar>(i, j)-1][2] == 0 )
			{
				lgcLabel.at<ushort>(i, j) = SMALL_LGC_LABEL;
				continue;
			}
			/* limit the max number of LGC regions */
			if( lgcNum >= MAX_LGC_NUM )
				detectLGC = false;
			int area;
			area = findLGC(objLabel, lumRatio, lgcLabel, Point(i, j), lgcNum+1, mgThr[objLabel.at<uchar>(i, j)-1], threshold1, threshold2);
			/* label each of the small region to a fixed number */
			if( area < MIN_LGC_AREA )
			{
				cleanRegionU16(lgcLabel, Point(i, j), lgcNum+1, SMALL_LGC_LABEL);
				continue;
			}
			++lgcNum;
			// cerr << "Get one lgc :" << lgcNum-1 << "Area: " << area << endl; 
			lgcArea.push_back(area);
			lgcToObj.push_back(objLabel.at<uchar>(i, j));
			meanLGC.push_back(0);
			tpw.push_back(0);
			lgcIsShadow.push_back(true);
		}
	}
	if( lgcNum == 0 )
	{
		post_mask.copyTo(output);
		return;
	}

#ifdef debug
	{
		/* draw the LGC area */
		for( i = 0 ; i < post_mask.rows && detectLGC ; ++i )
		{
			for( j = 0 ; j < post_mask.cols && detectLGC ; ++j )
			{
				if( lgcLabel.at<ushort>(i, j) == 0 )
				{
					lgcImg.at<Vec3b>(i, j)[0] = 0;
					lgcImg.at<Vec3b>(i, j)[1] = 0;
					lgcImg.at<Vec3b>(i, j)[2] = 0;
					continue;
				}
				else if( lgcLabel.at<ushort>(i, j) == SMALL_LGC_LABEL )
				{
					lgcImg.at<Vec3b>(i, j)[0] = 255;
					lgcImg.at<Vec3b>(i, j)[1] = 255;
					lgcImg.at<Vec3b>(i, j)[2] = 255;
					continue;
				}
				lgcImg.at<Vec3b>(i, j) = Vec3b( (lgcLabel.at<ushort>(i, j))%5*50+50, 205-(lgcLabel.at<ushort>(i, j))%5*50, (lgcLabel.at<ushort>(i, j))%5*50+50);
			}
		}
		imshow("lgcImg", lgcImg);
	}
	cerr << "Get " << lgcNum << " lgc regions" << endl;
#endif

	/* 
	 * calculate the mean value of each LGC
	 * (Part III. MOVING SHADOW DETECTION, E. Classification Process)
	 * */
	for( i = 0 ; i < post_mask.rows ; ++i )
	{
		for( j = 0 ; j < post_mask.cols ; ++j )
		{
			if( lgcLabel.at<ushort>(i, j) == 0 || lgcLabel.at<ushort>(i, j) == SMALL_LGC_LABEL )
				continue;
			float b = (lumRatio.at<Vec3f>(i, j))[0];
			float g = (lumRatio.at<Vec3f>(i, j))[1];
			float r = (lumRatio.at<Vec3f>(i, j))[2];
			meanLGC[lgcLabel.at<ushort>(i, j)-1][0] += b/lgcArea[lgcLabel.at<ushort>(i, j)-1];
			meanLGC[lgcLabel.at<ushort>(i, j)-1][1] += g/lgcArea[lgcLabel.at<ushort>(i, j)-1];
			meanLGC[lgcLabel.at<ushort>(i, j)-1][2] += r/lgcArea[lgcLabel.at<ushort>(i, j)-1];
		}
	}

	/*
	 * calculate the number of external terminal pixels and all terminal pixels
	 * (Part III. MOVING SHADOW DETECTION, E. Classification Process)
	 * */
	vector<int> external(lgcNum), all(lgcNum);
	for( i = 0 ; i < post_mask.rows ; ++i )
	{
		for( j = 0 ; j < post_mask.cols ; ++j )
		{
			int lgc, obj;
			lgc = lgcLabel.at<ushort>(i, j);
			obj = objLabel.at<uchar>(i, j);
			if( lgc == 0 || lgc == SMALL_LGC_LABEL )
				continue;
			if( (i>0 && (lgc != lgcLabel.at<ushort>(i-1, j))) ||
					(j>0 && (lgc != lgcLabel.at<ushort>(i, j-1)))	||
					(i<post_mask.rows-1 && (lgc != lgcLabel.at<ushort>(i+1, j))) ||
					(j<post_mask.cols-1 && (lgc != lgcLabel.at<ushort>(i, j+1))) ||
					((i>0 && (j<post_mask.cols-1)) && (lgc != lgcLabel.at<ushort>(i-1, j+1))) ||
					((i>0 && j>0) && (lgc != lgcLabel.at<ushort>(i-1, j-1))) ||
					(((i<post_mask.rows-1) && (j<post_mask.cols-1)) && (lgc != lgcLabel.at<ushort>(i+1, j+1))) ||
					(((i<post_mask.rows-1) && j>0) && (lgc != lgcLabel.at<ushort>(i+1, j-1))) )
			{
				++all[lgc-1];
				/* check external terminal pixels */
				if( (i>0 && (obj != objLabel.at<uchar>(i-1, j))) ||
						(j>0 && (obj != objLabel.at<uchar>(i, j-1))) ||
						((i<post_mask.rows-1) && (obj != objLabel.at<uchar>(i+1, j))) ||
						((j<post_mask.cols-1) && (obj != objLabel.at<uchar>(i, j+1))) ||
						((i>0 && (j<post_mask.cols-1)) && (obj != objLabel.at<uchar>(i-1, j+1))) ||
						((i>0 && j>0) && (obj != objLabel.at<uchar>(i-1, j-1))) ||
						(((i<post_mask.rows-1) && (j<post_mask.cols-1)) && (obj != objLabel.at<uchar>(i+1, j+1))) ||
						(((i<post_mask.rows-1) && j>0) && (obj != objLabel.at<uchar>(i+1, j-1))) )
					++external[lgc-1];
			}
		}
	}

	for( i = 0 ; i < lgcNum; ++i )
	{
		if( external[i] != 0 && all[i] != 0 )
			tpw[i] = 1.0*external[i]/all[i];
		else
			tpw[i] = 0;
#ifdef debug
		cerr << "meanLGC[" << i << "] = [" << meanLGC[i][0] << " " << meanLGC[i][1] << " " << meanLGC[i][2] << "]" << " tpw[" << i << "] = " << tpw[i] << "(external, all)" << external[i] << " " << all[i] << " lgcArea = " << lgcArea[i] << endl;
#endif
	}

	/* check if each lgc belongs to shadow */
	for( i = 0 ; i < lgcNum ; ++i )
	{
		lgcIsShadow[i] = true;
		if( (meanLGC[i][0] < 1) || (meanLGC[i][1] < 1) || (meanLGC[i][2] < 1))
		{
			// cerr << "lgc " << i << " is shadow = false: " << 1 << endl;
			lgcIsShadow[i] = false;
		}
		else if( tpw[i] < tao )
		{
			// cerr << "lgc " << i << " is shadow = false: " << 2 << endl;
			lgcIsShadow[i] = false;
		}
		else if( lgcArea[i] < objArea[lgcToObj[i]-1]*lambda_low )
		{
			// cerr << "lgc " << i << " is shadow = false: " << 3 << " objArea = " << objArea[lgcToObj[i]-1] << endl;
			lgcIsShadow[i] = false;
		}
	}


	/* 
	 * check each pixel if it belongs to shadow
	 * (Part III. MOVING SHADOW DETECTION, E. Classification Process)
	 * */
	for( i = 0 ; i < post_mask.rows ; ++i )
	{
		for( j = 0 ; j < post_mask.cols ; ++j )
		{
			if( lgcLabel.at<ushort>(i, j) == SMALL_LGC_LABEL )
				dst.at<uchar>(i, j) = 255;
			if( lgcLabel.at<ushort>(i, j) == 0 || dst.at<uchar>(i, j) == 255 || dst.at<uchar>(i, j) == 0 )
				continue;
			if( !lgcIsShadow[lgcLabel.at<ushort>(i, j)-1] ) 
			{
				dst.at<uchar>(i, j) = 255; 
			}
		}
	}

#if 0
	if( hasFringe )
	{
		tmp = Mat::zeros(dst.size(), CV_8UC1);
		for( i = 0 ; i < post_mask.rows ; ++i )
		{
			for( j = 0 ; j < post_mask.cols ; ++j )
			{
				if( mask.at<uchar>(i, j) && !post_mask.at<uchar>(i, j) )
				{
					int val = getNearestVal(Point(i, j));
					tmp.at<uchar>(i, j) = val;
					continue;
				}
			}
		}
		dst += tmp;
	}
#endif
	dst.copyTo(output);
	// imshow("tmp", tmp);
}
