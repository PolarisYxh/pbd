#ifndef __CONFIGURATIONLOADER_H__
#define __CONFIGURATIONLOADER_H__

#include <iostream>
#include <vector>


using namespace std;


namespace Utilities
{
	typedef vector<string> ClothPathVector;
	typedef vector<unsigned int> PatternNumVector;
	typedef vector<vector<float>> ClothCoeffVector;
	typedef vector<vector<unsigned int>> CollisionPairsVector;      //用于保存需要碰撞的pattern对
	typedef vector<vector<float>> TranslateVector;
	typedef vector<vector<float>> ScaleVector;
	typedef vector<bool> FindSpecialSeamVector;
	typedef unsigned int ZeroMassPointsNum;
	typedef vector<unsigned int> ZeroMassPointsVector;

	class Configuration
	{
	public:
		Configuration& operator=(const Configuration& other);
		Configuration();
		virtual ~Configuration();

	protected:

		//服装
		unsigned int m_clothNum;                                    //服装数量
		ClothPathVector m_clothPath;                                //服装obj文件路径
		PatternNumVector m_patternNum;                              //服装pattern数量
		ClothCoeffVector m_clothCoeff;                              //服装个pattern参数
		CollisionPairsVector m_collisonPairs;                        //需要进行碰撞检测的pattern对
		TranslateVector m_translate;                                //位移
		ScaleVector m_scale;                                        //缩放
		FindSpecialSeamVector m_findSpecialSeam;                    //是否找specialSeam
		ZeroMassPointsNum m_zeroMassPointsNum;
		ZeroMassPointsVector m_zeroMassPointsVector;

		//人体
		string m_bodyPath;
		string m_bvhPath;
		string m_bvhName;
		unsigned int m_startFrame;
		unsigned int m_currentFrame;
		unsigned int m_endFrame;
		unsigned int m_stepSize;
		unsigned int m_stepCount;

		//视角
		float m_zoom;
		float m_swingAngle;
		float m_elevateAngle;
		vector<float> m_center;

		bool m_isValid;

	public:

		void reset();
		Configuration getConfiguration(string filePath);
		void stringSplit(string& sourceStr, vector<string>& resultStr, const string& seperator);     //一个字符进行分割（字符串无空格）
		void stringSplitByBlank(string& sourceStr, vector<string>& resultStr);     //通过空格分割
		void reduceFrontBrackets(string& str);                                 //去除string前的空格或tab

		//服装
		void setClothNum(unsigned int clothNum);
		void setClothPath(string clothPath);
		void setPatternNum(unsigned int patternNum);
		void setClothCoeff(vector<float> clothCoeff);
		void setCollisionPairs(vector<unsigned int> collisionPairs);
		void setTranslate(vector<float> translate);
		void setScale(vector<float> scale);
		void setFindSpecialSeam(bool findSpecialSeam);
		void setZeroMassPointsNum(unsigned int zeroMassPointsNum);
		void setZeroMessPoints(unsigned int zeroMassPoints);

		unsigned int getClothNum();
		ClothPathVector getClothPath();
		PatternNumVector getPatternNum();
		ClothCoeffVector getClothCoeff();
		CollisionPairsVector getCollisionPairs();
		TranslateVector getTranslate();
		ScaleVector getScale();
		FindSpecialSeamVector getFindSpecialSeam();
		ZeroMassPointsNum getZeroMassPointsNum();
		ZeroMassPointsVector getZeroMassPointsVector();

		//人体
		void setBodyPath(string bodyPath);
		void setBvhPath(string bvhPath);
		void setBvhName(string bvhName);
		void setStartFrame(unsigned int startFrame);
		void setCurrentFrame(unsigned int currentFrame);
		void setEndFrame(unsigned int endFrame);
		void setStepSize(unsigned int stepSize);
		void setStepCount(unsigned int stepCount);

		string getBodyPath();
		string getBvhPath();
		string getBvhName();
		unsigned int getStartFrame();
		unsigned int getCurrentFrame();
		unsigned int getEndFrame();
		unsigned int getStepSize();
		unsigned int getStepCount();

		//视角
		void setZoom(float zoom);
		void setSwingAngle(float swingAngle);
		void setElevateAngle(float elevateAngle);
		void setCenter(vector<float> center);

		float getZoom();
		float getSwingAngle();
		float getElevateAngle();
		vector<float> getCenter();

		void setExtractValue(bool b);
		bool isExtractValid();
	};

}

#endif 
