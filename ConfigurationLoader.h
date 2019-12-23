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
	typedef vector<vector<unsigned int>> CollisionPairsVector;      //���ڱ�����Ҫ��ײ��pattern��
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

		//��װ
		unsigned int m_clothNum;                                    //��װ����
		ClothPathVector m_clothPath;                                //��װobj�ļ�·��
		PatternNumVector m_patternNum;                              //��װpattern����
		ClothCoeffVector m_clothCoeff;                              //��װ��pattern����
		CollisionPairsVector m_collisonPairs;                        //��Ҫ������ײ����pattern��
		TranslateVector m_translate;                                //λ��
		ScaleVector m_scale;                                        //����
		FindSpecialSeamVector m_findSpecialSeam;                    //�Ƿ���specialSeam
		ZeroMassPointsNum m_zeroMassPointsNum;
		ZeroMassPointsVector m_zeroMassPointsVector;

		//����
		string m_bodyPath;
		string m_bvhPath;
		string m_bvhName;
		unsigned int m_startFrame;
		unsigned int m_currentFrame;
		unsigned int m_endFrame;
		unsigned int m_stepSize;
		unsigned int m_stepCount;

		//�ӽ�
		float m_zoom;
		float m_swingAngle;
		float m_elevateAngle;
		vector<float> m_center;

		bool m_isValid;

	public:

		void reset();
		Configuration getConfiguration(string filePath);
		void stringSplit(string& sourceStr, vector<string>& resultStr, const string& seperator);     //һ���ַ����зָ�ַ����޿ո�
		void stringSplitByBlank(string& sourceStr, vector<string>& resultStr);     //ͨ���ո�ָ�
		void reduceFrontBrackets(string& str);                                 //ȥ��stringǰ�Ŀո��tab

		//��װ
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

		//����
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

		//�ӽ�
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
