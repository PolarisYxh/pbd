#include "ConfigurationLoader.h"
#include <iostream>
#include <vector>
#include <map>
#include <fstream>
#include <string>
#include <sstream>

using namespace std;
using namespace Utilities;

Configuration& Utilities::Configuration::operator=(const Configuration& other)
{
	m_clothNum = other.m_clothNum;
	m_clothPath = other.m_clothPath;
	m_patternNum = other.m_patternNum;
	m_clothCoeff = other.m_clothCoeff;
	m_collisonPairs = other.m_collisonPairs;
	m_translate = other.m_translate;
	m_scale = other.m_scale;
	m_findSpecialSeam = other.m_findSpecialSeam;
	m_zeroMassPointsNum = other.m_zeroMassPointsNum;
	m_zeroMassPointsVector = other.m_zeroMassPointsVector;

	m_bodyPath = other.m_bodyPath;
	m_bvhPath = other.m_bvhPath;
	m_bvhName = other.m_bvhName;
	m_startFrame = other.m_startFrame;
	m_currentFrame = other.m_currentFrame;
	m_endFrame = other.m_endFrame;
	m_stepSize = other.m_stepSize;
	m_stepCount = other.m_stepCount;


	m_zoom = other.m_zoom;
	m_swingAngle = other.m_swingAngle;
	m_elevateAngle = other.m_elevateAngle;
	m_center = other.m_center;

	m_isValid = other.m_isValid;
	return *this;
}

Configuration::Configuration()
{
	m_bodyPath = "";
}

Configuration::~Configuration()
{
}

void Utilities::Configuration::reset()
{
	m_clothNum = 0;
	m_clothPath.clear();
	//m_clothPath.push_back("");
	m_patternNum.clear();
	m_clothCoeff.clear();
	m_collisonPairs.clear();
	m_translate.clear();
	m_scale.clear();
	m_findSpecialSeam.clear();
	m_zeroMassPointsNum = 0;
	m_zeroMassPointsVector.clear();
	m_bodyPath = "";
	m_bvhPath = "";
	m_bvhName = "";
	m_startFrame = 0;
	m_currentFrame = 0;
	m_endFrame = 0;
	m_startFrame = 1;
	m_stepCount = -10;
	m_zoom = 0;
	m_swingAngle = 0;
	m_elevateAngle = 0;
	m_center.clear();
	m_isValid = true;
}

Configuration Utilities::Configuration::getConfiguration(string filePath)
{
	Configuration conf;
	conf.setExtractValue(true);
	//读取配置文件
	cout << "loading " << filePath << endl;
	ifstream filestream;
	filestream.open(filePath.c_str());
	if (filestream.fail())
	{
		cout << "Failed to open file:" << filePath << endl;
		//system("pause");
		conf.setExtractValue(false);
		return conf;
	}

	//clothNum
	string line_stream, str;
	getline(filestream, line_stream);
	getline(filestream, line_stream);
	vector<string> keyString;
	stringSplit(line_stream, keyString, "=");
	if (keyString.at(0) == "clothNum")
	{
		conf.setClothNum(atoi(keyString.at(1).c_str()));
	}

	//一次读取多件服装配置信息
	for (int i = 0; i < conf.getClothNum(); i++)
	{
		getline(filestream, line_stream);
		//clothPath
		getline(filestream, line_stream);
		stringSplit(line_stream, keyString, "=");
		if (keyString.at(0) == "clothPath")
		{
			conf.setClothPath(keyString.at(1));
		}

		//patternNum
		getline(filestream, line_stream);
		stringSplit(line_stream, keyString, "=");
		if (keyString.at(0) == "patternNum")
		{
			conf.setPatternNum(atoi(keyString.at(1).c_str()));
		}

		//clothCoeff
		getline(filestream, line_stream);
		reduceFrontBrackets(line_stream);
		if (line_stream == "clothCoeff=")
		{
			getline(filestream, line_stream);
			for (int j = 0; j < conf.getPatternNum().at(i); j++)
			{
				getline(filestream, line_stream);
				stringSplitByBlank(line_stream, keyString);
				vector<float> patternCoeff;               //每个pattern的参数
				for (int k = 0; k < keyString.size(); k++)
				{
					patternCoeff.push_back(atof(keyString.at(k).c_str()));
				}
				conf.setClothCoeff(patternCoeff);
			}
			getline(filestream, line_stream);
		}

		//translate
		getline(filestream, line_stream);
		stringSplit(line_stream, keyString, "=");
		if (keyString.at(0) == "translate")
		{
			getline(filestream, line_stream);
			stringSplitByBlank(line_stream, keyString);
			vector<float> translate;
			translate.push_back(atof(keyString.at(0).c_str()));
			translate.push_back(atof(keyString.at(1).c_str()));
			translate.push_back(atof(keyString.at(2).c_str()));
			conf.setTranslate(translate);
			getline(filestream, line_stream);
		}

		//scale
		getline(filestream, line_stream);
		stringSplit(line_stream, keyString, "=");
		if (keyString.at(0) == "scale")
		{
			getline(filestream, line_stream);
			stringSplitByBlank(line_stream, keyString);
			vector<float> scale;
			scale.push_back(atof(keyString.at(0).c_str()));
			scale.push_back(atof(keyString.at(1).c_str()));
			scale.push_back(atof(keyString.at(2).c_str()));
			conf.setScale(scale);
			getline(filestream, line_stream);
		}

		//findSpecialSeam
		getline(filestream, line_stream);
		stringSplit(line_stream, keyString, "=");
		if (keyString.at(0) == "findSpecialSeam")
		{
			bool b;
			stringstream str_stream1;
			istringstream(keyString.at(1)) >> boolalpha >> b;
			conf.setFindSpecialSeam(b);
		}
		getline(filestream, line_stream);
	}

	getline(filestream, line_stream);
	//zeroMassPointsNum
	getline(filestream, line_stream);
	stringSplit(line_stream, keyString, "=");
	if (keyString.at(0) == "zeroMassPointsNum")
	{
		conf.setZeroMassPointsNum(atoi(keyString.at(1).c_str()));
	}

	//zeroMassPoints
	if (conf.getZeroMassPointsNum() != 0)
	{
		getline(filestream, line_stream);
		reduceFrontBrackets(line_stream);
		if (line_stream == "zeroMassPoints=[")
		{
			getline(filestream, line_stream);
			stringSplitByBlank(line_stream, keyString);
			for (int k = 0; k < keyString.size(); k++)
			{
				conf.setZeroMessPoints(atoi(keyString.at(k).c_str()));
			}
		}
		getline(filestream, line_stream);
	}
	//collisionPairs
	getline(filestream, line_stream);
	reduceFrontBrackets(line_stream);
	if (line_stream == "collisionPairs=")
	{
		getline(filestream, line_stream);
		getline(filestream, line_stream);
		reduceFrontBrackets(line_stream);
		while (line_stream.size() > 1)
		{
			stringSplitByBlank(line_stream, keyString);
			vector<unsigned int> collisionPairs;
			collisionPairs.push_back(atoi(keyString.at(0).c_str()));
			collisionPairs.push_back(atoi(keyString.at(1).c_str()));
			conf.setCollisionPairs(collisionPairs);
			getline(filestream, line_stream);
			reduceFrontBrackets(line_stream);
		}
	}
	getline(filestream, line_stream);

	//视角
	getline(filestream, line_stream);
	//zoom
	getline(filestream, line_stream);
	stringSplit(line_stream, keyString, "=");
	if (keyString.at(0) == "zoom")
	{
		conf.setZoom(atof(keyString.at(1).c_str()));
	}

	//swingAngle
	getline(filestream, line_stream);
	stringSplit(line_stream, keyString, "=");
	if (keyString.at(0) == "swingAngle")
	{
		conf.setSwingAngle(atof(keyString.at(1).c_str()));
	}

	//elevateAngle
	getline(filestream, line_stream);
	stringSplit(line_stream, keyString, "=");
	if (keyString.at(0) == "elevateAngle")
	{
		conf.setElevateAngle(atof(keyString.at(1).c_str()));
	}

	//center
	getline(filestream, line_stream);
	stringSplit(line_stream, keyString, "=");
	if (keyString.at(0) == "center")
	{
		getline(filestream, line_stream);
		stringSplitByBlank(line_stream, keyString);
		vector<float> center;
		center.push_back(atof(keyString.at(0).c_str()));
		center.push_back(atof(keyString.at(1).c_str()));
		center.push_back(atof(keyString.at(2).c_str()));
		conf.setCenter(center);
		getline(filestream, line_stream);
	}
	getline(filestream, line_stream);

	//人体信息配置
	getline(filestream, line_stream);
	reduceFrontBrackets(line_stream);
	if (line_stream == "[")
	{
		//bodyPath
		getline(filestream, line_stream);
		stringSplit(line_stream, keyString, "=");
		if (keyString.at(0) == "bodyPath")
		{
			conf.setBodyPath(keyString.at(1));
		}

		//bvhPath
		getline(filestream, line_stream);
		stringSplit(line_stream, keyString, "=");
		if (keyString.at(0) == "bvhPath")
		{
			conf.setBvhPath(keyString.at(1));
		}

		//bvhName
		getline(filestream, line_stream);
		stringSplit(line_stream, keyString, "=");
		if (keyString.at(0) == "bvhName")
		{
			if(keyString.size()>1)
				conf.setBvhName(keyString.at(1));
			else
			{
				conf.setBvhPath("");
			}
		}

		//startFrame
		getline(filestream, line_stream);
		stringSplit(line_stream, keyString, "=");
		if (keyString.at(0) == "startFrame")
		{
			conf.setStartFrame(atoi(keyString.at(1).c_str()));
		}

		//currentFrame
		getline(filestream, line_stream);
		stringSplit(line_stream, keyString, "=");
		if (keyString.at(0) == "currentFrame")
		{
			conf.setCurrentFrame(atoi(keyString.at(1).c_str()));
		}

		//endFrame
		getline(filestream, line_stream);
		stringSplit(line_stream, keyString, "=");
		if (keyString.at(0) == "endFrame")
		{
			conf.setEndFrame(atoi(keyString.at(1).c_str()));
		}

		//stepSize
		getline(filestream, line_stream);
		stringSplit(line_stream, keyString, "=");
		if (keyString.at(0) == "stepSize")
		{
			conf.setStepSize(atoi(keyString.at(1).c_str()));
		}

		//stepCount
		getline(filestream, line_stream);
		stringSplit(line_stream, keyString, "=");
		if (keyString.at(0) == "stepCount")
		{
			conf.setStepCount(atoi(keyString.at(1).c_str()));
		}
		getline(filestream, line_stream);
	}
	getline(filestream, line_stream);
	if (line_stream != "}")
	{
		cout << "读取配置文件出错" << endl;
		conf.setExtractValue(false);
	}
	filestream.close();
	return conf;
}

void Utilities::Configuration::stringSplit(string & sourceStr, vector<string> & resultStr, const string & seperator)
{
	reduceFrontBrackets(sourceStr);
	resultStr.clear();

	string::size_type pos1, pos2;
	pos2 = sourceStr.find(seperator);
	pos1 = 0;
	while (string::npos != pos2)
	{
		resultStr.push_back(sourceStr.substr(pos1, pos2 - pos1));
		pos1 = pos2 + seperator.size();
		pos2 = sourceStr.find(seperator, pos1);
	}
	if (pos1 != sourceStr.length())
		resultStr.push_back(sourceStr.substr(pos1));
}

void Utilities::Configuration::stringSplitByBlank(string & sourceStr, vector<string> & resultStr)
{
	resultStr.clear();
	stringstream str_stream(sourceStr);
	string str;
	str_stream >> str;
	while (str != "")
	{
		resultStr.push_back(str);
		str.clear();
		str_stream >> str;
	}
}

void Utilities::Configuration::reduceFrontBrackets(string & str)
{
	int index;
	while (((str.find(' ')) == 0) || ((str.find("\t")) == 0))
	{
		str.erase(0, 1);
	}
}

void Utilities::Configuration::setClothNum(unsigned int clothNum)
{
	m_clothNum = clothNum;
}

void Utilities::Configuration::setClothPath(string clothPath)
{
	m_clothPath.push_back(clothPath);
}

void Utilities::Configuration::setPatternNum(unsigned int patternNum)
{
	m_patternNum.push_back(patternNum);
}

void Utilities::Configuration::setClothCoeff(vector<float> clothCoeff)
{
	m_clothCoeff.push_back(clothCoeff);
}

void Utilities::Configuration::setCollisionPairs(vector<unsigned int> collisionPairs)
{
	m_collisonPairs.push_back(collisionPairs);
}

void Utilities::Configuration::setTranslate(vector<float> translate)
{
	m_translate.resize(0);
	m_translate.push_back(translate);
}

void Utilities::Configuration::setScale(vector<float> scale)
{
	m_scale.push_back(scale);
}

void Utilities::Configuration::setFindSpecialSeam(bool findSpecialSeam)
{
	m_findSpecialSeam.push_back(findSpecialSeam);
}

void Utilities::Configuration::setZeroMassPointsNum(unsigned int zeroMassPointsNum)
{
	m_zeroMassPointsNum = zeroMassPointsNum;
}

void Utilities::Configuration::setZeroMessPoints(unsigned int zeroMassPoints)
{
	m_zeroMassPointsVector.push_back(zeroMassPoints);
}

unsigned int Utilities::Configuration::getClothNum()
{
	return m_clothNum;
}

ClothPathVector Utilities::Configuration::getClothPath()
{
	return m_clothPath;
}

PatternNumVector Utilities::Configuration::getPatternNum()
{
	return m_patternNum;
}

ClothCoeffVector Utilities::Configuration::getClothCoeff()
{
	return m_clothCoeff;
}

CollisionPairsVector Utilities::Configuration::getCollisionPairs()
{
	return m_collisonPairs;
}

TranslateVector Utilities::Configuration::getTranslate()
{
	return m_translate;
}

ScaleVector Utilities::Configuration::getScale()
{
	return m_scale;
}

FindSpecialSeamVector Utilities::Configuration::getFindSpecialSeam()
{
	return m_findSpecialSeam;
}

ZeroMassPointsNum Utilities::Configuration::getZeroMassPointsNum()
{
	return m_zeroMassPointsNum;
}

ZeroMassPointsVector Utilities::Configuration::getZeroMassPointsVector()
{
	return m_zeroMassPointsVector;
}

void Utilities::Configuration::setBodyPath(string bodyPath)
{
	m_bodyPath = bodyPath;
}

void Utilities::Configuration::setBvhPath(string bvhPath)
{
	m_bvhPath = bvhPath;
}

void Utilities::Configuration::setBvhName(string bvhName)
{
	m_bvhName = bvhName;
}

void Utilities::Configuration::setStartFrame(unsigned int startFrame)
{
	m_startFrame = startFrame;
}

void Utilities::Configuration::setCurrentFrame(unsigned int currentFrame)
{
	m_currentFrame = currentFrame;
}

void Utilities::Configuration::setEndFrame(unsigned int endFrame)
{
	m_endFrame = endFrame;
}

void Utilities::Configuration::setStepSize(unsigned int stepSize)
{
	m_stepSize = stepSize;
}

void Utilities::Configuration::setStepCount(unsigned int stepCount)
{
	m_stepCount = stepCount;
}

string Utilities::Configuration::getBodyPath()
{
	return m_bodyPath;
}

string Utilities::Configuration::getBvhPath()
{
	return m_bvhPath;
}

string Utilities::Configuration::getBvhName()
{
	return m_bvhName;
}

unsigned int Utilities::Configuration::getStartFrame()
{
	return m_startFrame;
}

unsigned int Utilities::Configuration::getCurrentFrame()
{
	return m_currentFrame;
}

unsigned int Utilities::Configuration::getEndFrame()
{
	return m_endFrame;
}

unsigned int Utilities::Configuration::getStepSize()
{
	return m_stepSize;
}

unsigned int Utilities::Configuration::getStepCount()
{
	return m_stepCount;
}

void Utilities::Configuration::setZoom(float zoom)
{
	m_zoom = zoom;
}

void Utilities::Configuration::setSwingAngle(float swingAngle)
{
	m_swingAngle = swingAngle;
}

void Utilities::Configuration::setElevateAngle(float elevateAngle)
{
	m_elevateAngle = elevateAngle;
}

void Utilities::Configuration::setCenter(vector<float> center)
{
	m_center = center;
}

float Utilities::Configuration::getZoom()
{
	return m_zoom;
}

float Utilities::Configuration::getSwingAngle()
{
	return m_swingAngle;
}

float Utilities::Configuration::getElevateAngle()
{
	return m_elevateAngle;
}

vector<float> Utilities::Configuration::getCenter()
{
	return m_center;
}

void Utilities::Configuration::setExtractValue(bool b)
{
	m_isValid = b;
}

bool Utilities::Configuration::isExtractValid()
{
	return m_isValid;
}
