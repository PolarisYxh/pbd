#ifndef __MYMATRIX_H__
#define __MYMATRIX_H__

#include<iostream>   
#include <fstream>      // std::ifstream
#include <stdlib.h>      
#include <cmath>     
#include <Common.h>
#include "MYVector.h"

/*
类的定义
*/

namespace Utilities
{
	template <class T>
	struct Matrix3
	{
		T arr[3][3];
		Matrix3()
		{
			for (int i = 0; i < 3; i++)
				for (int j = 0; j < 3; j++)
					arr[i][j] = 0.0;
		}
		void set(int i, int j, float val)
		{
			arr[i][j] = val;
		}
		void setOn(int i, int j, float val)
		{
			arr[i][j] = arr[i][j] + val;
		}
		void setZero()
		{
			for (int i = 0; i < 3; i++)
				for (int j = 0; j < 3; j++)
					arr[i][j] = 0.0;
		}
		void setIdentity()
		{
			setZero();
			for (int i = 0; i < 3; i++)
				arr[i][i] = 1.0;
		}

		T operator ()  (int i, int j) const { return arr[i][j]; }
		Vector3r operator * (Vector3r v) const
		{
			return Vector3r(arr[0][0] * v[0] + arr[0][1] * v[1] + arr[0][2] * v[2],
				arr[1][0] * v[0] + arr[1][1] * v[1] + arr[1][2] * v[2],
				arr[2][0] * v[0] + arr[2][1] * v[1] + arr[2][2] * v[2]);
		}
	};

	typedef Matrix3<double> Matrix3r;

	template <class T>
	struct Matrix4
	{
		T arr[4][4];
		Matrix4()
		{
			for (int i = 0; i < 4; i++)
				for (int j = 0; j < 4; j++)
					arr[i][j] = 0.0;
		}
		void setZero()
		{
			for (int i = 0; i < 4; i++)
				for (int j = 0; j < 4; j++)
					arr[i][j] = 0.0;
		}
		void setIdentity()
		{
			setZero();
			for (int i = 0; i < 4; i++)
				arr[i][i] = 1.0;
		}

		T& operator ()  (int i, int j) const { return arr[i][j]; }
	};

	typedef Matrix4<double> Matrix4r;



	template <class T>
	struct Matrix35
	{
		Matrix35()
		{
			for (int i = 0; i < 3; i++)
				for (int j = 0; j < 5; j++)
					arr[i][j] = 0.0;
		}
		Matrix35 setZero()
		{
			for (int i = 0; i < 3; i++)
				for (int j = 0; j < 5; j++)
					arr[i][j] = 0.0;
		}
		Vector3r & col(int i)
		{
			return Vector3r(arr[0][i], arr[1][i], arr[2][i]);
		}

		Matrix35& setCol(int i, Vector3r v)
		{
			arr[0][i] = v.x;
			arr[1][i] = v.y;
			arr[2][i] = v.z;
			return *this;
		}

		T& operator ()  (int i, int j)  { return arr[i][j]; }

		T arr[3][5];
	};

	typedef Matrix35<double> Matrix35r;
}
#endif