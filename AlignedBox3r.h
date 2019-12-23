#ifndef __ALIGNEDBOX3R_H__
#define __ALIGNEDBOX3R_H__
#include "MyVector.h"

struct AlignedBox3r
{
	float maxX, maxY, maxZ, minX, minY, minZ;
	AlignedBox3r()
	{
		maxX = 0.0; maxY = 0.0; maxZ = 0.0;
		minX = 0.0; minY = 0.0; minZ = 0.0;
	}
	void extend(Vector3r x)
	{
		if (x.x > maxX)maxX = x.x;
		if (x.y > maxY)maxY = x.y;
		if (x.z > maxZ)maxZ = x.z;
		if (x.x < minX)minX = x.x;
		if (x.y < minY)minY = x.y;
		if (x.z < minZ)minZ = x.z;
	}
	Vector3r diagonal()
	{
		return Vector3r(maxX - minX, maxY - minY, maxZ - minZ);
	}
	Vector3r max()
	{
		return Vector3r(maxX, maxY, maxZ);
	}
	Vector3r min()
	{
		return Vector3r(minX, minY, minZ);
	}
	void setBox(AlignedBox3r other)
	{
		maxX = other.maxX;
		maxY = other.maxY;
		maxZ = other.maxZ;
		minX = other.minX;
		minY = other.minY;
		minZ = other.minZ;
	}

	AlignedBox3r operator =(AlignedBox3r& other)
	{
		maxX = other.maxX;
		maxY = other.maxY;
		maxZ = other.maxZ;
		minX = other.minX;
		minY = other.minY;
		minZ = other.minZ;
		return *this;
	}
};


#endif 

