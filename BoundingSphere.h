#ifndef __BOUNDINGSPHERE_H__
#define __BOUNDINGSPHERE_H__

#include "Common.h"
#include "MyVector.h"

namespace PBD
{

	class BoundingSphere
	{
	public:

		BoundingSphere() = default;
		BoundingSphere(float * x, float r) 
		{
			m_x[0] = x[0];
			m_x[1] = x[1];
			m_x[2] = x[2];
			m_r = r;
		}

		float*  x() { return m_x; }

		float r() const { return m_r; }
		float& r() { return m_r; }

		bool overlaps(BoundingSphere const& other) const
		{
			float rr = m_r + other.m_r;
			return (Vector3r(m_x[0],m_x[1],m_x[2]) - Vector3r(other.m_x[0], other.m_x[1], other.m_x[2])).squaredNorm() < rr * rr;
		}

		bool contains(BoundingSphere const& other) const
		{
			float rr = r() - other.r();
			return (Vector3r(m_x[0], m_x[1], m_x[2]) - Vector3r(other.m_x[0], other.m_x[1], other.m_x[2])).squaredNorm() < rr * rr;
		}

		bool contains(float* other) const
		{
			return (Vector3r(m_x[0], m_x[1], m_x[2]) - Vector3r(other[0],other[1],other[2])).squaredNorm() < m_r * m_r;
		}

	public:

		float m_x[3];
		float m_r;
	};

}

#endif
