#ifndef __PARTICLEDATA_H__
#define __PARTICLEDATA_H__

#include "Common.h"
#include <MyVector.h>
#include <vector>
namespace PBD
{
	/** This class encapsulates the state of all vertices.
	* All parameters are stored in individual arrays.
	*/
	struct VertexData
	{
	private:
		std::vector<Vector3r> m_x;
	

	public:
		FORCE_INLINE VertexData(void) :
			m_x()
		{
		}

		FORCE_INLINE ~VertexData(void)
		{
			m_x.clear();
		}

		FORCE_INLINE void addVertex(const Vector3r& vertex)
		{
			m_x.push_back(vertex);
		}

		FORCE_INLINE Vector3r& getPosition(const unsigned int i)
		{
			return m_x[i];
		}

		FORCE_INLINE const Vector3r& getPosition(const unsigned int i) const
		{
			return m_x[i];
		}

		FORCE_INLINE void setPosition(const unsigned int i, const Vector3r& pos)
		{
			m_x[i] = pos;
		}

		/** Resize the array containing the particle data.
		*/
		FORCE_INLINE void resize(const unsigned int newSize)
		{
			m_x.resize(newSize);
		}

		/** Reserve the array containing the particle data.
		*/
		FORCE_INLINE void reserve(const unsigned int newSize)
		{
			m_x.reserve(newSize);
		}

		/** Release the array containing the particle data.
		*/
		FORCE_INLINE void release()
		{
			m_x.clear();
		}

		/** Release the array containing the particle data.
		*/
		FORCE_INLINE unsigned int size() const
		{
			return (unsigned int)m_x.size();
		}

		FORCE_INLINE const std::vector<Vector3r>* getVertices()
		{
			return &m_x;
		}
	};

	/** This class encapsulates the state of all particles of a particle model.
	 * All parameters are stored in individual arrays.
	 */
	struct ParticleData
	{
	private:
		// Mass
		// If the mass is zero, the particle is static
		std::vector<Real> m_masses;
		std::vector<Real> m_invMasses;

		// Dynamic state
		std::vector<Vector3r> m_x0;          //模型初始位置
		std::vector<Vector3r> m_x;           //当前位置
		std::vector<Vector3r> m_v;       
		std::vector<Vector3r> m_a;
		std::vector<Vector3r> m_oldX;        //上一帧位置
		std::vector<Vector3r> m_lastX;		 //当前帧开始位置
		std::vector<Real>     m_area;        //该点计算空气阻力时的面积

		std::vector<std::vector<unsigned int>> m_collisionIndex;   //存储与之碰撞物体的index

	public:
		ParticleData operator=(const ParticleData& p)
		{
			this->m_masses = p.m_masses;
			this->m_invMasses = p.m_masses;
			this->m_x0 = p.m_x0;
			this->m_x = p.m_x;
			this->m_v = p.m_v;
			this->m_a = p.m_a;
			this->m_oldX = p.m_oldX;
			this->m_lastX = p.m_lastX;
			this->m_area = p.m_area;
			return *this;
		}
		FORCE_INLINE ParticleData(void) :
			m_masses(),
			m_invMasses(),
			m_x0(),
			m_x(),
			m_v(),
			m_a(),
			m_oldX(),
			m_lastX(),
			m_collisionIndex(),
			m_area()
		{
		}

		FORCE_INLINE ~ParticleData(void)
		{
			m_masses.clear();
			m_invMasses.clear();
			m_x0.clear();
			m_x.clear();
			m_v.clear();
			m_a.clear();
			m_oldX.clear();
			m_lastX.clear();
			m_collisionIndex.clear();
			m_area.clear();
		}

		FORCE_INLINE void addVertex(const Vector3r& vertex)
		{
			m_x0.push_back(vertex);
			m_x.push_back(vertex);
			m_oldX.push_back(vertex);
			m_lastX.push_back(vertex);
			m_masses.push_back(1.0);
			m_invMasses.push_back(1.0);
			m_v.push_back(Vector3r{ 0.0, 0.0, 0.0 });
			m_a.push_back(Vector3r{ 0.0, 0.0, 0.0 });
			std::vector<unsigned int> t; t.push_back(100);
			m_collisionIndex.push_back(t);
			t.clear();
			m_area.push_back(0.0);
		}
		//
		std::vector<unsigned int>& getCollisionIndex(unsigned int i)
		{
			return m_collisionIndex[i];
		}

		//for GPU
		FORCE_INLINE std::vector<Vector3r>& getX()
		{
			return m_x;
		}

		FORCE_INLINE const std::vector<Vector3r>& getX() const
		{
			return m_x;
		}
		FORCE_INLINE std::vector<Vector3r>& getX0()
		{
			return m_x0;
		}

		FORCE_INLINE const std::vector<Vector3r>& getX0() const
		{
			return m_x0;
		}

		FORCE_INLINE Real& getArea(const unsigned int i)
		{
			return m_area[i];
		}

		FORCE_INLINE void setArea(const unsigned int i, Real area)
		{
			m_area[i] = area;
		}

		//存储每次碰撞的速度调整数据
		FORCE_INLINE Vector3r& getPosition(const unsigned int i)
		{
			return m_x[i];
		}

		FORCE_INLINE const Vector3r& getPosition(const unsigned int i) const
		{
			return m_x[i];
		}

		FORCE_INLINE void setPosition(const unsigned int i, const Vector3r& pos)
		{
			m_x[i] = pos;
		}

		FORCE_INLINE Vector3r& getPosition0(const unsigned int i)
		{
			return m_x0[i];
		}

		FORCE_INLINE const Vector3r& getPosition0(const unsigned int i) const
		{
			return m_x0[i];
		}

		FORCE_INLINE void setPosition0(const unsigned int i, const Vector3r& pos)
		{
			m_x0[i] = pos;
		}

		FORCE_INLINE Vector3r& getLastPosition(const unsigned int i)
		{
			return m_lastX[i];
		}

		FORCE_INLINE const Vector3r& getLastPosition(const unsigned int i) const
		{
			return m_lastX[i];
		}

		FORCE_INLINE void setLastPosition(const unsigned int i, const Vector3r& pos)
		{
			m_lastX[i] = pos;
		}

		FORCE_INLINE Vector3r& getOldPosition(const unsigned int i)
		{
			return m_oldX[i];
		}

		FORCE_INLINE const Vector3r& getOldPosition(const unsigned int i) const
		{
			return m_oldX[i];
		}

		FORCE_INLINE void setOldPosition(const unsigned int i, const Vector3r& pos)
		{
			m_oldX[i] = pos;
		}

		FORCE_INLINE Vector3r& getVelocity(const unsigned int i)
		{
			return m_v[i];
		}

		FORCE_INLINE const Vector3r& getVelocity(const unsigned int i) const
		{
			return m_v[i];
		}

		FORCE_INLINE void setVelocity(const unsigned int i, const Vector3r& vel)
		{
			m_v[i] = vel;
		}

		FORCE_INLINE Vector3r& getAcceleration(const unsigned int i)
		{
			return m_a[i];
		}

		FORCE_INLINE const Vector3r& getAcceleration(const unsigned int i) const
		{
			return m_a[i];
		}

		FORCE_INLINE void setAcceleration(const unsigned int i, const Vector3r& accel)
		{
			m_a[i] = accel;
		}

		FORCE_INLINE const Real getMass(const unsigned int i) const
		{
			return m_masses[i];
		}

		FORCE_INLINE Real& getMass(const unsigned int i)
		{
			return m_masses[i];
		}

		FORCE_INLINE void setMass(const unsigned int i, const Real mass)
		{
			m_masses[i] = mass;
			if (mass != 0.0)
				m_invMasses[i] = 1.0 / mass;
			else
				m_invMasses[i] = 0.0;
		}

		FORCE_INLINE const Real getInvMass(const unsigned int i) const
		{
			return m_invMasses[i];
		}

		FORCE_INLINE const unsigned int getNumberOfParticles() const
		{
			return (unsigned int)m_x.size();
		}

		/** Resize the array containing the particle data.
		 */
		FORCE_INLINE void resize(const unsigned int newSize)
		{
			m_masses.resize(newSize);
			m_invMasses.resize(newSize);
			m_x0.resize(newSize);
			m_x.resize(newSize);
			m_v.resize(newSize);
			m_a.resize(newSize);
			m_oldX.resize(newSize);
			m_lastX.resize(newSize);
			m_collisionIndex.resize(newSize);
		}

		/** Reserve the array containing the particle data.
		 */
		FORCE_INLINE void reserve(const unsigned int newSize)
		{
			m_masses.reserve(newSize);
			m_invMasses.reserve(newSize);
			m_x0.reserve(newSize);
			m_x.reserve(newSize);
			m_v.reserve(newSize);
			m_a.reserve(newSize);
			m_oldX.reserve(newSize);
			m_lastX.reserve(newSize);
			m_collisionIndex.reserve(newSize);
		}

		/** Release the array containing the particle data.
		 */
		FORCE_INLINE void release()
		{
			m_masses.clear();
			m_invMasses.clear();
			m_x0.clear();
			m_x.clear();
			m_v.clear();
			m_a.clear();
			m_oldX.clear();
			m_lastX.clear();
			m_collisionIndex.clear();
		}

		/** Release the array containing the particle data.
		 */
		FORCE_INLINE unsigned int size() const
		{
			return (unsigned int)m_x.size();
		}
	};
}

#endif