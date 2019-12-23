#ifndef __TRIANGLEMODEL_H__
#define __TRIANGLEMODEL_H__

#include "Common.h"
#include <vector>
#include "IndexedFaceMesh.h"
#include "ParticleData.h"

namespace PBD
{
	struct TriangleModel
	{
	public:
		TriangleModel();
		TriangleModel& operator=(TriangleModel const& other);
		virtual ~TriangleModel();


	protected:
		/** offset which must be added to get the correct index in the particles array */
		unsigned int m_indexOffset;
		/** Face mesh of particles which represents the simulation model */
		IndexedFaceMesh m_particleMesh;
		Real m_restitutionCoeff;         //压缩参数
		Real m_frictionCoeff;            //拉伸参数
		Real m_bendingCoeff;             //弯曲系数
		Real m_dampingCoeff;           //迭代系数
		Real m_airDragCoeff;           //空气阻力参数
		Real m_slideFrictionCoeff;          //滑动摩擦参数
		Real m_collisionCoeff;
		bool m_findSpecialSeam;

	public:
		void updateConstraints();

		IndexedFaceMesh& getParticleMesh();
		void initMesh(const unsigned int nPoints, const unsigned int nFaces, const unsigned int indexOffset, unsigned int* indices, const IndexedFaceMesh::UVIndices& uvIndices, const IndexedFaceMesh::NormalIndices& normalIndices, const IndexedFaceMesh::UVs& uvs);
		void cleanupModel();

		unsigned int getIndexOffset() const;

		void updateMeshNormals(const ParticleData& pd);

		FORCE_INLINE Real getRestitutionCoeff() const
		{
			return m_restitutionCoeff;
		}

		FORCE_INLINE void setRestitutionCoeff(Real val)
		{
			m_restitutionCoeff = val;
		}

		FORCE_INLINE Real getFrictionCoeff() const
		{
			return m_frictionCoeff;
		}

		FORCE_INLINE void setFrictionCoeff(Real val)
		{
			m_frictionCoeff = val;
		}

		FORCE_INLINE Real getBendingCoeff() const
		{
			return m_bendingCoeff;
		}

		FORCE_INLINE void setBendingCoeff(Real val)
		{
			m_bendingCoeff = val;
		}

		FORCE_INLINE void setSpecialSeam(bool fSS)
		{
			m_findSpecialSeam = fSS;
		}
		FORCE_INLINE bool getSpecialSeam()const
		{
			return m_findSpecialSeam;
		}

		FORCE_INLINE void setDampingCoeff(unsigned int dampingCoeff)
		{
			m_dampingCoeff = dampingCoeff;
		}
		FORCE_INLINE unsigned int getDampingCoeff()const
		{
			return m_dampingCoeff;
		}
		FORCE_INLINE void setSlideFrictionCoeff(Real slideFriction)
		{
			m_slideFrictionCoeff = slideFriction;
		}
		FORCE_INLINE Real getSlideFrictionCoeff() const
		{
			return m_slideFrictionCoeff;
		}
		FORCE_INLINE void setAirDragCoeff(Real airDragCoeff)
		{
			m_airDragCoeff = airDragCoeff;
		}
		FORCE_INLINE Real getAirDragCoeff()
		{
			return m_airDragCoeff;
		}
		FORCE_INLINE void setCollisionCoeff(Real collisionCoeff)
		{
			m_collisionCoeff = collisionCoeff;
		}
		FORCE_INLINE Real getCollisionCoeff()
		{
			return m_collisionCoeff;
		}
	};
}

#endif