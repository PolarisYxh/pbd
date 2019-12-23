#ifndef __RIGIDBODY_H__
#define __RIGIDBODY_H__

#include <vector>
#include "Common.h"
#include<iostream>
#include<IndexedFaceMesh.h>
#include<ParticleData.h>
using namespace std;

namespace PBD
{
	/** This class encapsulates the state of a rigid body.
	 */
	struct RigidBody
	{
	protected:
		IndexedFaceMesh m_mesh;
		VertexData m_vertexData_local;
		VertexData m_vertexData;
		
		Real m_restitutionCoeff;
		Real m_frictionCoeff;


	public:
		RigidBody(void)
		{
		}

		~RigidBody(void)
		{
		}
		void release();
		void initBody(const VertexData& vertices, const PBD::IndexedFaceMesh& mesh,
			const Vector3r& scale);
		VertexData& getVertexData();
		const VertexData& getVertexData() const;
		VertexData& getVertexDataLocal();
		const VertexData& getVertexDataLocal() const;
		IndexedFaceMesh& getMesh();
		const IndexedFaceMesh& getMesh() const;
		void updateMeshNormals(const VertexData& vd);
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
	};
}

#endif
