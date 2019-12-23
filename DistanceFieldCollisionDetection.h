#ifndef _SIMPLECOLLISIONDETECTION_H
#define _SIMPLECOLLISIONDETECTION_H

#include "Common.h"
#include "CollisionDetection.h"

#include "triangleHierarchy.h"
#include "ConfigurationLoader.h"
#include <MyVector.h>
using namespace Utilities;

namespace PBD
{
	/** Distance field collision detection. */
	class DistanceFieldCollisionDetection : public CollisionDetection
	{
	public:
		//以三角面片为基本元素
		struct DistanceFieldCollisionObjectOnFaces : public CollisionObject
		{
			bool m_testMesh;
			Real m_invertSDF;
			BVHOnFaces m_bvhf;

			DistanceFieldCollisionObjectOnFaces() { m_testMesh = true; m_invertSDF = 1.0; }
			virtual ~DistanceFieldCollisionObjectOnFaces() {}
			virtual bool collisionTest(const Vector3r& x, const Real tolerance, Vector3r& cp, Vector3r& n, Real& dist, const Real maxDist = 0.0);
			virtual void approximateNormal(const Vector3r& x, const Real tolerance, Vector3r& n);

			virtual Real distance(const Vector3r& x, const Real tolerance) = 0;
		};
		struct DistanceFieldCollisionSphereOnFaces : public DistanceFieldCollisionObjectOnFaces
		{
			Real m_radius;
			static int TYPE_ID;

			virtual ~DistanceFieldCollisionSphereOnFaces() {}
			virtual int& getTypeId() const { return TYPE_ID; }
			virtual bool collisionTest(const Vector3r& x, const Real tolerance, Vector3r& cp, Vector3r& n, Real& dist, const Real maxDist = 0.0);
			virtual Real distance(const Vector3r& x, const Real tolerance);
		};

		struct ContactData
		{
			unsigned int m_first;
			unsigned int m_second;
			char m_type;
			unsigned int m_index1;
			unsigned int m_index2;
			unsigned int m_index3;
			unsigned int m_index4;
			Vector3r m_cp1;
			Vector3r m_cp2;
			Vector3r m_rbCenter;
			Vector3r m_normal;
			Real m_dist;
			Real m_restitution;
			Real m_friction;
		};

	protected:
		void collisionDetectionParticlesOnFaces(unsigned int i, unsigned int k, const ParticleData& pd, const unsigned int offset1, const unsigned int numVert1,
			DistanceFieldCollisionSphereOnFaces* co1, const unsigned int offset2, const unsigned int numVert2,
			DistanceFieldCollisionSphereOnFaces* co2,
			const Real restitutionCoeff, const Real frictionCoeff,
			std::vector<std::vector<ContactData>>& contacts_mt);
		void collisionDetectionRBSolidOnFaces(unsigned int i, unsigned int k, const ParticleData& pd, const unsigned int offset, const unsigned int numVert,
			DistanceFieldCollisionSphereOnFaces* co1, RigidBody* rb2, DistanceFieldCollisionSphereOnFaces* co2,
			const Real restitutionCoeff, const Real frictionCoeff
			, std::vector<std::vector<ContactData> >& contacts_mt, float tolerance
		);

	public:
		DistanceFieldCollisionDetection();
		virtual ~DistanceFieldCollisionDetection();

		void updateBVH(SimulationModel& model);

		void collisionDetection(SimulationModel& mode);
		void collisionDetection(SimulationModel& model, Configuration& conf);

		virtual bool isDistanceFieldCollisionObject(CollisionObject* co) const;


		void addCollisionSphereOnFaces(const unsigned int bodyIndex, const unsigned int bodyType, vector<vector<unsigned int>>faces, const unsigned int numFaces, const Vector3r* vertices, const unsigned int numVertices, const Real radius, const bool testMesh = true, const bool invertSDF = false);
		void popCollisionObject();
	};
}

#endif
