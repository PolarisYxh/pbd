#ifndef _COLLISIONDETECTION_H
#define _COLLISIONDETECTION_H

#include "Common.h"
#include "ObjectArray.h"
#include "SimulationModel.h"
#include "ConfigurationLoader.h"
#include "MyVector.h"

namespace PBD
{
	class CollisionDetection
	{
	public:
		static const unsigned int RigidBodyContactType = 0;
		static const unsigned int ParticlesContactType = 2;
		static const unsigned int ParticleRigidBodyContactType = 1;

		typedef void(*ContactCallbackFunction)(unsigned int first, unsigned int second, const unsigned int contactType, const unsigned int bodyIndex1, const unsigned int bodyIndex2,
			const unsigned int bodyIndex3, const unsigned int bodyIndex4,
			const Vector3r& cp1, const Vector3r& cp2, const Vector3r& rbCenter,
			const Vector3r& normal, const Real dist,
			const Real restitutionCoeff, const Real frictionCoeff, void* userData);

		struct CollisionObject
		{
			static const unsigned int RigidBodyCollisionObjectType = 0;
			static const unsigned int TriangleModelCollisionObjectType = 1;
			static const unsigned int TetModelCollisionObjectType = 2;

			unsigned int m_bodyIndex;
			unsigned int m_bodyType;

			virtual ~CollisionObject() {}
			virtual int& getTypeId() const = 0;

		};

		struct CollisionObjectWithoutGeometry : public CollisionObject
		{
			static int TYPE_ID;
			virtual int& getTypeId() const { return TYPE_ID; }
			virtual ~CollisionObjectWithoutGeometry() {}
		};

	protected:
		Real m_tolerance;
		ContactCallbackFunction m_contactCB;
		void* m_contactCBUserData;
		Utilities::ObjectArray<CollisionObject*> m_collisionObjects;

	public:
		CollisionDetection();
		virtual ~CollisionDetection();

		void cleanup();

		Real getTolerance() const { return m_tolerance; }
		void setTolerance(Real val) { m_tolerance = val; }

		void addRigidBodyContact(const unsigned int rbIndex1, const unsigned int rbIndex2,
			const Vector3r& cp1, const Vector3r& cp2, const Vector3r& rbCenter,
			const Vector3r& normal, const Real dist,
			const Real restitutionCoeff, const Real frictionCoeff);

		void addParticleRigidBodyContact(unsigned int first, unsigned int second, const unsigned int particleIndex, const unsigned int rbIndex,
			const Vector3r& cp1, const Vector3r& cp2, const Vector3r& rbCenter,
			const Vector3r& normal, const Real dist,
			const Real restitutionCoeff, const Real frictionCoeff);

		void addParticlesContact(unsigned int first, unsigned int second, const unsigned int particleIndex, const unsigned int triIndex1, const unsigned int triIndex2, const unsigned int triIndex3,
			const Vector3r& cp1, const Vector3r& cp2, const Vector3r& normal, const Real dist,
			const Real restitutionCoeff, const Real frictionCoeff);

		virtual void addCollisionObject(const unsigned int bodyIndex, const unsigned int bodyType);

		Utilities::ObjectArray<CollisionObject*>& getCollisionObjects() { return m_collisionObjects; }

		virtual void collisionDetection(SimulationModel& model) = 0;
		virtual void collisionDetection(SimulationModel& model, Utilities::Configuration& conf) = 0;

		void setContactCallback(CollisionDetection::ContactCallbackFunction val, void* userData);
		virtual void updateBVH(SimulationModel& model) = 0;
	};
}

#endif
