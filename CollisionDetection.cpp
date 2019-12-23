#include "CollisionDetection.h"
#include "IDFactory.h"
#include "MyVector.h"
using namespace PBD;
using namespace Utilities;

int CollisionDetection::CollisionObjectWithoutGeometry::TYPE_ID = 7;

CollisionDetection::CollisionDetection() :
	m_collisionObjects()
{
	m_collisionObjects.reserve(100);
	m_contactCB = NULL;
	m_tolerance = 0.01;
}

CollisionDetection::~CollisionDetection()
{
	cleanup();
}

void CollisionDetection::cleanup()
{
	for (unsigned int i = 0; i < m_collisionObjects.size(); i++)
		delete m_collisionObjects[i];
	m_collisionObjects.clear();
}

void CollisionDetection::addRigidBodyContact(const unsigned int rbIndex1, const unsigned int rbIndex2,
	const Vector3r & cp1, const Vector3r & cp2, const Vector3r & rbCenter,
	const Vector3r & normal, const Real dist,
	const Real restitutionCoeff, const Real frictionCoeff)
{
	if (m_contactCB)
		m_contactCB(0, 0, RigidBodyContactType, rbIndex1, rbIndex2, NULL, NULL, cp1, cp2, rbCenter, normal, dist, restitutionCoeff, frictionCoeff, m_contactCBUserData);
}

void CollisionDetection::addParticleRigidBodyContact(unsigned int first, unsigned int second, const unsigned int particleIndex, const unsigned int rbIndex,
	const Vector3r & cp1, const Vector3r & cp2, const Vector3r & rbCenter,
	const Vector3r & normal, const Real dist,
	const Real restitutionCoeff, const Real frictionCoeff)
{
	if (m_contactCB)
		m_contactCB(first, second, ParticleRigidBodyContactType, particleIndex, rbIndex, NULL, NULL, cp1, cp2, rbCenter, normal, dist, restitutionCoeff, frictionCoeff, m_contactCBUserData);
}

void CollisionDetection::addParticlesContact(unsigned int first, unsigned int second, const unsigned int particleIndex, const unsigned int triIndex1,
	const unsigned int triIndex2, const unsigned int triIndex3, const Vector3r & cp1, const Vector3r & cp2,
	const Vector3r & normal, const Real dist, const Real restitutionCoeff, const Real frictionCoeff)
{
	if (m_contactCB)
		m_contactCB(first, second, ParticlesContactType, particleIndex, triIndex1, triIndex2, triIndex3, cp1, cp2, Vector3r(0.0, 0.0, 0.0), normal, dist, restitutionCoeff, frictionCoeff, m_contactCBUserData);
}

void CollisionDetection::addCollisionObject(const unsigned int bodyIndex, const unsigned int bodyType)
{
	CollisionObjectWithoutGeometry* co = new CollisionObjectWithoutGeometry();
	co->m_bodyIndex = bodyIndex;
	co->m_bodyType = bodyType;
	m_collisionObjects.push_back(co);
}

void CollisionDetection::setContactCallback(CollisionDetection::ContactCallbackFunction val, void* userData)
{
	m_contactCB = val;
	m_contactCBUserData = userData;
}

