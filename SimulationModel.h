#ifndef __SIMULATIONMODEL_H__
#define __SIMULATIONMODEL_H__

#include "Common.h"
#include <vector>
#include "RigidBody.h"
#include "ParticleData.h"
#include "TriangleModel.h"
#include "ObjectArray.h"
#include "Constraint.h"
#include "MyVector.h"

namespace PBD
{
	class Constraint;
	class RigidBody;
	class TriangleModel;
	class IndexedFaceMesh;
	class ParticleRigidBodyContactConstraint;
	class ParticlesContactConstraint;
	class ParticleData;
	class SimulationModel
	{
	public:
		SimulationModel();
		virtual ~SimulationModel();
		typedef std::vector<Constraint*> ConstraintVector;
		typedef Utilities::ObjectArray<ParticleRigidBodyContactConstraint> ParticleRigidBodyContactConstraintVector;
		typedef Utilities::ObjectArray<ParticlesContactConstraint> ParticlesContactConstraintVector;
		typedef std::vector<RigidBody*> RigidBodyVector;
		typedef std::vector<TriangleModel*> TriangleModelVector;
		typedef std::vector<unsigned int> ConstraintGroup;
		typedef std::vector<ConstraintGroup> ConstraintGroupVector;


	protected:
		RigidBodyVector m_rigidBodies;
		TriangleModelVector m_triangleModels;
		ParticleData m_particles;
		ConstraintVector m_constraints;
		ParticleRigidBodyContactConstraintVector m_particleRigidBodyContactConstraints;
		ParticlesContactConstraintVector m_particlesContactConstraints;
		ConstraintGroupVector m_constraintGroups;

		Real m_cloth_stiffness;
		Real m_cloth_bendingStiffness;

		Real m_contactStiffnessRigidBody;
		Real m_contactStiffnessParticleRigidBody;

	public:
		void reset();
		void cleanup();
		void cleanRigidBodies();
		SimulationModel& operator= (const SimulationModel& s);

		RigidBodyVector& getRigidBodies();
		ParticleData& getParticles();
		TriangleModelVector& getTriangleModels();
		ConstraintVector& getConstraints();
		ParticleRigidBodyContactConstraintVector& getParticleRigidBodyContactConstraints();
		ParticlesContactConstraintVector& getParticlesContactConstraints();
		ConstraintGroupVector& getConstraintGroups();
		bool m_groupsInitialized;

		void resetContacts();

		void addTriangleModel(
			const unsigned int nPoints,
			const unsigned int nFaces,
			Vector3r* points,
			unsigned int* indices,
			const IndexedFaceMesh::UVIndices& uvIndices,
			const IndexedFaceMesh::NormalIndices& normalIndices,
			const IndexedFaceMesh::UVs& uvs,
			bool findSpecialSeam);

		void updateConstraints();
		void initConstraintGroups();

		bool addParticleRigidBodyContactConstraint(unsigned int first, unsigned int second, const unsigned int particleIndex, const unsigned int rbIndex,
			const Vector3r& cp1, const Vector3r& cp2, const Vector3r& rbCenter,
			const Vector3r& normal, const Real dist,
			const Real restitutionCoeff, const Real frictionCoeff);
		bool addParticlesContactConstraint(unsigned int first, unsigned int second, const unsigned int particleIndex, const unsigned int triIndex1, const unsigned int triIndex2,
			const unsigned int triIndex3, const Vector3r& cp1, const Vector3r& cp2, const Vector3r& normal,
			const Real dist, const Real restitutionCoeff, const Real frictionCoeff);

		bool addSeamConstraint(const unsigned int particle1, const unsigned int particle2);

		bool addDistanceConstraint(const unsigned int particle1, const unsigned int particle2, const unsigned int triModelNum = 0);
		bool addDihedralConstraint(const unsigned int particle1, const unsigned int particle2,
			const unsigned int particle3, const unsigned int particle4);
		Real getClothStiffness() const { return m_cloth_stiffness; }
		void setClothStiffness(Real val) { m_cloth_stiffness = val; }
		Real getClothBendingStiffness() const { return m_cloth_bendingStiffness; }
		void setClothBendingStiffness(Real val) { m_cloth_bendingStiffness = val; }

		Real getContactStiffnessRigidBody() const { return m_contactStiffnessRigidBody; }
		void setContactStiffnessRigidBody(Real val) { m_contactStiffnessRigidBody = val; }
		Real getContactStiffnessParticleRigidBody() const { return m_contactStiffnessParticleRigidBody; }
		void setContactStiffnessParticleRigidBody(Real val) { m_contactStiffnessParticleRigidBody = val; }
	};
}

#endif