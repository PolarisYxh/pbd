#include "SimulationModel.h"
#include "PositionBasedDynamics.h"
#include "PositionBasedRigidBodyDynamics.h"
#include "Constraint.h"
#include <iostream>
using namespace std;
using namespace Utilities;
using namespace PBD;

SimulationModel::SimulationModel()
{
	m_cloth_stiffness = 1.0;
	m_cloth_bendingStiffness = 0.01;        //0.01

	m_contactStiffnessRigidBody = 1.0;
	m_contactStiffnessParticleRigidBody = 100.0;

	m_groupsInitialized = false;

	m_particleRigidBodyContactConstraints.reserve(10000);
	m_particlesContactConstraints.reserve(10000);
}

SimulationModel::~SimulationModel(void)
{
	cleanup();
}

SimulationModel& SimulationModel::operator=(const SimulationModel& s)
{
	this->m_rigidBodies = s.m_rigidBodies;
	this->m_triangleModels = s.m_triangleModels;
	this->m_particles = s.m_particles;
	this->m_constraints = s.m_constraints;
	this->m_particleRigidBodyContactConstraints = s.m_particleRigidBodyContactConstraints;
	this->m_particlesContactConstraints = s.m_particlesContactConstraints;
	this->m_constraintGroups = s.m_constraintGroups;
	this->m_cloth_stiffness = s.m_cloth_stiffness;
	this->m_cloth_bendingStiffness = s.m_cloth_bendingStiffness;
	this->m_contactStiffnessRigidBody = s.m_contactStiffnessRigidBody;
	this->m_contactStiffnessParticleRigidBody = s.m_contactStiffnessParticleRigidBody;
	return *this;
}

void SimulationModel::cleanRigidBodies()
{
	m_particleRigidBodyContactConstraints.clear();

	for (unsigned int i = 0; i < m_rigidBodies.size(); i++)
		delete m_rigidBodies[i];
	m_rigidBodies.clear();
}

void SimulationModel::cleanup()
{
	resetContacts();
	for (unsigned int i = 0; i < m_rigidBodies.size(); i++)
		delete m_rigidBodies[i];
	m_rigidBodies.clear();
	for (unsigned int i = 0; i < m_triangleModels.size(); i++)
		delete m_triangleModels[i];
	m_triangleModels.clear();
	for (unsigned int i = 0; i < m_constraints.size(); i++)
		delete m_constraints[i];
	m_constraints.clear();
	m_particles.release();
	m_groupsInitialized = false;
}

void SimulationModel::reset()
{
	resetContacts();

	// rigid bodies
	for (size_t i = 0; i < m_rigidBodies.size(); i++)
	{
	}

	// particles
	for (unsigned int i = 0; i < m_particles.size(); i++)
	{
		const Vector3r& x0 = m_particles.getPosition0(i);
		m_particles.getPosition(i) = x0;
		m_particles.getLastPosition(i) = m_particles.getPosition(i);
		m_particles.getOldPosition(i) = m_particles.getPosition(i);
		m_particles.getVelocity(i).setZero();
		m_particles.getAcceleration(i).setZero();
	}

	updateConstraints();
}

SimulationModel::RigidBodyVector& SimulationModel::getRigidBodies()
{
	return m_rigidBodies;
}

ParticleData& SimulationModel::getParticles()
{
	return m_particles;
}


SimulationModel::TriangleModelVector& SimulationModel::getTriangleModels()
{
	return m_triangleModels;
}

SimulationModel::ConstraintVector& SimulationModel::getConstraints()
{
	return m_constraints;
}


SimulationModel::ParticleRigidBodyContactConstraintVector& SimulationModel::getParticleRigidBodyContactConstraints()
{
	return m_particleRigidBodyContactConstraints;
}

SimulationModel::ParticlesContactConstraintVector& SimulationModel::getParticlesContactConstraints()
{
	return m_particlesContactConstraints;
}

SimulationModel::ConstraintGroupVector& SimulationModel::getConstraintGroups()
{
	return m_constraintGroups;
}

void SimulationModel::updateConstraints()
{
	for (unsigned int i = 0; i < m_constraints.size(); i++)
		m_constraints[i]->updateConstraint(*this);
}

bool SimulationModel::addParticleRigidBodyContactConstraint(unsigned int first, unsigned int second, const unsigned int particleIndex, const unsigned int rbIndex,
	const Vector3r & cp1, const Vector3r & cp2, const Vector3r & rbCenter,
	const Vector3r & normal, const Real dist,
	const Real restitutionCoeff, const Real frictionCoeff)
{
	ParticleRigidBodyContactConstraint& cc = m_particleRigidBodyContactConstraints.create();
	const bool res = cc.initConstraint(*this, first, second, particleIndex, rbIndex, cp1, cp2, rbCenter, normal, dist, restitutionCoeff, m_contactStiffnessParticleRigidBody, frictionCoeff);
	if (!res)
		m_particleRigidBodyContactConstraints.pop_back();
	return res;
}
bool SimulationModel::addParticlesContactConstraint(unsigned int first, unsigned int second, const unsigned int particleIndex, const unsigned int triIndex1, const unsigned int triIndex2, const unsigned int triIndex3, const Vector3r & cp1, const Vector3r & cp2, const Vector3r & normal, const Real dist, const Real restitutionCoeff, const Real frictionCoeff)
{
	ParticlesContactConstraint& cc = m_particlesContactConstraints.create();
	const bool res = cc.initConstraint(*this, first, second, particleIndex, triIndex1, triIndex2, triIndex3, cp1, cp2, normal, dist, restitutionCoeff, m_cloth_stiffness, frictionCoeff);
	if (!res)
		m_particlesContactConstraints.pop_back();
	return res;
}
bool SimulationModel::addSeamConstraint(const unsigned int particle1, const unsigned int particle2)
{
	SeamConstraint* c = new SeamConstraint();
	const bool res = c->initConstraint(*this, particle1, particle2);
	if (res)
	{
		m_constraints.push_back(c);
		m_groupsInitialized = false;
	}
	return res;
}

bool SimulationModel::addDistanceConstraint(const unsigned int particle1, const unsigned int particle2, const unsigned int triModelNum)
{
	DistanceConstraint* c = new DistanceConstraint();
	const bool res = c->initConstraint(*this, particle1, particle2, triModelNum);
	if (res)
	{
		m_constraints.push_back(c);
		m_groupsInitialized = false;
	}
	return res;
}

bool SimulationModel::addDihedralConstraint(const unsigned int particle1, const unsigned int particle2,
	const unsigned int particle3, const unsigned int particle4)
{
	DihedralConstraint* c = new DihedralConstraint();
	const bool res = c->initConstraint(*this, particle1, particle2, particle3, particle4);
	if (res)
	{
		m_constraints.push_back(c);
		m_groupsInitialized = false;
	}
	return res;
}

void SimulationModel::addTriangleModel(
	const unsigned int nPoints,
	const unsigned int nFaces,
	Vector3r * points,
	unsigned int* indices,
	const IndexedFaceMesh::UVIndices & uvIndices,
	const IndexedFaceMesh::NormalIndices & normalIndices,
	const IndexedFaceMesh::UVs & uvs,
	bool findSpecialSeam)
{
	TriangleModel* triModel = new TriangleModel();
	m_triangleModels.push_back(triModel);

	unsigned int startIndex = m_particles.size();
	m_particles.reserve(startIndex + nPoints);

	for (unsigned int i = 0; i < nPoints; i++)
		m_particles.addVertex(points[i]);

	triModel->initMesh(nPoints, nFaces, startIndex, indices, uvIndices,normalIndices, uvs);
	triModel->setSpecialSeam(findSpecialSeam);

	// Update normals
	triModel->updateMeshNormals(m_particles);
}


void SimulationModel::initConstraintGroups()
{
	if (m_groupsInitialized)
		return;

	const unsigned int numConstraints = (unsigned int)m_constraints.size();     //     cout << "m_constrains:" << m_constraints.size() << endl;       17984
	const unsigned int numParticles = (unsigned int)m_particles.size();        //       cout << "m_particles:" << m_particles.size() << endl;          6672
	const unsigned int numRigidBodies = (unsigned int)m_rigidBodies.size();    //       cout << "m_rigidBodies:" << m_rigidBodies.size() << endl;      2
	const unsigned int numBodies = numParticles + numRigidBodies;
	m_constraintGroups.clear();

	// Maps in which group a particle is or 0 if not yet mapped
	std::vector<unsigned char*> mapping;

	for (unsigned int i = 0; i < numConstraints; i++)
	{

		Constraint* constraint = m_constraints[i];

		bool addToNewGroup = true;
		for (unsigned int j = 0; j < m_constraintGroups.size(); j++)
		{
			bool addToThisGroup = true;

			for (unsigned int k = 0; k < constraint->m_numberOfBodies; k++)
			{
				if (mapping[j][constraint->m_bodies[k]] != 0)
				{
					addToThisGroup = false;
					break;
				}
			}

			if (addToThisGroup)
			{
				m_constraintGroups[j].push_back(i);

				for (unsigned int k = 0; k < constraint->m_numberOfBodies; k++)
					mapping[j][constraint->m_bodies[k]] = 1;

				addToNewGroup = false;
				break;
			}
		}
		if (addToNewGroup)
		{
			mapping.push_back(new unsigned char[numBodies]);
			memset(mapping[mapping.size() - 1], 0, sizeof(unsigned char) * numBodies);
			m_constraintGroups.resize(m_constraintGroups.size() + 1);
			m_constraintGroups[m_constraintGroups.size() - 1].push_back(i);                    //m_constraintGroups[0]=0
			for (unsigned int k = 0; k < constraint->m_numberOfBodies; k++)
				mapping[m_constraintGroups.size() - 1][constraint->m_bodies[k]] = 1;
		}
	}

	for (unsigned int i = 0; i < mapping.size(); i++)
	{
		delete[] mapping[i];
	}
	mapping.clear();

	m_groupsInitialized = true;
}

void SimulationModel::resetContacts()
{
	m_particleRigidBodyContactConstraints.reset();
	m_particlesContactConstraints.reset();
}


