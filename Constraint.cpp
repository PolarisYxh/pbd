#include "Constraint.h"
#include "SimulationModel.h"
#include "PositionBasedDynamics.h"
#include "PositionBasedRigidBodyDynamics.h"
#include "TimeManager.h"
#include "IDFactory.h"


using namespace PBD;

int SeamConstraint::TYPE_ID = 0;
int DistanceConstraint::TYPE_ID = 1;
int DihedralConstraint::TYPE_ID = 2;

int ParticleRigidBodyContactConstraint::TYPE_ID = 3;
int ParticlesContactConstraint::TYPE_ID = 4;

Real alpha = 0.4;     //0.4
Real beta = 0.003;
//Real gamma = alpha * beta / 0.005;
Real gamma = 0.24;   //0.24before

//////////////////////////////////////////////////////////////////////////
// DistanceConstraint
//////////////////////////////////////////////////////////////////////////
bool DistanceConstraint::initConstraint(SimulationModel& model, const unsigned int particle1, const unsigned int particle2, unsigned int triModelNum)
{
	m_bodies[0] = particle1;
	m_bodies[1] = particle2;

	ParticleData& pd = model.getParticles();
	const Vector3r& x1_0 = pd.getPosition0(particle1);
	const Vector3r& x2_0 = pd.getPosition0(particle2);

	m_restLength = (x2_0 - x1_0).norm() ;                              // 定义边之间的约束为点之间的距离

	m_triModelNum = triModelNum;

	return true;
}

bool DistanceConstraint::solvePositionConstraint(SimulationModel & model)
{
	ParticleData& pd = model.getParticles();

	const unsigned i1 = m_bodies[0];
	const unsigned i2 = m_bodies[1];

	Vector3r& x1 = pd.getPosition(i1);
	Vector3r& x2 = pd.getPosition(i2);
	const Real invMass1 = pd.getInvMass(i1);
	const Real invMass2 = pd.getInvMass(i2);

	const unsigned int triModelNum = m_triModelNum;
	Real restitutionCoeff = model.getTriangleModels().at(triModelNum)->getRestitutionCoeff();
	Real frictionCoeff = model.getTriangleModels().at(triModelNum)->getFrictionCoeff();

	Vector3r corr1, corr2;
	Real wSum = invMass1 + invMass2;
	if (wSum == 0.0)
		return false;

	Vector3r n = x2 - x1;
	Real d = n.norm();
	n = n.normalize();

	Real corrLambda1 = (-(m_restLength - d) - alpha * m_lambda1 - gamma * n.dot(x1 - pd.getOldPosition(i1))) / ((1 + gamma) * wSum + alpha);
	Real corrLambda2 = (-(m_restLength - d) - alpha * m_lambda1 - gamma * n.dot(x2 - pd.getOldPosition(i2))) / ((1 + gamma) * wSum + alpha);

	Vector3r corr;
	if (d < m_restLength)
	{
		corr1 = n * frictionCoeff * corrLambda1;
		corr2 = n * frictionCoeff * corrLambda2;
	}

	else
	{
		corr1 = n * restitutionCoeff *  corrLambda1;
		corr2 = n * restitutionCoeff *  corrLambda2;
	}


	corr1 =  corr1* invMass1;
	corr2 = -corr2* invMass2;

	m_lambda1 = m_lambda1 + corrLambda1;
	m_lambda2 = m_lambda2 + corrLambda2;

	if (invMass1 != 0.0)
	{
		x1 += corr1;
	}

	if (invMass2 != 0.0)
	{
		x2 += corr2;
	}

	return true;
}


//////////////////////////////////////////////////////////////////////////
// DihedralConstraint
//////////////////////////////////////////////////////////////////////////

bool DihedralConstraint::initConstraint(SimulationModel & model, const unsigned int particle1, const unsigned int particle2,
	const unsigned int particle3, const unsigned int particle4, unsigned int triModelNum)
{
	m_bodies[0] = particle1;
	m_bodies[1] = particle2;
	m_bodies[2] = particle3;
	m_bodies[3] = particle4;
	ParticleData& pd = model.getParticles();

	const Vector3r& p0 = pd.getPosition0(particle1);
	const Vector3r& p1 = pd.getPosition0(particle2);
	const Vector3r& p2 = pd.getPosition0(particle3);
	const Vector3r& p3 = pd.getPosition0(particle4);

	Vector3r e = p3 - p2;
	Real  elen = e.norm();
	if (elen < 1e-6)
		return false;

	Real invElen = 1.0 / elen;

	Vector3r n1 = (p2 - p0).cross(p3 - p0); n1 /= n1.squaredNorm();
	Vector3r n2 = (p3 - p1).cross(p2 - p1); n2 /= n2.squaredNorm();

	n1 = n1.normalize();
	n2 = n2.normalize();
	Real dot = n1.dot(n2);

	if (dot < -1.0) dot = -1.0;
	if (dot > 1.0) dot = 1.0;

	m_restAngle = acos(dot);
	m_triModelNum = triModelNum;

	return true;
}

bool DihedralConstraint::solvePositionConstraint(SimulationModel & model)
{
	ParticleData& pd = model.getParticles();

	const unsigned i1 = m_bodies[0];
	const unsigned i2 = m_bodies[1];
	const unsigned i3 = m_bodies[2];
	const unsigned i4 = m_bodies[3];

	Vector3r& x1 = pd.getPosition(i1);
	Vector3r& x2 = pd.getPosition(i2);
	Vector3r& x3 = pd.getPosition(i3);
	Vector3r& x4 = pd.getPosition(i4);

	const Real invMass1 = pd.getInvMass(i1);
	const Real invMass2 = pd.getInvMass(i2);
	const Real invMass3 = pd.getInvMass(i3);
	const Real invMass4 = pd.getInvMass(i4);

	Real bendStiffness = model.getTriangleModels().at(m_triModelNum)->getBendingCoeff();

	Vector3r corr1, corr2, corr3, corr4;
	const bool res = PositionBasedDynamics::solve_DihedralConstraint(
		x1, invMass1, x2, invMass2, x3, invMass3, x4, invMass4,
		m_restAngle,
		bendStiffness,
		corr1, corr2, corr3, corr4);

	if (res)
	{
		if (invMass1 != 0.0)
			x1 += corr1;
		if (invMass2 != 0.0)
			x2 += corr2;
		if (invMass3 != 0.0)
			x3 += corr3;
		if (invMass4 != 0.0)
			x4 += corr4;
	}
	return res;
}

//////////////////////////////////////////////////////////////////////////
// SeamConstraint
//////////////////////////////////////////////////////////////////////////
bool PBD::SeamConstraint::initConstraint(SimulationModel & model, const unsigned int particle1, const unsigned int particle2)
{
	m_bodies[0] = particle1;
	m_bodies[1] = particle2;
	ParticleData& pd = model.getParticles();

	const Vector3r& x1_0 = pd.getPosition0(particle1);
	const Vector3r& x2_0 = pd.getPosition0(particle2);

	m_restLength = (x2_0 - x1_0).norm();                                // 定义边之间的约束为点之间的距离

	return true;
}

bool PBD::SeamConstraint::solvePositionConstraint(SimulationModel & model)
{
	ParticleData& pd = model.getParticles();

	const unsigned i1 = m_bodies[0];
	const unsigned i2 = m_bodies[1];

	Vector3r& x1 = pd.getPosition(i1);
	Vector3r& x2 = pd.getPosition(i2);
	const Real invMass1 = pd.getInvMass(i1);
	const Real invMass2 = pd.getInvMass(i2);

	Vector3r corr1, corr2;
	const bool res = PositionBasedDynamics::solve_DistanceConstraint(
		x1, invMass1, x2, invMass2,
		m_restLength, model.getClothStiffness(), model.getClothStiffness(), corr1, corr2);

	if (res)
	{
		if (invMass1 != 0.0)
			x1 += corr1;
		if (invMass2 != 0.0)
			x2 += corr2;
	}
	return true;
}

//////////////////////////////////////////////////////////////////////////
// ParticleRigidBodyContactConstraint
//////////////////////////////////////////////////////////////////////////
bool ParticleRigidBodyContactConstraint::initConstraint(SimulationModel & model, unsigned int first, unsigned int second,
	const unsigned int particleIndex, const unsigned int rbIndex,
	const Vector3r & cp1, const Vector3r & cp2, const Vector3r & rbCenter,
	const Vector3r & normal, const Real dist,
	const Real restitutionCoeff, const Real stiffness, const Real frictionCoeff)
{
	m_stiffness = stiffness;
	m_frictionCoeff = frictionCoeff;

	m_bodies[0] = particleIndex;
	m_bodies[1] = rbIndex;
	m_triModelNum = first;

	SimulationModel::RigidBodyVector& rbs = model.getRigidBodies();
	ParticleData& pd = model.getParticles();

	vector<unsigned int> collisionIndex = pd.getCollisionIndex(particleIndex);
	bool haveK = false;
	for (int i = 0; i < collisionIndex.size(); i++)
	{
		if (collisionIndex[i] == second)
		{
			haveK = true; break;
		}
	}
	if (!haveK) collisionIndex.push_back(second);

	RigidBody& rb = *rbs[m_bodies[1]];

	m_sum_impulses = 0.0;

	return PositionBasedRigidBodyDynamics::init_ParticleRigidBodyContactConstraint(
		pd.getInvMass(particleIndex),
		pd.getPosition(particleIndex),
		pd.getVelocity(particleIndex),
		0.0,
		rbCenter,
		Vector3r(0.0,0.0,0.0),
		Matrix3r(),
		Vector3r(0.0, 0.0, 0.0),
		cp1, cp2, normal, restitutionCoeff,
		m_constraintInfo);
}

bool ParticleRigidBodyContactConstraint::solveVelocityConstraint(SimulationModel & model)
{

	SimulationModel::RigidBodyVector& rbs = model.getRigidBodies();
	ParticleData& pd = model.getParticles();

	RigidBody& rb = *rbs[m_bodies[1]];

	Vector3r corr_v1, corr_v2;
	Vector3r corr_omega2;
	const bool res = PositionBasedRigidBodyDynamics::velocitySolve_ParticleRigidBodyContactConstraint(
		pd.getInvMass(m_bodies[0]),
		pd.getPosition(m_bodies[0]),
		pd.getVelocity(m_bodies[0]),
		0.0,
		Vector3r(0.0, 0.0, 0.0),
		Vector3r(0.0, 0.0, 0.0),
		Matrix3r(),
		Vector3r(0.0, 0.0, 0.0),
		m_stiffness,
		m_frictionCoeff,
		m_sum_impulses,
		m_constraintInfo,
		corr_v1,
		corr_v2,
		corr_omega2);

	if (res)
	{
		if (pd.getMass(m_bodies[0]) != 0.0)
		{
			if (pd.getCollisionIndex(m_bodies[0]).size() > 1)
			{
				pd.getVelocity(m_bodies[0]) = Vector3r(0.0, 0.0, 0.0);
				pd.getPosition(m_bodies[0]) = m_constraintInfo.col(1);
			}
			else
			{
				pd.getPosition(m_bodies[0]) = m_constraintInfo.col(1);
				pd.getVelocity(m_bodies[0]) = corr_v1 * 0.01;    //0.01 for PR   0.032


				Real slideFrictionCoeff = model.getTriangleModels().at(m_triModelNum)->getSlideFrictionCoeff();
				Vector3r collPoint = m_constraintInfo.col(1);
				Vector3r prePoint = pd.getLastPosition(m_bodies[0]);
				Vector3r change = prePoint - collPoint;
				Vector3r normal = m_constraintInfo.col(2);
				Vector3r n1 = normal * (normal.dot(change));
				Vector3r n2 = change - n1;
				pd.getPosition(m_bodies[0]) = collPoint + n2 * 0.84 * slideFrictionCoeff;
			}
		}
	}
	return res;
}

//////////////////////////////////////////////////////////////////////////
// ParticlesContactConstraint
//////////////////////////////////////////////////////////////////////////
bool ParticlesContactConstraint::initConstraint(SimulationModel& model, unsigned int first, unsigned int second, const unsigned int particleIndex, const unsigned int triIndex1,
	const unsigned int triIndex2, const unsigned int triIndex3, const Vector3r& cp1, const Vector3r& cp2,
	const Vector3r& normal, const Real dist, const Real restitutionCoeff, const Real stiffness, const Real frictionCoeff)
{
	m_stiffness = stiffness;
	m_frictionCoeff = frictionCoeff;

	m_bodies[0] = particleIndex;
	m_bodies[1] = triIndex1;
	m_bodies[2] = triIndex2;
	m_bodies[3] = triIndex3;
	ParticleData& pd = model.getParticles();
	vector<unsigned int> collisionIndex = pd.getCollisionIndex(particleIndex);
	bool haveK = false;
	for (int i = 0; i < collisionIndex.size(); i++)
	{
		if (collisionIndex[i] == second)
		{
			haveK = true; break;
		}
	}
	if (!haveK) collisionIndex.push_back(second);

	m_sum_impulses = 0.0;

	return PositionBasedRigidBodyDynamics::init_ParticlesContactConstraint(
		pd.getInvMass(particleIndex),
		pd.getPosition(particleIndex),
		pd.getVelocity(particleIndex),
		pd.getInvMass(triIndex1),
		pd.getPosition(triIndex1),
		pd.getVelocity(triIndex1),
		pd.getInvMass(triIndex2),
		pd.getPosition(triIndex2),
		pd.getVelocity(triIndex2),
		pd.getInvMass(triIndex3),
		pd.getPosition(triIndex3),
		pd.getVelocity(triIndex3),
		cp1, cp2, normal, restitutionCoeff,
		m_constraintInfo);
}

bool ParticlesContactConstraint::solveVelocityConstraint(SimulationModel& model)
{
	ParticleData& pd = model.getParticles();

	Vector3r corr_v0, corr_v1, corr_v2, corr_v3;
	const bool res = PositionBasedRigidBodyDynamics::velocitySolve_ParticlesContactConstraint(
		pd.getInvMass(m_bodies[0]),
		pd.getPosition(m_bodies[0]),
		pd.getVelocity(m_bodies[0]),
		pd.getInvMass(m_bodies[1]),
		pd.getPosition(m_bodies[1]),
		pd.getVelocity(m_bodies[1]),
		pd.getInvMass(m_bodies[2]),
		pd.getPosition(m_bodies[2]),
		pd.getVelocity(m_bodies[2]),
		pd.getInvMass(m_bodies[3]),
		pd.getPosition(m_bodies[3]),
		pd.getVelocity(m_bodies[3]),
		m_stiffness,
		m_frictionCoeff,
		m_sum_impulses,
		m_constraintInfo,
		corr_v0,
		corr_v1,
		corr_v2,
		corr_v3);
	if (res)
	{	
		Real threshold = 0.4;
		Real coeff1 = 10, coeff2 = 10;
		if (pd.getMass(m_bodies[0]) != 0.0)
		{
			if (pd.getCollisionIndex(m_bodies[0]).size() > 1)
			{
				pd.getVelocity(m_bodies[0]) = Vector3r(0.0, 0.0, 0.0);
			}
			else
			{
				pd.getVelocity(m_bodies[0]) += corr_v0 * coeff1;
				if (corr_v0.norm() < threshold)                     //范围0.05~0.5
				{
					corr_v0 = corr_v0.normalize();
					pd.getVelocity(m_bodies[0]) += corr_v0 * threshold;
				}
			}
		}
		if (pd.getMass(m_bodies[1]) != 0.0)
		{
			pd.getVelocity(m_bodies[1]) -= corr_v1 * coeff2;
			if (pd.getCollisionIndex(m_bodies[1]).size() > 1)
			{
				pd.getVelocity(m_bodies[1]) = Vector3r(0.0, 0.0, 0.0);
			}
		}
		if (pd.getMass(m_bodies[2]) != 0.0)
		{
			pd.getVelocity(m_bodies[2]) -= corr_v2 * coeff2;
			if (pd.getCollisionIndex(m_bodies[2]).size() > 1)
			{
				pd.getVelocity(m_bodies[2]) = Vector3r(0.0, 0.0, 0.0);
			}
		}
		if (pd.getMass(m_bodies[3]) != 0.0)
		{
			pd.getVelocity(m_bodies[3]) -= corr_v3 * coeff2;
			if (pd.getCollisionIndex(m_bodies[3]).size() > 1)
			{
				pd.getVelocity(m_bodies[3]) = Vector3r(0.0, 0.0, 0.0);
			}
		}
	}
	return res;
}
