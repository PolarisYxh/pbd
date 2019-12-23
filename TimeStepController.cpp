#include "TimeStepController.h"
#include "TimeManager.h"
#include "PositionBasedRigidBodyDynamics.h"
#include "TimeIntegration.h"
#include <iostream>
#include "PositionBasedDynamics.h"
#include "Timing.h"
#include "ConfigurationLoader.h"

using namespace PBD;
using namespace std;
using namespace Utilities;

TimeStepController::TimeStepController()
{
	m_velocityUpdateMethod = 0;
	m_maxIter = 15;           //5
	m_maxIterVel = 15;        //5
	m_collisionDetection = NULL;
	m_gravity = Vector3r(0.0, -9.81, 0.0);
}

TimeStepController::~TimeStepController(void)
{
}

void TimeStepController::step(SimulationModel& model, Configuration& conf)
{
	//START_TIMING("每帧总时长：");
	TimeManager* tm = TimeManager::getCurrent();
	const Real h = tm->getTimeStepSize();
	clearAccelerations(model);

	ParticleData& pd = model.getParticles();

	#pragma omp parallel  default(shared)
	{
		//粒子速度更新，并添加速度阻尼效果
		#pragma omp for schedule(static) 
		for (int i = 0; i < model.getTriangleModels().size(); i++)
		{
			double airDragCoeff = model.getTriangleModels().at(i)->getAirDragCoeff();
			unsigned int indexoffset = model.getTriangleModels().at(i)->getIndexOffset();
			unsigned int pdNum = model.getTriangleModels().at(i)->getParticleMesh().numVertices();
			for (unsigned int j = 0; j < pdNum; j++)
			{
				unsigned int index = j + indexoffset;
				Vector3r v = pd.getVelocity(index).normalize();
				Real v0 = pd.getVelocity(index).norm();
				Real mass = pd.getMass(index) / pdNum;
				Real area = pd.getArea(index);
				Vector3r airDragForce = v * 0.5 * airDragCoeff * area * v0 * v0 * 15;

				Vector3r acceleration = (pd.getAcceleration(index) * mass - airDragForce) * (pd.getInvMass(index)) * pdNum;
				pd.getLastPosition(index) = pd.getOldPosition(index);
				pd.getOldPosition(index) = pd.getPosition(index);
				TimeIntegration::semiImplicitEuler(h, pd.getMass(index)/pdNum, pd.getPosition(index), pd.getVelocity(index), acceleration);
			}
		}
	}

	//START_TIMING("约束投影");
	positionConstraintProjection(model);
	//STOP_TIMING_AVG_PRINT;

	#pragma omp parallel  default(shared)
	{
		#pragma omp for schedule(static) 
		//更新当前速度
		for (int i = 0; i < (int)pd.size(); i++)
		{
			if (m_velocityUpdateMethod == 0)
				TimeIntegration::velocityUpdateFirstOrder(h, pd.getMass(i), pd.getPosition(i), pd.getOldPosition(i), pd.getVelocity(i));
			else
				TimeIntegration::velocityUpdateSecondOrder(h, pd.getMass(i), pd.getPosition(i), pd.getOldPosition(i), pd.getLastPosition(i), pd.getVelocity(i));
		}
	}

	if (m_collisionDetection)
	{
		//START_TIMING("碰撞检测");
		m_collisionDetection->collisionDetection(model, conf);
		//STOP_TIMING_AVG_PRINT;
		//		STOP_TIMING_AVG;
	}

	//START_TIMING("碰撞处理");
	velocityConstraintProjection(model);
	//STOP_TIMING_AVG_PRINT;
	//平滑
	positionConstraintProjection2(model);

	// compute new time	
	tm->setTime(tm->getTime() + h);
	//	STOP_TIMING_AVG;
	//STOP_TIMING_AVG_PRINT;
	#pragma omp parallel  default(shared)
	{
		//粒子速度更新，并添加速度阻尼效果
		#pragma omp for schedule(static) 
		for (int i = 0; i < model.getTriangleModels().size(); i++)
		{
			double airDragCoeff = conf.getClothCoeff().at(i)[4];
			unsigned int indexoffset = model.getTriangleModels().at(i)->getIndexOffset();
			unsigned int pdNum = model.getTriangleModels().at(i)->getParticleMesh().numVertices();
			for (unsigned int j = 0; j < pdNum; j++)
			{
				unsigned int index = j + indexoffset;
				if (m_velocityUpdateMethod == 0)
					TimeIntegration::velocityUpdateFirstOrder(h, pd.getMass(index), pd.getPosition(index), pd.getOldPosition(index), pd.getVelocity(index));
				else
					TimeIntegration::velocityUpdateSecondOrder(h, pd.getMass(index), pd.getPosition(index), pd.getOldPosition(index), pd.getLastPosition(index), pd.getVelocity(index));
			}
		}
	}
}

void TimeStepController::clearAccelerations(SimulationModel & model)
{
	//////////////////////////////////////////////////////////////////////////
	// particle model
	//////////////////////////////////////////////////////////////////////////

	ParticleData& pd = model.getParticles();
	const unsigned int count = pd.size();
	for (unsigned int i = 0; i < count; i++)
	{
		// Clear accelerations of dynamic particles
		if (pd.getMass(i) != 0.0)
		{
			Vector3r& a = pd.getAcceleration(i);
			a = m_gravity;
		}
	}
}

void TimeStepController::reset()
{

}

void TimeStepController::positionConstraintProjection(SimulationModel & model)
{
	unsigned int iter = 0;

	// init constraint groups if necessary                                                  
	//model.initConstraintGroups();

	SimulationModel::ConstraintVector& constraints = model.getConstraints();
	//SimulationModel::ConstraintGroupVector& groups = model.getConstraintGroups();

	/*for (unsigned int group = 0; group < groups.size(); group++)
	{
		const int groupSize = (int)groups[group].size();
		#pragma omp parallel if(groupSize > MIN_PARALLEL_SIZE) default(shared)
		{
			#pragma omp for schedule(static) 
			for (int i = 0; i < groupSize; i++)
			{
				const unsigned int constraintIndex = groups[group][i];
				constraints[constraintIndex]->resetLambda();
			}
		}
	}*/
	for (int i = 0; i < constraints.size(); i++)
	{
		constraints[i]->resetLambda();
	}
	while (iter < m_maxIter)
	{
		//for (unsigned int group = 0; group < groups.size(); group++)
		//{
		//	const int groupSize = (int)groups[group].size();
		//	#pragma omp parallel if(groupSize > MIN_PARALLEL_SIZE) default(shared)
		//	{
		//		#pragma omp for schedule(static) 
		//		for (int i = 0; i < groupSize; i++)
		//		{
		//			const unsigned int constraintIndex = groups[group][i];

		//			constraints[constraintIndex]->solvePositionConstraint(model);            //根据公式求Δx1,Δx2
		//																					//DistanceConstrain:   Constraints.cpp:888
		//																					 //FEMTriangleConstrain:Constraints.cpp:1075
		//																					 //DihedralConstraint:  Constraints.cpp:937
		//		}
		//	}
		//}
		#pragma omp parallel default(shared)
		{
			#pragma omp for schedule(static) 
			for (int i = 0; i < constraints.size(); i++)
			{ 
				constraints[i]->solvePositionConstraint(model);
			}
		}

		iter++;
	}
}
void TimeStepController::positionConstraintProjection2(SimulationModel& model)
{
	unsigned int iter = 0;

	SimulationModel::ConstraintVector& constraints = model.getConstraints();
	while (iter < 5)
	{
		#pragma omp parallel default(shared)
		{
			#pragma omp for schedule(static) 
			for (int i = 0; i < constraints.size(); i++)
			{
				constraints[i]->solvePositionConstraint(model);
				if (constraints[i]->getTypeId() == 2)
				{
					constraints[i]->solvePositionConstraint(model);
					constraints[i]->solvePositionConstraint(model);
					//constraints[i]->solvePositionConstraint(model);
					//constraints[i]->solvePositionConstraint(model);
				}
			}
		}
		iter++;
	}
}


void TimeStepController::velocityConstraintProjection(SimulationModel & model)
{
	unsigned int iter = 0;
	SimulationModel::ParticleRigidBodyContactConstraintVector& particleRigidBodyContacts = model.getParticleRigidBodyContactConstraints();
	SimulationModel::ParticlesContactConstraintVector& particlesContacts = model.getParticlesContactConstraints();

	while (iter < m_maxIterVel)
	{
		for (unsigned int i = 0; i < particleRigidBodyContacts.size(); i++)
			particleRigidBodyContacts[i].solveVelocityConstraint(model);

		for (unsigned int i = 0; i < particlesContacts.size(); i++)
			particlesContacts[i].solveVelocityConstraint(model);

		ParticleData & pd = model.getParticles();
		for (int i = 0; i < pd.size(); i++)
		{
			pd.getCollisionIndex(i).clear();
		}

		iter++;
	}
}

void TimeStepController::setCollisionDetection(SimulationModel & model, CollisionDetection * cd)
{
	m_collisionDetection = cd;
	m_collisionDetection->setContactCallback(contactCallbackFunction, &model);
}

CollisionDetection* TimeStepController::getCollisionDetection()
{
	return m_collisionDetection;
}

void TimeStepController::contactCallbackFunction(unsigned int first, unsigned int second, const unsigned int contactType, const unsigned int bodyIndex1, const unsigned int bodyIndex2,
	const unsigned int bodyIndex3, const unsigned int bodyIndex4,
	const Vector3r & cp1, const Vector3r & cp2, const Vector3r & rbCenter,
	const Vector3r & normal, const Real dist,
	const Real restitutionCoeff, const Real frictionCoeff, void* userData)
{
	SimulationModel* model = (SimulationModel*)userData;
	if (contactType == CollisionDetection::ParticleRigidBodyContactType)
		model->addParticleRigidBodyContactConstraint(first, second, bodyIndex1, bodyIndex2, cp1, cp2, rbCenter, normal, dist, restitutionCoeff, frictionCoeff);
	else if (contactType == CollisionDetection::ParticlesContactType)
		model->addParticlesContactConstraint(first, second, bodyIndex1, bodyIndex2, bodyIndex3, bodyIndex4, cp1, cp2, normal, dist, restitutionCoeff, frictionCoeff);
}

