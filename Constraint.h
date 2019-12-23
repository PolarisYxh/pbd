#ifndef _CONSTRAINT_H
#define _CONSTRAINT_H

#include "Common.h"
#define _USE_MATH_DEFINES
#include "math.h"
#include <vector>
#include<MyMatrix.h>
using namespace Utilities;

namespace PBD
{
	class SimulationModel;

	class Constraint
	{
	public:
		unsigned int m_numberOfBodies;
		/** indices of the linked bodies */
		unsigned int* m_bodies;
		//XPBD中λ参数
		Real m_lambda1;
		Real m_lambda2;

		Constraint(const unsigned int numberOfBodies)
		{
			m_numberOfBodies = numberOfBodies;
			m_bodies = new unsigned int[numberOfBodies];
			m_lambda1 = 0.0;
			m_lambda2 = 0.0;
		}

		virtual ~Constraint() { delete[] m_bodies; };
		virtual int& getTypeId() const = 0;

		virtual bool updateConstraint(SimulationModel& model) { return true; };
		virtual bool solvePositionConstraint(SimulationModel& model) { return true; };
		virtual bool solveVelocityConstraint(SimulationModel& model) { return true; };
		void resetLambda() { m_lambda1 = 0.0; m_lambda2 = 0.0; };

	};

	class SeamConstraint : public Constraint
	{
	public:
		static int TYPE_ID;
		Real m_restLength;

		SeamConstraint() : Constraint(2) {}
		virtual int& getTypeId() const { return TYPE_ID; }

		virtual bool initConstraint(SimulationModel& model, const unsigned int particle1, const unsigned int particle2);
		virtual bool solvePositionConstraint(SimulationModel& model);
	};

	class DistanceConstraint : public Constraint
	{
	public:
		static int TYPE_ID;
		Real m_restLength;
		//保存该约束所属的pattern数，即在triangleModel中的次序
		unsigned int m_triModelNum;

		DistanceConstraint() : Constraint(2) {}
		virtual int& getTypeId() const { return TYPE_ID; }

		virtual bool initConstraint(SimulationModel& model, const unsigned int particle1, const unsigned int particle2, unsigned int triModelNum = 0);
		virtual bool solvePositionConstraint(SimulationModel& model);
	};

	class DihedralConstraint : public Constraint
	{
	public:
		static int TYPE_ID;
		Real m_restAngle;
		unsigned int m_triModelNum;

		DihedralConstraint() : Constraint(4) {}
		virtual int& getTypeId() const { return TYPE_ID; }

		virtual bool initConstraint(SimulationModel& model, const unsigned int particle1, const unsigned int particle2,
			const unsigned int particle3, const unsigned int particle4, unsigned int triModelNum = 0);
		virtual bool solvePositionConstraint(SimulationModel& model);
	};

	class ParticleRigidBodyContactConstraint
	{
	public:
		static int TYPE_ID;
		/** indices of the linked bodies */
		unsigned int m_bodies[2];
		Real m_stiffness;
		Real m_frictionCoeff;
		Real m_currentCorrection;
		Real m_sum_impulses;
		Matrix35r m_constraintInfo;
		unsigned int m_triModelNum;

		ParticleRigidBodyContactConstraint() { m_currentCorrection = 0.0; }
		~ParticleRigidBodyContactConstraint() {}
		virtual int& getTypeId() const { return TYPE_ID; }

		bool initConstraint(SimulationModel& model, unsigned int first, unsigned int second, const unsigned int particleIndex, const unsigned int rbIndex,
			const Vector3r& cp1, const Vector3r& cp2, const Vector3r& rbCenter,
			const Vector3r& normal, const Real dist,
			const Real restitutionCoeff, const Real stiffness, const Real frictionCoeff);
		virtual bool solveVelocityConstraint(SimulationModel& model);
	};

	class ParticlesContactConstraint
	{
	public:
		static int TYPE_ID;
		/** indices of the linked bodies */
		unsigned int m_bodies[4];
		Real m_stiffness;
		Real m_frictionCoeff;
		Real m_currentCorrection;
		Real m_sum_impulses;
		Matrix35r m_constraintInfo;

		ParticlesContactConstraint() { m_currentCorrection = 0.0; }
		~ParticlesContactConstraint() {}
		virtual int& getTypeId() const { return TYPE_ID; }

		bool initConstraint(SimulationModel& model, unsigned int first, unsigned int second, const unsigned int particleIndex, const unsigned int triIndex1,
			const unsigned int triIndex2, const unsigned int triIndex3, const Vector3r& cp1, const Vector3r& cp2,
			const Vector3r& normal, const Real dist,
			const Real restitutionCoeff, const Real stiffness, const Real frictionCoeff);
		virtual bool solveVelocityConstraint(SimulationModel& model);
	};
}

#endif
