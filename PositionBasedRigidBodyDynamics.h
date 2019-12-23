#ifndef POSITION_BASED_RIGID_BODY_DYNAMICS_H
#define POSITION_BASED_RIGID_BODY_DYNAMICS_H

#include "Common.h"
#include "MyVector.h"
#include "MyMatrix.h"
using namespace Utilities;
// ------------------------------------------------------------------------------------
namespace PBD
{
	class PositionBasedRigidBodyDynamics
	{
		// -------------- Position Based Rigid Body Dynamics  -----------------------------------------------------
	private:
		static void computeMatrixK(
			const Vector3r& connector,
			const Real invMass,
			const Vector3r& x,
			const Matrix3r& inertiaInverseW,
			Matrix3r& K);

		static void computeMatrixK(
			const Vector3r& connector0,
			const Vector3r& connector1,
			const Real invMass,
			const Vector3r& x,
			const Matrix3r& inertiaInverseW,
			Matrix3r& K);

	public:
		static bool init_ParticleRigidBodyContactConstraint(
			const Real invMass0,							// inverse mass is zero if body is static
			const Vector3r& x0,						// center of mass of body 0
			const Vector3r& v0,						// velocity of body 0
			const Real invMass1,							// inverse mass is zero if body is static
			const Vector3r& x1,						// center of mass of body 1
			const Vector3r& v1,						// velocity of body 1
			const Matrix3r& inertiaInverseW1,		// inverse inertia tensor (world space) of body 1	
			const Vector3r& omega1,					// angular velocity of body 1
			const Vector3r& cp0,						// contact point of body 0
			const Vector3r& cp1,						// contact point of body 1
			const Vector3r& normal,					// contact normal in body 1
			const Real restitutionCoeff,					// coefficient of restitution
			Matrix35r& constraintInfo);

		static bool velocitySolve_ParticleRigidBodyContactConstraint(
			const Real invMass0,							// inverse mass is zero if body is static
			const Vector3r& x0, 						// center of mass of body 0
			const Vector3r& v0,						// velocity of body 0
			const Real invMass1,							// inverse mass is zero if body is static
			const Vector3r& x1, 						// center of mass of body 1
			const Vector3r& v1,						// velocity of body 1
			const Matrix3r& inertiaInverseW1,		// inverse inertia tensor (world space) of body 1
			const Vector3r& omega1,					// angular velocity of body 1
			const Real stiffness,							// stiffness parameter of penalty impulse
			const Real frictionCoeff,						// friction coefficient
			Real& sum_impulses,							// sum of all impulses
			Matrix35r& constraintInfo,		// precomputed contact info
			Vector3r& corr_v0,
			Vector3r& corr_v1, Vector3r& corr_omega1);
		static bool init_ParticlesContactConstraint(
			const Real invMass0,
			const Vector3r& x0,
			const Vector3r& v0,
			const Real invMass1,
			const Vector3r& x1,
			const Vector3r& v1,
			const Real invMass2,
			const Vector3r& x2,
			const Vector3r& v2,
			const Real invMass3,
			const Vector3r& x3,
			const Vector3r& v3,
			const Vector3r& cp0,
			const Vector3r& cp1,
			const Vector3r& normal,
			const Real restitutionCoeff,
			Matrix35r& constraintInfo);

		static bool velocitySolve_ParticlesContactConstraint(
			const Real invMass0,
			const Vector3r& x0,
			const Vector3r& v0,
			const Real invMass1,
			const Vector3r& x1,
			const Vector3r& v1,
			const Real invMass2,
			const Vector3r& x2,
			const Vector3r& v2,
			const Real invMass3,
			const Vector3r& x3,
			const Vector3r& v3,
			const Real stiffness,
			const Real frictionCoeff,
			Real& sum_impulses,
			Matrix35r& constraintInfo,
			Vector3r& corr_v0,
			Vector3r& corr_v1,
			Vector3r& corr_v2,
			Vector3r& corr_v3);
	};
}

#endif