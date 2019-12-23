#include "PositionBasedRigidBodyDynamics.h"
#include <cfloat>
#include <iostream>
#define _USE_MATH_DEFINES
#include "math.h"

#include<MyVector.h>
#include<MyMatrix.h>
#include<iostream>
using namespace std;

using namespace PBD;
using namespace Utilities;

// ----------------------------------------------------------------------------------------------
void PositionBasedRigidBodyDynamics::computeMatrixK(
	const Vector3r& connector,
	const Real invMass,
	const Vector3r& x,
	const Matrix3r& inertiaInverseW,
	Matrix3r& K)
{
	if (invMass != 0.0)
	{
		const Vector3r v = connector - x;
		const Real a = v[0];
		const Real b = v[1];
		const Real c = v[2]; 

		// J is symmetric
		const Real j11 = 0;
		const Real j12 = 0;
		const Real j13 = 0;
		const Real j22 = 0;
		const Real j23 = 0;
		const Real j33 = 0;

		K.set(0, 0, c * c * j22 - b * c * (j23 + j23) + b * b * j33 + invMass);
		K.set(0, 1, -(c * c * j12) + a * c * j23 + b * c * j13 - a * b * j33);
		K.set(0, 2, b * c * j12 - a * c * j22 - b * b * j13 + a * b * j23);
		K.set(1, 0, -(c * c * j12) + a * c * j23 + b * c * j13 - a * b * j33);
		K.set(1, 1, c * c * j11 - a * c * (j13 + j13) + a * a * j33 + invMass);
		K.set(1, 2, -(b * c * j11) + a * c * j12 + a * b * j13 - a * a * j23);
		K.set(2, 0, b * c * j12 - a * c * j22 - b * b * j13 + a * b * j23);
		K.set(2, 1, -(b * c * j11) + a * c * j12 + a * b * j13 - a * a * j23);
		K.set(2, 2, b * b * j11 - a * b * (j12 + j12) + a * a * j22 + invMass);
	}
	else
		K.setZero();
}

// ----------------------------------------------------------------------------------------------
void PositionBasedRigidBodyDynamics::computeMatrixK(
	const Vector3r & connector0,
	const Vector3r & connector1,
	const Real invMass,
	const Vector3r & x,
	const Matrix3r & inertiaInverseW,
	Matrix3r & K)
{
	if (invMass != 0.0)
	{
		const Vector3r v0 = connector0 - x;
		const Real a = v0[0];
		const Real b = v0[1];
		const Real c = v0[2];

		const Vector3r v1 = connector1 - x;
		const Real d = v1[0];
		const Real e = v1[1];
		const Real f = v1[2];

		// J is symmetric
		 Real j11 = inertiaInverseW(0, 0);
		const Real j12 = inertiaInverseW(0, 1);
		const Real j13 = inertiaInverseW(0, 2);
		const Real j22 = inertiaInverseW(1, 1);
		const Real j23 = inertiaInverseW(1, 2);
		const Real j33 = inertiaInverseW(2, 2);

		K.set(0, 0, c * f * j22 - c * e * j23 - b * f * j23 + b * e * j33 + invMass);
		K.set(0, 1, -(c * f * j12) + c * d * j23 + b * f * j13 - b * d * j33);
		K.set(0, 2, c * e * j12 - c * d * j22 - b * e * j13 + b * d * j23);
		K.set(1, 0, -(c * f * j12) + c * e * j13 + a * f * j23 - a * e * j33);
		K.set(1, 1, c * f * j11 - c * d * j13 - a * f * j13 + a * d * j33 + invMass);
		K.set(1, 2,-(c * e * j11) + c * d * j12 + a * e * j13 - a * d * j23);
		K.set(2, 0, b * f * j12 - b * e * j13 - a * f * j22 + a * e * j23);
		K.set(2, 1, -(b * f * j11) + b * d * j13 + a * f * j12 - a * d * j23);
		K.set(2, 2, b * e * j11 - b * d * j12 - a * e * j12 + a * d * j22 + invMass);
	}
	else
		K.setZero();
}


// ----------------------------------------------------------------------------------------------
bool PositionBasedRigidBodyDynamics::init_ParticleRigidBodyContactConstraint(
	const Real invMass0,							// inverse mass is zero if body is static
	const Vector3r & x0,						// center of mass of body 0
	const Vector3r & v0,						// velocity of body 0
	const Real invMass1,							// inverse mass is zero if body is static
	const Vector3r & x1,						// center of mass of body 1
	const Vector3r & v1,						// velocity of body 1
	const Matrix3r & inertiaInverseW1,		// inverse inertia tensor (world space) of body 1
	const Vector3r & omega1,					// angular velocity of body 1
	const Vector3r & cp0,						// contact point of body 0
	const Vector3r & cp1,						// contact point of body 1
	const Vector3r & normal,					// contact normal in body 1
	const Real restitutionCoeff,					// coefficient of restitution
	Matrix35r& constraintInfo)
{
	// constraintInfo contains
	// 0:	contact point in body 0 (global)
	// 1:	contact point in body 1 (global)
	// 2:	contact normal in body 1 (global)
	// 3:	contact tangent (global)
	// 0,4:  1.0 / normal^T * K * normal
	// 1,4: maximal impulse in tangent direction
	// 2,4: goal velocity in normal direction after collision

	// compute goal velocity in normal direction after collision
	const Vector3r r1 = cp1 - x1;

	const Vector3r u1 = v1 + omega1.cross(r1);
	const Vector3r u_rel = v0 - u1;
	const Real u_rel_n = normal.dot(u_rel);

	constraintInfo.setCol(0, cp0);
	constraintInfo.setCol(1, cp1);
	constraintInfo.setCol(2, normal);
//	cout << constraintInfo.col(0)[0] << " " << constraintInfo.col(0)[1] << " " << constraintInfo.col(0)[2] << endl;

	// tangent direction
	Vector3r t = u_rel - normal*u_rel_n;
	Real tl2 = t.squaredNorm();
	if (tl2 > 1.0e-6)
		t *= 1.0 / sqrt(tl2);

	constraintInfo.setCol(3, t);

	// determine K matrix
	Matrix3r K;
	computeMatrixK(cp1, invMass1, x1, inertiaInverseW1, K);
	if (invMass0 != 0.0)
	{
		K.setOn(0, 0, invMass0);
		K.setOn(1, 1, invMass0);
		K.setOn(2, 2, invMass0);
	}

	constraintInfo(0, 4) = 1.0 / normal.dot(K * normal);

	// maximal impulse in tangent direction
	constraintInfo(1, 4) = 1.0 / (t.dot(K * t)) * u_rel.dot(t);

	// goal velocity in normal direction after collision
	constraintInfo(2, 4) = 0.0;
	if (u_rel_n < 0.0)
		constraintInfo(2, 4) = -restitutionCoeff * u_rel_n;

	return true;
}

//--------------------------------------------------------------------------------------------
bool PositionBasedRigidBodyDynamics::velocitySolve_ParticleRigidBodyContactConstraint(
	const Real invMass0,							// inverse mass is zero if body is static
	const Vector3r & x0, 						// center of mass of body 0
	const Vector3r & v0,						// velocity of body 0
	const Real invMass1,							// inverse mass is zero if body is static
	const Vector3r & x1, 						// center of mass of body 1
	const Vector3r & v1,						// velocity of body 1
	const Matrix3r & inertiaInverseW1,		// inverse inertia tensor (world space) of body 1
	const Vector3r & omega1,					// angular velocity of body 1
	const Real stiffness,							// stiffness parameter of penalty impulse
	const Real frictionCoeff,						// friction coefficient
	Real & sum_impulses,							// sum of all impulses
	Matrix35r& constraintInfo,		// precomputed contact info
	Vector3r & corr_v0,
	Vector3r & corr_v1, Vector3r & corr_omega1)
{
	// constraintInfo contains
	// 0:	contact point in body 0 (global)
	// 1:	contact point in body 1 (global)
	// 2:	contact normal in body 1 (global)
	// 3:	contact tangent (global)
	// 0,4:  1.0 / normal^T * K * normal
	// 1,4: maximal impulse in tangent direction
	// 2,4: goal velocity in normal direction after collision

	if ((invMass0 == 0.0) && (invMass1 == 0.0))
		return false;

	const Vector3r & connector0 = constraintInfo.col(0);
	const Vector3r & connector1 = constraintInfo.col(1);
	const Vector3r & normal = constraintInfo.col(2);
	const Vector3r & tangent = constraintInfo.col(3);

	// 1.0 / normal^T * K * normal
	const Real nKn_inv = constraintInfo(0, 4);

	// penetration depth 
	const Real d = normal.dot(connector0 - connector1);

	// maximal impulse in tangent direction
	const Real pMax = constraintInfo(1, 4);

	// goal velocity in normal direction after collision
	const Real goal_u_rel_n = constraintInfo(2, 4);

	const Vector3r r1 = connector1 - x1;
	const Vector3r u1 = v1 + omega1.cross(r1);

	const Vector3r u_rel = v0 - u1;
	const Real u_rel_n = u_rel.dot(normal);
	const Real delta_u_reln = goal_u_rel_n - u_rel_n;

	Real correctionMagnitude = nKn_inv * delta_u_reln;

	if (correctionMagnitude < -sum_impulses)
		correctionMagnitude = -sum_impulses;

	// add penalty impulse to counteract penetration
	if (d < 0.0)
		correctionMagnitude -= stiffness * nKn_inv * d;


	Vector3r p( normal*correctionMagnitude);
	sum_impulses += correctionMagnitude;

	const Real pn = p.dot(normal);
	if (frictionCoeff * pn > pMax)
		p -=  tangent* pMax;
	else if (frictionCoeff * pn < -pMax)
		p +=  tangent* pMax;
	else
		p -=  tangent*frictionCoeff * pn;

	if (invMass0 != 0.0)
	{
		corr_v0 =  p* invMass0;
	}

	if (invMass1 != 0.0)
	{
		corr_v1 =  p* invMass1;
		corr_omega1 = inertiaInverseW1 * (r1.cross(-p));
	}

	return true;
}

// ----------------------------------------------------------------------------------------------
bool PositionBasedRigidBodyDynamics::init_ParticlesContactConstraint(
	const Real invMass0,
	const Vector3r & x0,
	const Vector3r & v0,
	const Real invMass1,
	const Vector3r & x1,
	const Vector3r & v1,
	const Real invMass2,
	const Vector3r & x2,
	const Vector3r & v2,
	const Real invMass3,
	const Vector3r & x3,
	const Vector3r & v3,
	const Vector3r & cp0,						// contact point of body 0
	const Vector3r & cp1,						// contact point of body 1
	const Vector3r & normal,					// contact normal in body 1
	const Real restitutionCoeff,					// coefficient of restitution
	Matrix35r& constraintInfo)
{
	//const Vector3r u1 = (v1 + v2 + v3) / 3.0;    //三角形平均速度
	//const Real u1_n_n = normal.dot(u1);            //三角形在法线方向速度大小
	//const Vector3r u1_t = u1 - normal*u1_n_n;      //切线方向速度
	//const Vector3r u1_n = normal*u1_n_n;           //法线方向
	//const Real mass0 = 1.0 / invMass0;

	//const Real v0_n_n = normal.dot(v0);
	//const Vector3r v0_t = v0 - normal*v0_n_n;     //切线方向
	//const Vector3r v0_n = normal*u1_n_n;          //法线方向
	//const Real mass1 = 1.0 / (1.0 / invMass1 + 1.0 / invMass2 + 1.0 / invMass3);

	//const Vector3r v0_c = ((mass0 - mass1)*v0_n + 2.0*mass1*u1_n) / (mass0 + mass1);
	//const Vector3r u1_c = ((mass1 - mass0)*u1_n + 2.0*mass0*v0_n) / (mass0 + mass1);

	//const Vector3r v0_total = v0_c + v0_t;
	//const Vector3r u1_total = u1_c + u1_t;

	//constraintInfo.col(0) = v0_total;
	//constraintInfo.col(1) = u1_total;
	//constraintInfo.col(2) = cp1;
	//constraintInfo(0,3) = restitutionCoeff;


	const Vector3r u1 = (v1 + v2 + v3) / 3.0;
	const Vector3r u_rel = v0 - u1;
	const Real u_rel_n = normal.dot(u_rel);

	constraintInfo.setCol(0, cp0);
	constraintInfo.setCol(1, cp1);
	constraintInfo.setCol(2, normal);

	//tangent direction
	Vector3r t = u_rel -  normal*u_rel_n;
	Real tl2 = t.squaredNorm();
	if (tl2 > 1.0e-6)
		t *= 1.0 / sqrt(tl2);

	constraintInfo.setCol(3, t);

	Matrix3r K, I;
	I.setIdentity();
	//	I.setOnes();
	computeMatrixK(cp1, invMass1, (x1 + x2 + x3) / 3.0, I, K);
	if (invMass0 != 0.0)
	{
		K.setOn(0, 0, invMass0);
		K.setOn(1, 1, invMass0);
		K.setOn(2, 2, invMass0);
	}

	constraintInfo(0, 4) = 1.0 / (normal.dot(K * normal));

	// maximal impulse in tangent direction
	constraintInfo(1, 4) = 1.0 / (t.dot(K * t)) * u_rel.dot(t);

	// goal velocity in normal direction after collision
	constraintInfo(2, 4) = 0.0;
	if (u_rel_n < 0.0)
		constraintInfo(2, 4) = -restitutionCoeff * u_rel_n;

	return true;
}

bool PositionBasedRigidBodyDynamics::velocitySolve_ParticlesContactConstraint(
	const Real invMass0,							// inverse mass is zero if body is static
	const Vector3r & x0, 						// center of mass of body 0
	const Vector3r & v0,						// velocity of body 0
	const Real invMass1,
	const Vector3r & x1,
	const Vector3r & v1,
	const Real invMass2,
	const Vector3r & x2,
	const Vector3r & v2,
	const Real invMass3,
	const Vector3r & x3,
	const Vector3r & v3,
	const Real stiffness,
	const Real frictionCoeff,
	Real & sum_impulses,
	Matrix35r& constraintInfo,
	Vector3r & corr_v0,
	Vector3r & corr_v1,
	Vector3r & corr_v2,
	Vector3r & corr_v3)
{
	if ((invMass0 == 0.0) && (invMass1 == 0.0))
		return false;
	const Vector3r & connector0 = constraintInfo.col(0);
	const Vector3r & connector1 = constraintInfo.col(1);
	const Vector3r & normal = constraintInfo.col(2);
	const Vector3r & tangent = constraintInfo.col(3);

	// 1.0 / normal^T * K * normal
	const Real nKn_inv = constraintInfo(0, 4);

	// penetration depth 
	const Real d = normal.dot(connector0 - connector1);

	// maximal impulse in tangent direction
	const Real pMax = constraintInfo(1, 4);

	// goal velocity in normal direction after collision
	const Real goal_u_rel_n = constraintInfo(2, 4);

	const Vector3r u_rel = v0 - v1;
	const Real u_rel_n = u_rel.dot(normal);
	const Real delta_u_reln = goal_u_rel_n - u_rel_n;

	Real correctionMagnitude = nKn_inv * delta_u_reln;

	if (correctionMagnitude < -sum_impulses)
		correctionMagnitude = -sum_impulses;

	// add penalty impulse to counteract penetration
	if (d < 0.0)
		correctionMagnitude -= stiffness * nKn_inv * d;


	Vector3r p( normal* correctionMagnitude);
	sum_impulses += correctionMagnitude;

	const Real pn = p.dot(normal);
	if (frictionCoeff * pn > pMax)
		p -=  tangent* pMax;
	else if (frictionCoeff * pn < -pMax)
		p +=  tangent* pMax;
	else
		p -= tangent*frictionCoeff * pn;

	if (invMass0 != 0.0)
	{
		corr_v0 =  p* invMass0;
	}
	if (invMass1 != 0.0)
	{
		corr_v1 = corr_v0;
	}
	if (invMass2 != 0.0)
	{
		corr_v2 = corr_v0;
	}
	if (invMass3 != 0.0)
	{
		corr_v3 = corr_v0;
	}
	return true;
}