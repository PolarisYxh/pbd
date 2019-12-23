#include "PositionBasedDynamics.h"
//#include "MathFunctions.h"
#include <cfloat>
#include <iostream>
using namespace std;

using namespace PBD;

const Real eps = 1e-6;

//////////////////////////////////////////////////////////////////////////
// PositionBasedDynamics
//////////////////////////////////////////////////////////////////////////

bool PositionBasedDynamics::solve_DistanceConstraint(
	const Vector3r& p0, Real invMass0,
	const Vector3r& p1, Real invMass1,
	const Real restLength,
	const Real compressionStiffness,
	const Real stretchStiffness,
	Vector3r& corr0, Vector3r& corr1)
{
	Real wSum = invMass0 + invMass1;
	if (wSum == 0.0)
		return false;

	Vector3r n = p1 - p0;
	//cout << n[0] << " " << n[1] << " " << n[2] << endl;
	Real d = n.norm();
	//cout << d << " " << n[0] * n[0] + n[1] * n[1] + n[2] * n[2] << endl;
	n=n.normalize();
	//cout << n[0] << " " << n[1] << " " << n[2] << endl;
	Vector3r corr;
	if (d < restLength)
		corr = n * compressionStiffness *  (d - restLength) / wSum;
	else
		corr = n * stretchStiffness *  (d - restLength) / wSum;
	//cout<< corr[0] << " " << corr[1] << " " << corr[2] << endl;
	corr0 =  corr*invMass0;
	corr1 = - corr* invMass1;
	//cout << corr0[0] << " " << corr0[1] << " " << corr0[2] << endl;
	return true;
}


bool PositionBasedDynamics::solve_DihedralConstraint(
	const Vector3r & p0, Real invMass0,
	const Vector3r & p1, Real invMass1,
	const Vector3r & p2, Real invMass2,
	const Vector3r & p3, Real invMass3,
	const Real restAngle,
	const Real stiffness,
	Vector3r & corr0, Vector3r & corr1, Vector3r & corr2, Vector3r & corr3)
{
	// derivatives from Bridson, Simulation of Clothing with Folds and Wrinkles
	// his modes correspond to the derivatives of the bending angle arccos(n1 dot n2) with correct scaling

	if (invMass0 == 0.0 && invMass1 == 0.0)
		return false;

	Vector3r e = p3 - p2;
	Real  elen = e.norm();
	if (elen < eps)
		return false;

	Real invElen = 1.0 / elen;

	Vector3r n1 = (p2 - p0).cross(p3 - p0); n1 /= n1.squaredNorm();
	Vector3r n2 = (p3 - p1).cross(p2 - p1); n2 /= n2.squaredNorm();

	Vector3r d0 =  n1*elen;
	Vector3r d1 =  n2* elen;
	Vector3r d2 = n1*((p0 - p3).dot(e)) * invElen  + n2*((p1 - p3).dot(e)) * invElen;
	Vector3r d3 = n1*((p2 - p0).dot(e)) * invElen + n2*((p2 - p1).dot(e)) * invElen;

	n1 = n1.normalize();
	n2 = n2.normalize();
	Real dot = n1.dot(n2);

	if (dot < -1.0) dot = -1.0;
	if (dot > 1.0) dot = 1.0;
	Real phi = acos(dot);

	// Real phi = (-0.6981317 * dot * dot - 0.8726646) * dot + 1.570796;	// fast approximation

	Real lambda =
		invMass0 * d0.squaredNorm() +
		invMass1 * d1.squaredNorm() +
		invMass2 * d2.squaredNorm() +
		invMass3 * d3.squaredNorm();

	if (lambda == 0.0)
		return false;

	// stability
	// 1.5 is the largest magic number I found to be stable in all cases :-)
	//if (stiffness > 0.5 && fabs(phi - b.restAngle) > 1.5)		
	//	stiffness = 0.5;

	lambda = (phi - restAngle) / lambda * stiffness;

	if (n1.cross(n2).dot(e) > 0.0)
		lambda = -lambda;

	corr0 = -d0*invMass0 * lambda;
	corr1 = -d1*invMass1 * lambda ;
	corr2 = -d2*invMass2 * lambda ;
	corr3 = -d3*invMass3 * lambda ;

	return true;
}
