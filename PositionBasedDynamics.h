#ifndef POSITION_BASED_DYNAMICS_H
#define POSITION_BASED_DYNAMICS_H

#include "Common.h"
#include "MyVector.h"
#include "MyMatrix.h"
using namespace Utilities;

// ------------------------------------------------------------------------------------
namespace PBD
{
	class PositionBasedDynamics
	{
	public:

		static bool solve_DistanceConstraint(
			const Vector3r& p0, Real invMass0,
			const Vector3r& p1, Real invMass1,
			const Real restLength,
			const Real compressionStiffness,
			const Real stretchStiffness,
			Vector3r& corr0, Vector3r& corr1);

		static bool solve_DihedralConstraint(
			const Vector3r& p0, Real invMass0,		// angle on (p2, p3) between triangles (p0, p2, p3) and (p1, p3, p2)
			const Vector3r& p1, Real invMass1,
			const Vector3r& p2, Real invMass2,
			const Vector3r& p3, Real invMass3,
			const Real restAngle,
			const Real stiffness,
			Vector3r& corr0, Vector3r& corr1, Vector3r& corr2, Vector3r& corr3);
		
	};
}

#endif