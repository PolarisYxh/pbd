#include "TimeIntegration.h"

using namespace PBD;


// ----------------------------------------------------------------------------------------------
void TimeIntegration::semiImplicitEuler(
	const Real h,
	const Real mass,
	Vector3r& position,
	Vector3r& velocity,
	const Vector3r& acceleration)
{
	if (mass != 0.0)
	{
		velocity += acceleration * h;
		position += velocity * h;
	}
}

// ----------------------------------------------------------------------------------------------
void TimeIntegration::velocityUpdateFirstOrder(
	const Real h,
	const Real mass,
	const Vector3r& position,
	const Vector3r& oldPosition,
	Vector3r& velocity)
{
	if (mass != 0.0)
		velocity =  (position - oldPosition)*(1.0 / h);
}


// ----------------------------------------------------------------------------------------------
void TimeIntegration::velocityUpdateSecondOrder(
	const Real h,
	const Real mass,
	const Vector3r & position,
	const Vector3r & oldPosition,
	const Vector3r & positionOfLastStep,
	Vector3r & velocity)
{
	if (mass != 0.0)
		velocity = (position * 1.5 - oldPosition * 2.0 + positionOfLastStep * 0.5) * (1.0 / h);
}
