#ifndef _TIMEMANAGER_H
#define _TIMEMANAGER_H

#include "Common.h"

namespace Utilities
{
	class TimeManager
	{
	private:
		Real time;
		static TimeManager* current;
		Real h;

	public:
		TimeManager();
		~TimeManager();

		// Singleton
		static TimeManager* getCurrent();
		static void setCurrent(TimeManager* tm);
		static bool hasCurrent();

		Real getTime();
		void setTime(Real t);
		Real getTimeStepSize();
		void setTimeStepSize(Real tss);
	};
}

#endif
