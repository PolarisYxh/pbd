#ifndef COMMON_H
#define COMMON_H

#define USE_DOUBLE
#define MIN_PARALLEL_SIZE 64

#ifdef USE_DOUBLE
typedef double Real;

#define REAL_MAX DBL_MAX
#define REAL_MIN DBL_MIN
#else
typedef double Real;

#define REAL_MAX FLT_MAX
#define REAL_MIN FLT_MIN
#endif

#define FORCE_INLINE __forceinline



#endif
