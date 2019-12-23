#ifndef __MYVECTOR_H__
#define __MYVECTOR_H__
#include <math.h>
#include <cuda_runtime.h>
	struct Vector3r
	{
		// 三维向量坐标
		float x, y, z;
		// 创建一个三维向量向量
		__host__ __device__ Vector3r() {
			x = 0; y = 0; z = 0;
		}

		__host__ __device__ Vector3r(float x_, float y_, float z_){
			x = x_; y = y_; z = z_;
		}

		// 设置三维向量三个方向上的坐标
		__host__ __device__ void set(float x_, float y_, float z_) { x = x_; y = y_; z = z_; }
		__host__ __device__ void setOnVal(int x, float v) { if (x == 0)x += v; if (x == 1)y += v; if (x == 2)z += v; }
		// 三维向量归一化
		__host__ __device__ Vector3r	normalize() const { if (norm() == 0) return *this;else return((*this) / sqrt(x * x + y * y + z * z)); }
		__host__ __device__ float norm() const { return sqrt(x * x + y * y + z * z); }
		__host__ __device__ float      squaredNorm() const { return x * x + y * y + z * z; }
		__host__ __device__ Vector3r cross(Vector3r r)const {
			return Vector3r(
				y * r.z - z * r.y,
				z * r.x - x * r.z,
				x * r.y - y * r.x);
		}
		__host__ __device__ float dot(Vector3r r) const { return x* r.x + y * r.y + z * r.z; }
		__host__ __device__ void setZero() { x = 0.0; y = 0.0; z = 0.0; }


		// BOOL型操作运算符
		__host__ __device__ bool operator == (const Vector3r & v) const { return x == v.x && y == v.y && z == v.z; }
		__host__ __device__ bool operator != (const Vector3r & v) const { return x != v.x || y != v.y || z != v.z; }

		// 常见的运算符
		__host__ __device__ Vector3r& operator = (const Vector3r& v) { x = v.x; y = v.y; z = v.z; return *this; }
		__host__ __device__ Vector3r  operator +  (const Vector3r & v) const { return Vector3r(x + v.x, y + v.y, z + v.z); }
		__host__ __device__ Vector3r& operator += (const Vector3r & v) { x += v.x; y += v.y; z += v.z; return *this; }
		__host__ __device__ Vector3r  operator -  () const { return Vector3r(-x, -y, -z); }
		__host__ __device__ Vector3r  operator -  (const Vector3r & v) const { return Vector3r(x - v.x, y - v.y, z - v.z); }
		__host__ __device__ Vector3r& operator -= (const Vector3r & v) { x -= v.x; y -= v.y; z -= v.z; return *this; }
		__host__ __device__ Vector3r  operator *  (float s) const { return Vector3r(x * s, y * s, z * s); }
		__host__ __device__ Vector3r& operator *= (float s) { x *= s; y *= s; z *= s; return *this; }
		__host__ __device__ Vector3r  operator /  (float s) const {  return (*this)* (1 / s); }
		__host__ __device__ Vector3r& operator /= (float s) {  return (*this) *= (1 / s); }
		__host__ __device__ float  operator []  (int i) const { if (i == 0) return x; else if (i == 1) return y; else if (i == 2) return z; return 0; }
	};


	struct Vector2r
	{
		// 创建一个二维向量向量
		Vector2r(float x_ = 0, float y_ = 0) : x(x_), y(y_) {}

		// 设置二维向量二个方向上的坐标
		void set(float x_, float y_) { x = x_; y = y_; }

		// 二维向量归一化
		Vector2r	normalize() const { return((*this) / norm()); }
		float norm() const { return sqrt(normSquared()); }
		float      normSquared() const { return x * x + y * y; }


		// BOOL型操作运算符
		bool operator == (const Vector2r & v) const { return x == v.x && y == v.y; }
		bool operator != (const Vector2r & v) const { return x != v.x || y != v.y; }

		// 常见的运算符
		Vector2r  operator +  (const Vector2r& v) const { return Vector2r(x + v.x, y + v.y); }
		Vector2r& operator += (const Vector2r & v) { x += v.x; y += v.y;; return *this; }
		Vector2r  operator -  () const { return Vector2r(-x, -y); }
		Vector2r  operator -  (const Vector2r & v) const { return Vector2r(x - v.x, y - v.y); }
		Vector2r& operator -= (const Vector2r & v) { x -= v.x; y -= v.y; return *this; }
		Vector2r  operator *  (float s) const { return Vector2r(x * s, y * s); }
		Vector2r& operator *= (float s) { x *= s; y *= s; return *this; }
		Vector2r  operator /  (float s) const {  return (*this)* (1 / s); }
		Vector2r& operator /= (float s) { return (*this) *= (1 / s); }
		float  operator []  (int i) const { if (i == 0) return x; else if (i == 1) return y; return 0; }

		// 二维向量坐标
		float x, y;
	};

#endif