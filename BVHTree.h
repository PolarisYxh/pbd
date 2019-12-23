#ifndef __BVHTREE_H__
#define __BVHTREE_H__

#include <vector>
#include <functional>
#include <algorithm>
#include <numeric>
#include <queue>
#include <iostream>
#include "Common.h"
#include <array>
#include <list>
#include "AlignedBox3r.h"
#include "cuda_runtime.h"
#include "MyVector.h"

namespace PBD
{
	template  <typename HullType>
	struct BVHTree
	{
	public:
		using TraversalPredicate    = std::function<bool(unsigned int node_index, unsigned int depth)>;
		using TraversalCallback     = std::function<void(unsigned int node_index, unsigned int depth)>;
		using TraversalPriorityLess = std::function<bool(std::array<int, 2>const& nodes)>;

		using Predicate = std::function<void(unsigned int node_index1, unsigned int node_index2, std::vector<unsigned int>& collisionIndex)>;
		using Callback = std::function<void(unsigned int node_index, unsigned int depth)>;

		struct Node
		{
			Node(unsigned int b_, unsigned int n_)
				: begin(b_), n(n_)
			{
				children[0] = -1;
				children[1] = -1;
			}
			Node() = default;

			bool is_leaf() const { return children[0] < 0 && children[1] < 0; }

			//左右子节点索引，-1代表无对应子节点
			int children[2];
			//当前节点包含元素的开始索引
			unsigned int begin;
			//当前节点包含元素个数
			unsigned int n;
		};

		struct QueueItem { unsigned int n, d; };

		using TraversalQueue = std::queue<QueueItem>;

		BVHTree(std::size_t n)
			:m_lst(n) {}

		virtual ~BVHTree() {}

		Node const& node(unsigned int i) const { return m_nodes[i]; }
		HullType & hull(unsigned int i)  { return m_hulls[i]; }
		unsigned int entity(unsigned int i) const { return m_lst[i]; }

		void construct();
		void traverse_depth_first(TraversalPredicate pred, TraversalCallback cb) const;
		void DFS(Predicate pred, Callback cb) const;
		void update();

	protected:

		void construct(unsigned int node, AlignedBox3r & box,
			unsigned int b, unsigned int n, unsigned int count);
		void traverse_depth_first(unsigned int node, unsigned int depth, TraversalPredicate pred, TraversalCallback cb) const;
		void DFS(unsigned int node, unsigned int depth, Predicate pred, Callback cb) const;

		unsigned int add_node(unsigned int b, unsigned int n);

		virtual Vector3r const& entity_position(unsigned int i) const = 0;
		virtual void compute_hull_approx(unsigned int b, unsigned int n, HullType& hull) const {};
		unsigned int getFaceIndex(unsigned int node_index);

		//for GPU
	public:
		std::vector<unsigned int> getList() { return m_lst; };
		std::vector<Node> getNode() { return m_nodes; };
		std::vector<HullType>getHulls() { return m_hulls; };
		std::vector<unsigned int> getIndex32() { return m_index32; };
		std::vector<unsigned int> getLeaf() { return m_leaf; };

	protected:
		std::vector<unsigned int> m_lst;
		std::vector<Node> m_nodes;
		std::vector<HullType> m_hulls;
		std::vector<unsigned int> m_index32;
		std::vector<unsigned int> m_leaf;

	};
#include "BVHTree.inl"
}

#endif