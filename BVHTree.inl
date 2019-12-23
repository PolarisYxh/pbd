#include "BoundingSphere.h"
#include <cstring>
#include "omp.h"
#include <iostream>
#include "BVHTree.h"
#include "cuda_runtime.h"
using namespace std;

template <typename HullType> void
BVHTree<HullType>::construct()
{
	m_nodes.clear();
	m_hulls.clear();
	m_index32.clear();
	if (m_lst.empty()) return;

	std::iota(m_lst.begin(), m_lst.end(), 0);             // m_lst={0,1,2...(m_lst.size()-1)}

														  //获取根节点的box区域
	AlignedBox3r box;
	for (auto i = 0u; i < m_lst.size(); ++i)
	{
		for (auto j = 0u; j < 3; j++)
		{
			box.extend(entity_position(i));
		}
	}
	unsigned int count = 0;
	auto ni = add_node(0, static_cast<unsigned int>(m_lst.size()));   //将根节点信息添加KDTree
	construct(ni, box, 0, static_cast<unsigned int>(m_lst.size()),++count);
}

template<typename HullType> void
BVHTree<HullType>::construct(unsigned int node, AlignedBox3r& box, unsigned int b,             //node:根节点   b:起始索引， n:元素个数
	unsigned int n, unsigned int count)
{
	//递归值叶子节点结束
	if (n == 1)
	{
		m_leaf.push_back(node);
		return;
	}
	
		//计算最长轴
	int max_dir = 0;
	Vector3r d = box.diagonal();
	if (d[1] >= d[0] && d[1] >= d[2])                    //确定最长轴
		max_dir = 1;
	else if (d[2] >= d[0] && d[2] >= d[1])
		max_dir = 2;

		//将面根据质心在最长轴的顺序排序
		std::sort(m_lst.begin() + b, m_lst.begin() + b + n,
			[&](unsigned int a, unsigned int b)
	{
		return entity_position(a)[max_dir] < entity_position(b)[max_dir];
	}
	);

	int  hal = n / 2;
	int n0 = add_node(b, hal);
	int n1 = add_node(b + hal, n - hal);

	if (count++ == 4)
	{
		m_index32.push_back(n0);
		m_index32.push_back(n1);
	}

	//添加左、右子节点
	m_nodes[node].children[0] = n0;
	m_nodes[node].children[1] = n1;

	//分裂面
	auto c = 0.5 * (
		entity_position(m_lst[b + hal - 1])[max_dir] +
		entity_position(m_lst[b + hal])[max_dir]);
	auto l_box = box;
	l_box.max().setOnVal(max_dir, c);
	auto r_box = box;
	r_box.min().setOnVal(max_dir, c);

	//递归构造BVH树
	construct(m_nodes[node].children[0], l_box, b, hal, count);
	construct(m_nodes[node].children[1], r_box, b + hal, n - hal, count);
}

template<typename HullType> void
BVHTree<HullType>::traverse_depth_first(TraversalPredicate pred, TraversalCallback cb) const
{
	if (m_nodes.empty())
		return;

	if (pred(0, 0))
		traverse_depth_first(0, 0, pred, cb);
}

template<typename HullType> void
BVHTree<HullType>::traverse_depth_first(unsigned int node_index, unsigned int depth, TraversalPredicate pred, TraversalCallback cb) const
{
	Node const& node = m_nodes[node_index];

	cb(node_index, depth);
	auto is_pred = pred(node_index, depth);
	if (!node.is_leaf() && is_pred)
	{
		traverse_depth_first(m_nodes[node_index].children[0], depth + 1, pred, cb);
		traverse_depth_first(m_nodes[node_index].children[1], depth + 1, pred, cb);
	}
}

template<typename HullType>
inline void BVHTree<HullType>::DFS(Predicate pred, Callback cb) const
{
	if (m_nodes.empty())
		return;

	vector<unsigned int> index;
	index.clear();
	pred(0, 0, index);
	if (index.size())
	{
		//	DFS(0, 0, pred, cb);
		for (int i = 0; i < index.size(); i++)
		{
			DFS(0, index[i], pred, cb);
		}
	}
	index.clear();
}

template<typename HullType>
inline void BVHTree<HullType>::DFS(unsigned int node1_index, unsigned int node2_index, Predicate pred, Callback cb) const
{
	Node node = m_nodes[node1_index];

	vector<unsigned int> index;
	index.clear();
	pred(node1_index, node2_index, index);
	for (int i = 0; i < index.size(); i++)
	{
		cb(node1_index, index.at(i));
	}

	if (!node.is_leaf() && index.size())
	{
		for (int i = 0; i < index.size(); i++)
		{
			DFS(m_nodes[node1_index].children[0], index[i], pred, cb);
			DFS(m_nodes[node1_index].children[1], index[i], pred, cb);
		}
	}
	index.clear();
}


template <typename HullType> unsigned int
BVHTree<HullType>::add_node(unsigned int b, unsigned int n)
{
	HullType hull;
	compute_hull_approx(b, n, hull);                        //计算当前BoundingSphere的中心和半径
	m_hulls.push_back(hull);
	m_nodes.push_back({ b, n });                            //在m_lst中的排序
	return static_cast<unsigned int>(m_nodes.size() - 1);
}

template<typename HullType>
inline unsigned int BVHTree<HullType>::getFaceIndex(unsigned int node_index)
{
	return m_lst[m_nodes[node_index].begin];
}

template <typename HullType> void
BVHTree<HullType>::update()
{
	DFS(
		[&](unsigned int, unsigned int, vector<unsigned int> & index)
	{
		index.push_back(0);
	},
		[&](unsigned int node_index, unsigned int)
	{
		auto const& nd = node(node_index);
		compute_hull_approx(nd.begin, nd.n, m_hulls[node_index]);
	}
	);
}

