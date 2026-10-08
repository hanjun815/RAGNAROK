/**
 * OKVIS2-X - Open Keyframe-based Visual-Inertial SLAM Configurable with Dense 
 * Depth or LiDAR, and GNSS
 *
 * Copyright (c) 2015, Autonomous Systems Lab / ETH Zurich
 * Copyright (c) 2020, Smart Robotics Lab / Imperial College London
 * Copyright (c) 2025, Mobile Robotics Lab / Technical University of Munich 
 * and ETH Zurich
 *
 * SPDX-License-Identifier: BSD-3-Clause, see LICENESE file for details
 */

/**
 * @file MstGraph.hpp
 * @brief Maximum Spanning Tree Computation.
 * @author Stefan Leutenegger
 */


#ifndef MSTGRAPH_HPP
#define MSTGRAPH_HPP

#include <vector>

namespace okvis {



/// \brief Structure to represent a graph.
/// C++ program for Kruskal's algorithm to find Minimum
/// Spanning Tree of a given connected, undirected and
/// weighted graph
/// adapted from https://www.geeksforgeeks.org/kruskals-minimum-spanning-tree-using-stl-in-c/
struct MstGraph
{
    int V; ///< No. vertices.
    int E; ///< No. edges.
    std::vector< std::pair<int, std::pair<int, int>> > edges; ///< Edges.

    /// \brief To represent Disjoint Sets.
    struct DisjointSets
    {
        int *parent; ///< Parent pointer.
        int *rnk; ///< Rank.
        int n; ///< Number of vertices.

        /// \brief Constructor.
        /// \param n Number of vertices.
        DisjointSets(int n)
        {
            // Allocate memory
            this->n = n;
            parent = new int[size_t(n+1)];
            rnk = new int[size_t(n+1)];

            // Initially, all vertices are in
            // different sets and have rank 0.
            for (int i = 0; i <= n; i++)
            {
                rnk[i] = 0;

                //every element is parent of itself
                parent[i] = i;
            }
        }

        /// \brief Destructor.
        ~DisjointSets() {
            delete[] parent;
            delete[] rnk;
        }

        /// \brief Find the parent of a node 'u'. Path Compression.
        /// \param[in] u Node u.
        /// \return the parent idx.
        int find(int u)
        {
            /* Make the parent of the nodes in the path
               from u--> parent[u] point to parent[u] */
            if (u != parent[u])
                parent[u] = find(parent[u]);
            return parent[u];
        }

        /// \brief Union by rank.
        /// \param x Vertex.
        /// \param y Vertex.
        void merge(int x, int y)
        {
            x = find(x);
            y = find(y);

            /* Make tree with smaller height
               a subtree of the other tree  */
            if (rnk[x] > rnk[y])
                parent[y] = x;
            else // If rnk[x] <= rnk[y]
                parent[x] = y;

            if (rnk[x] == rnk[y])
                rnk[y]++;
        }
    };

    /// Constructor.
    /// \param V No. vertices.
    /// \param E No. edges.
    MstGraph(int V, int E)
    {
        this->V = V;
        this->E = E;
    }

    /// \brief Utility function to add an edge.
    /// \param u Vertex.
    /// \param v Vertex.
    /// \param w Weight.
    void addEdge(int u, int v, int w)
    {
        edges.push_back({w, {u, v}});
    }

    /// \brief Function to find MST using Kruskal's MST algorithm.
    /// @param[out] edgesOut Computed MST edges.
    int kruskalMst(std::vector<std::pair<int, int> > &edgesOut);
};

 /* Function to find MST using Kruskal's MST algorithm, returns weight of the MST*/
int MstGraph::kruskalMst(std::vector<std::pair<int,int>> & edgesOut)
{
    int mst_wt = 0; // Initialize result

    // Sort edges in increasing order on basis of cost
    std::sort(edges.begin(), edges.end());

    // Create disjoint sets
    MstGraph::DisjointSets ds(V);

    // Iterate through all sorted edges
    std::vector< std::pair<int, std::pair<int, int> > >::iterator it;
    for (it=edges.begin(); it!=edges.end(); it++)
    {
        int u = it->second.first;
        int v = it->second.second;

        int set_u = ds.find(u);
        int set_v = ds.find(v);

        // Check if the selected edge is creating
        // a cycle or not (Cycle is created if u
        // and v belong to same set)
        if (set_u != set_v)
        {
            // Current edge will be in the MST
            edgesOut.push_back({u,v});

            // Update MST weight
            mst_wt += it->first;

            // Merge two sets
            ds.merge(set_u, set_v);
        }
    }

    return mst_wt;
}

} // namespace okvis

#endif // MSTGRAPH_HPP
