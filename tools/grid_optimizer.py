import numpy as np
import pandas as pd
from munkres import Munkres
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import minimum_spanning_tree
from sklearn.datasets import make_blobs
from sklearn.cluster import KMeans

# Dependencies to other tools
from tools.grid import Grid


class GridOptimizer:
    """
    Object containing methods for optimizing Grid object .

    Attributes
    ----------
    mst_algorithm_linking_hubs (str):
        Name of the minimum spanning tree algorithm used for connecting the
        hubs.

    """

    def __init__(self,
                 mst_algorithm_linking_hubs="Kruskal",
                 ):

        self.mst_algorithm_linking_hubs = mst_algorithm_linking_hubs

    # ------------ MINIMUM SPANNING TREE ALGORITHM ------------ #

    def connect_hubs(self, grid: Grid):
        """
        This method creates links between all meterhubs following
        Prim's or Kruskal minimum spanning tree method depending on the
        mst_algorithm value.

        Parameters
        ----------
        grid (~grids.Grid):
            Grid object
        """

        if self.mst_algorithm_linking_hubs == "Prims":
            self.connect_hubs_using_MST_Prims(grid)
        elif self.mst_algorithm_linking_hubs == "Kruskal":
            self.connect_hubs_using_MST_Kruskal(grid)
        else:
            raise Exception("Invalid value provided for mst_algorithm.")

    def connect_hubs_using_MST_Prims(self, grid: Grid):
        """
        This  method creates links between all meterhubs following
        Prim's minimum spanning tree method. The idea goes as follow:
        a first node is selected and it is connected to the nearest neighbour,
        together they compose the a so-called forest. Then a loop
        runs over all node of the forest, the node that is the closest to
        the forest without being part of it is added to the forest and
        connected to the node of the forest it is the closest to.

        Parameters
        ----------
        grid (~grids.Grid):
            Grid object whose hubs shall be connected.
        """

        # create list to keep track of nodes that have already been added
        # to the forest

        for cluster in grid.get_poles()['cluster'].unique():
            # Create dataframe containing the hubs from the cluster
            # and add temporary column to keep track of wheter the
            # hub has already been added to the forest or not
            hubs = grid.get_poles()[grid.get_poles()['cluster'] == cluster]
            hubs['in_forest'] = [False] * hubs.shape[0]

            # Makes sure that there are at least two meterhubs in cluster
            if hubs[- (hubs['in_forest'])].shape[0] > 0:
                # First, pick one meterhub and add it to the forest by
                # setting its value in 'in_forest' to True
                index_first_forest_meterhub =\
                    hubs[- hubs['in_forest']].index[0]
                hubs.at[index_first_forest_meterhub, 'in_forest'] = True

                # while there are hubs not connected to the forest,
                # find nereast hub to the forest and connect it to the forest
                count = 0    # safety parameter to avoid staying stuck in loop
                while len(hubs[- hubs['in_forest']]) and\
                        count < hubs.shape[0]:

                    # create variables to compare hubs distances and store best
                    # candidates
                    shortest_dist_to_meterhub_outside_forest = (
                        grid.distance_between_nodes(
                            hubs[hubs['in_forest']].index[0],
                            hubs[- hubs['in_forest']].index[0])
                    )
                    index_closest_hub_in_forest = (
                        hubs[hubs['in_forest']].index[0]
                    )
                    index_closest_hub_to_forest =\
                        hubs[- hubs['in_forest']].index[0]

                    # Iterate over all hubs within the forest and over all the
                    # ones outside of the forest and find shortest distance
                    for index_hub_in_forest, row_forest_hub in\
                            hubs[hubs['in_forest']].iterrows():
                        for (index_hub_outside_forest,
                             row_hub_outside_forest) in (
                            hubs[- hubs['in_forest']].iterrows()
                        ):
                            if grid.distance_between_nodes(
                                    index_hub_in_forest,
                                    index_hub_outside_forest) <= (
                                    shortest_dist_to_meterhub_outside_forest
                            ):
                                index_closest_hub_in_forest = (
                                    index_hub_in_forest
                                )
                                index_closest_hub_to_forest =\
                                    index_hub_outside_forest
                                shortest_dist_to_meterhub_outside_forest =\
                                    grid.distance_between_nodes(
                                        index_closest_hub_in_forest,
                                        index_closest_hub_to_forest)
                    # create a link between hub pair
                    grid.add_link(index_closest_hub_in_forest,
                                  index_closest_hub_to_forest)
                    hubs.at[index_closest_hub_to_forest, 'in_forest'] = True
                    count += 1

    def connect_hubs_using_MST_Kruskal(self, grid: Grid):
        """
        This  method creates links between all meterhubs following
        Kruskal's minimum spanning tree method from scpicy.sparse.csgraph.

        Parameters
        ----------
        grid (~grids.Grid):
            Grid object whose hubs shall be connected
        """

        # iterate over all clusters and connect hubs using MST
        for cluster in grid.get_poles()['cluster'].unique():
            hubs = grid.get_poles()[grid.get_poles()['cluster'] == cluster]

            # configure input matrix for calling csr_matrix function
            X = np.zeros((hubs.shape[0], hubs.shape[0]))
            for i in range(hubs.shape[0]):
                for j in range(hubs.shape[0]):
                    if i > j:
                        index_node_i = hubs.index[i]
                        index_node_j = hubs.index[j]

                        X[j][i] = grid.distance_between_nodes(index_node_i,
                                                              index_node_j)
            M = csr_matrix(X)

            # run minimum_spanning_tree_function
            Tcsr = minimum_spanning_tree(M)
            A = Tcsr.toarray().astype(float)

            # Read output matrix and create corresponding links to grid
            for i in range(len(hubs.index)):
                for j in range(len(hubs.index)):
                    if i > j:
                        if A[j][i] > 0:
                            index_node_i = hubs.index[i]
                            index_node_j = hubs.index[j]

                            grid.add_link(index_node_i, index_node_j)
