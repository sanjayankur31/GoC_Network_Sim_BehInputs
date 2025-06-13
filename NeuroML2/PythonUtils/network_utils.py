# import math
import pickle as pkl

# from numpy.core import multiarray
import sys

import numpy as np
from scipy.spatial import distance

sys.path.append("../Parameters")

"""
 Helpers:
    - Distributing cells in space
    - Setting up GJ coupling probability and conductance
    - Distributing presynaptic inputs in space and defining connectivity
"""


def set_random_locations(numCells, volume, density, seed=-1):
    """
    Distribute GoCs in volume (uniformly at random).
    Returns number of GoCs and their x,y,z coordinates.
    Also used for MFs and PFs.

    Parameters:
    ===========
    numCells: number of cells to distribute. If 0, then calculated based on
            density and volume.
    volume: length, width, height of simulated volume in microns
            (generated xyz coordinates are within 0 and these limits).
    density: cell density (count/mm3)
    seed:   sSimulation seed. If -1, then seed is not controlled.
            For different fns, this seed is used differently -
            but deterministically - to set up seed for the random
            number generators.
    """
    if seed != -1:
        np.random.seed(seed + 1000)
    x, y, z = volume
    if numCells == 0:
        numCells = int(density * 1e-9 * x * y * z)  # units mm -> um
    xyz = np.random.random(size=[numCells, 3]) * [x, y, z]  # in um

    return numCells, xyz


def get_hetero_GoC_id(
    nGoC: int,
    nGoC_types: int,
    GoC_id_File: str,
    seed: int = -1,
):
    """Create list of nGoC cells which includes equal numbers of cells of
    nGoC_types.

    This randomly selects nGoC_types ids from the provided parameter file
    (which presumably contains cell ids), and creates a list of cells.
    The list of cells represents nGoC_types populations, with nGoC/nGoC_types
    cells in each. So, it will look like:

    [x, x, x, x, y, y, y, y, z, z, z, z..]

    :param nGoC: total number of GoCs
    :type nGoC: int
    :param nGoC_types: number of GoCs per type/population
    :type nGoC_types: int
    :param GoC_id_file: path to file containing all GoC ids
    :type GoC_id_file: str
    :param seed: random seed
    :type seed: int
    :returns: [nGoC, list of ids]

    """
    # distribute GoC types (different channel distributions)
    if seed != -1:
        np.random.seed(seed + 6000)

    # Load ids from pickled parameter file
    # TODO: how was this file generated?
    with open(GoC_id_File, 'rb') as f:
        allP = pkl.load(f, encoding="bytes")  # JSR added bytes
        # allP: list [1, 25, 32, 128 ... 975, 977, 995] n = 52

    # pick nGoC_types from the list
    ids = np.random.choice(allP, nGoC_types)

    # round off nGoC to nearest tens
    nGoC_per_type = int(nGoC / nGoC_types)  # JSR added int() # 8
    nGoC = nGoC_per_type * nGoC_types  # 8 * 5 = 40

    # generate nGoC_types populations, each with nGoC_per_type cells from the
    # chosen ids from the parameter file
    allid = []
    for jj in range(nGoC_types):
        for kk in range(nGoC_per_type):
            allid.append(ids[jj])

    allid.sort()
    # [1, 1, 1, 1, 1, 1, 1, 1 ...
    # 774, 774, 774, 774, 774, 774, 774, 774] n = 40
    # print(allid)

    return nGoC, allid


def GJ_conn(
    GoC_pos,
    prob_type="Boltzmann",
    GJw_type="Szo16_oneGJ",
    nGJ_dend=3,
    dist_SF=1,
    prob_k=1.0,
    seed=-1,
):
    """
    Generate Electrical connectivity matrix between GoCs.
    As connectivity is based on distance-dependent coupling probability and
    strength, need GoC locations (GoC_pos) to compute pairwise distance.
    nGJ_dend is morphology specific for locating GJs in detailed models.
    Modelling as single Gap junction between cells
    (multiple GJs are collapsed into larger conductance).

    Returns:
    list of connected GoC pairs,
    GJ weight (used as multiplicative factor for net conductance)
    and the dendritic location for each GoC on the pair.

    Parameters:
    ===========
    GoC_pos:    x,y,z positions for all gocs (in um)
    prob_type:  Distance-dependent prob function
    GJw_type:   Distance-dependent coupling strength
                'Vervaeke2010' or 'Szo16_oneGJ'
    nGJ_dend:   how many dendritic compartments
                (to compute location of the GJ conductance)
    dist_SF:    divisive scale factor for distances used in GJ weight
                (for changing coupling scale, 1 for no change)
    prob_k:     divisive factor for distances used in GJ prob
                (for changing coupling scale, 1 for no change)
    seed:       Simulation seed -> used to set random number generators
                (deterministically)
    """
    if seed != -1:
        np.random.seed(seed + 3000)
    # nGoC = GoC_pos.shape[0] JSR: not used  # n = 40
    dist_1D = distance.pdist(GoC_pos, "euclidean")  # pairwise distances
    # nGoC x nGoC = 40 x 40 = 1600
    # vector-form distance vector (removes redundancies)
    # numpy.ndarray n = 780
    # nGoC*(nGoC - 1)/2 = 780
    # distrange = dist_1D

    # convert vector-form distance vector to square-form distance matrix
    dist_2D = distance.squareform(dist_1D)

    # get boolean matrix of connected pairs (nGoC x nGoC)
    if prob_type == "Boltzmann":  # TRUE
        isconn_2D = connProb_Boltzmann(dist_1D / prob_k)  # boolean
        # numpy.ndarray n = 40 x 40
    elif prob_type == "Boltzmann_scaled":
        isconn_2D = connProb_Boltzmann_scaled(dist_1D, prob_k)

    # np.nonzero: grab natrix indices of pairs that are non-zero (True)
    ijTrue = np.nonzero(isconn_2D)  # tuple (array0, array1)
    GJ_pairs = np.asarray(ijTrue)  # np array of connected pairs
    # size = (2, 420), nPairs = 420
    # has duplicates [GoC1, GoC2] = [GoC2, GoC1]

    # nPairs x 2 array = [GoC1, GoC2] of each pair
    # remove redundancies and reverse order
    gjp = []
    nPairs = GJ_pairs.shape[1]
    for iPair in range(nPairs):
        if GJ_pairs[0, iPair] < GJ_pairs[1, iPair]:
            gjp.append(GJ_pairs[:, iPair])
    GJ_pairs = np.asarray(gjp)
    # array size = (210, 2)
    # pair #0: GoC1 = GJ_pairs[0][0], GoC2 = GJ_pairs[0][1]

    # GJ conductance as a function of distance
    # GJ weight is a function of coupling coefficient
    dpairs = [dist_2D[GJ_pairs[j, 0], GJ_pairs[j, 1]] for j in range(GJ_pairs.shape[0])]
    dpairs_1D = np.asarray(dpairs)  # distance matrix of connected pairs
    # np array, size=(210,)

    if GJw_type == "Vervaeke2010":  # False
        # list of gj conductance for corresponding pair
        GJ_cond = set_GJ_strength_Vervaeke2010(dpairs_1D, dist_SF=dist_SF)
    elif GJw_type == "Szo16_oneGJ":  # True
        GJ_cond = set_GJ_strength_Szo2016_oneGJ(dpairs_1D, dist_SF=dist_SF)
    GJ_cond[GJ_cond < 0] = 0  # lower limit = 0
    # np array, size=(210,)

    # Numerical normalization such that average total GJ conductance is
    # the same for different distance-dependent scaling
    if dist_SF == 1:
        pass  # no change
    else:
        if GJw_type == "Vervaeke2010":  # False
            GJf = set_GJ_strength_Vervaeke2010(dist_1D, dist_SF=dist_SF)
            GJ0 = set_GJ_strength_Vervaeke2010(dist_1D)
        elif GJw_type == "Szo16_oneGJ":  # True
            GJf = set_GJ_strength_Szo2016_oneGJ(dist_1D, dist_SF=dist_SF)
            GJ0 = set_GJ_strength_Szo2016_oneGJ(dist_1D)
        GJf[GJf < 0] = 0
        GJ0[GJ0 < 0] = 0
        curr_sum = np.mean(np.sum(GJf))
        avg_sum = np.mean(np.sum(GJ0))
        sf = avg_sum / curr_sum
        GJ_cond = GJ_cond * sf  # sf = 1 if dist_SF = 1

    # get dendritic id to locate GJ for each GoC in a connected pair
    dend_id = np.random.randint(nGJ_dend, size=GJ_pairs.shape)
    # random dendrite sigment (0, 1, 2)
    # high = nGJ_dend, size = (210, 2)

    dend_seg = np.random.random(size=GJ_pairs.shape)
    # random [0.0, 1.0)
    # size = (210, 2)

    GJ_loc = np.c_[dend_id, dend_seg]  # concatenate
    # size = (210, 4)
    # col 0,1 = id, col 2,3 = seg
    # NumPy c_ translates slice objects to concatenation along second axis

    """
    # convert to nS for alex:
    GJ_connmat = np.zeros([nGoC, nGoC])
    GJ_connmat[GJ_pairs[:,0],GJ_pairs[:,1]]=GJ_cond
    GJ_connmat[GJ_pairs[:,1],GJ_pairs[:,0]]=GJ_cond
    GJ_connmat = np.round(GJ_connmat) * .94 #nS
    """

    return GJ_pairs, GJ_cond, GJ_loc  # size = (210, 2), (210,), (210, 4)


def connProb_Boltzmann(  # see GJ_conn()
    dist_1D, seed=-1
):
    """
    Return boolean matrix of nGoC x nGoC where True signifies electrical
    connection exists. Coupling probability is distance-dependent Boltzmann
    function - from Vervaeke 2010.

    Parameters:
    ==========
    dist_1D: Pairwise distances between GoCs (flattened into 1D array)
    [these could be scaled effective distances]
    """
    if seed != -1:
        np.random.seed(seed)

    # Vervaeke 2010 Fig 7A
    # P = 0 - 0.9
    # dist_1D < 160 um
    # for dist_1D > 152 this function returns negative probabilties
    connProb = 1e-2 * (-1745 + 1836 / (1 + np.exp((dist_1D - 267) / 39)))
    connProb = np.maximum(connProb, 0)  # JSR

    connGen = np.random.random(size=dist_1D.shape)  # random [0.0, 1.0)

    # symmetric boolean matrix with diag=0 -> GJ or no GJ
    connProb -= connGen
    isconn_2D = distance.squareform(connProb > 0)
    # numpy.ndarray size = (40, 40)
    return isconn_2D  # is connected (True or False)


def connProb_Boltzmann_scaled(  # see GJ_conn()
    dist_1D, probscale=1, seed=-1
):
    """
    Return boolean matrix of nGoC x nGoC where True signifies electrical
    connection exists. Coupling probability is distance-dependent Boltzmann
    function - from Vervaeke 2010.

    Parameters:
    ==========
    dist_1D:    Pairwise distances between GoCs (flattened into 1D array)
    probscale:  Scaling factor for probability
                (doesn't scale distances, just at the end)
    """
    if seed != -1:
        np.random.seed(seed)
    connProb = 1e-2 * (-1745 + 1836 / (1 + np.exp((dist_1D - 267) / 39))) * probscale
    connGen = np.random.random(size=dist_1D.shape)
    # symmetric boolean matrix with diag=0 -> GJ or not
    isconn_2D = distance.squareform((connProb - connGen) > 0)
    return isconn_2D


def set_GJ_strength_Szo2016_oneGJ(  # see GJ_conn()
    dist_1D, dist_SF=1, seed=-1
):
    """
    Return weights to determine total GJ conductance between each
    connected pair. Distance-dependent coupling strength as used in
    Szobozlay 2016. (Exponential fall-off of coupling coefficient,
    convert to GJ_cond as linear scaling)

    Parameters:
    ===========
    dist_1D:    Pairwise distances between GoCs (flattened into 1D array)
    dist_SF:     Scaling factor (divisive) for pairwise distances to change
                coupling scale
                [factor > 1 means more long-range coupling]
    """
    # Coupling Coefficient
    CC = -2.3 + 29.7 * np.exp(-(dist_1D / dist_SF) / 70.4)
    # GJw = np.round(2*CC/5.0)
    GJw = 2 * CC / 5.0
    return GJw


def set_GJ_strength_Vervaeke2010(  # see GJ_conn()
    dist_1D, dist_SF=1
):
    """
    Return weights to determine total GJ conductance between each
    connected pair. Distance-dependent coupling strength as used in
    Vervaeke2010. (Exponential fall-off of coupling coefficient,
    convert to GJ_cond as sum of exponentials)

    Parameters:
    ===========
    dist_1D:    Pairwise distances between GoCs (flattened into 1D array)
    dist_SF:     Scaling factor (divisive) for pairwise distances to change
                coupling scale
                [factor > 1 means more long-range coupling]
    """
    # Coupling Coefficient
    CC = -2.3 + 29.7 * np.exp(-dist_1D / (70.4 * dist_SF))
    GJw = 0.576 * np.exp(CC / 12.4) + 0.00059 * np.exp(CC / 2.79) - 0.564
    return GJw


# --------------------- Presynaptic Inputs -----------------------------
# ----------------------------------------------------------------------


def set_random_inputs(nInp, nGoC, pConn=0.1, nConn=0, seed=-1):
    """
    Randomly connect a population of synaptic inputs to nGoC Golgi Cells,
    with probability of connection pConn with iid draws (default if pConn>0).
    Otherwise by choosing (without replacement) nConn fibres for each GoC
    i.e. same number of synapses for each GoC.
    """
    if seed != -1:
        np.random.seed(seed + 5000)

    if pConn > 0:
        connGen_2D = np.random.random(size=(nInp, nGoC))
        isconn_2D = connGen_2D < pConn
    else:
        isconn_2D = np.zeros((nInp, nGoC))
        for goc in range(nGoC):
            isconn_2D[np.random.permutation(nInp)[0:nConn], goc] = 1

    # convert 2D connectivity matrix to 1D array
    ijTrue = np.nonzero(isconn_2D)  # tuple (array0, array1)
    input_pairs = np.asarray(ijTrue)  # np array of connected pairs
    # size = (2, 717)
    # size = (2, 2966)
    # note, array order is backward to GJ_pairs
    return input_pairs


def get_syn_weights(
    MF_Syn_list,
    # MF_pos,
    # GoC_pos,
    # method='mult',
    conn_wt=1,
):
    """
    Create a list of synaptic weights for all pre-post pairs.
    Currently set all to conn_wt.
    """
    # if method == 'mult':
    syn_wt = np.ones(MF_Syn_list.shape[1], dtype=int) * conn_wt
    # elif method=='local':
    #   syn_wt = np.ones(MF_Syn_list.shape[1])
    return syn_wt


def connect_inputs(
    maxn=0,
    frac=0,
    density=6000,
    volume=[350, 350, 80],
    # mult=0,
    loc_type="random",
    connType="random_prob",
    connProb=0.5,
    connGoC=0,
    connWeight=1,
    connDist=[0],
    GoC_pos=[],
    syn_loc="soma",
    nGJ_dend=3,
    seed=-1,
):
    """
    Generate connectivity from presynaptic inputs to GoCs
    - Distribute inputs in space
    - set up connections (based on probability or fixed no. of connections)
    - prune any connections beyond defined spatial extent, if needed


    Parameters:
    ===========
    maxn:       Maximum number of inputs (if 0, then density and volume are
                used to determine total number of inputs)
    frac:       Fraction of generated inputs (nInputs = maxn*frac)
    density:    number of inputs/mm3, used only if maxn==0
    volume:     length/breadth/height of simulated volume in microns
                (to generate input coordinates)
    mult:       one or multiple synapses (NOT CURRENTLY USED)
    loc_type:   'random' to distribute inputs uniformly in volume
                (rosette code not yet added)
    connType:   'random_prob' (independently connect with fixed prob) or
                'random_sample' (sample postsynaptic partners)
    connProb:   connection probability for each input-GoC pair
                (used if MF_conntype=='random_prob')
    connGoC:    Number of inputs/GoC (used if MF_conntype=='random_sample')
    connWeight: Scale synaptic weights
    connDist:   Allowed extent of spatial connectivity (if 0, no pruning).
                If scalar, net distance used. If 3-element array, then
                separate limits can be applied to x,y,z distances.
    GoC_pos:    x,y,z coordiates of all GoCs - used for pruning
    syn_loc:   'soma' or 'dend', where should synapses be distributed?
    nGJ_dend:   number of dendritic segments (if syn_loc=='dend', also choose
                dendritic segment to put synapse in)


    Returns:
    ===========
    nInp:       number of inputs
    Inp_pos:    input coordiates
    conn_pairs: list of pre-post pairs
    conn_wt:    list of synaptic weights
    conn_loc:   list of synapse location (dendritic segment if applicable)
    """

    if frac == 0:
        return 0, [], [], [], []  # nothing to do

    nInp = int(maxn * frac)
    nInp, Inp_pos = set_random_locations(nInp, volume, density, seed=seed)
    # numpy.ndarray size (nInp, 3)

    nGoC = GoC_pos.shape[0]

    # --- to add code for rosettes

    # Get list of connected pairs
    # JSR: if/elif statements call same code
    # conn_pairs numpy.ndarray size (2, nPairs)
    # conn_pairs[0, iPair]: Input ID
    # conn_pairs[1, iPair]: GoC ID
    if connType == "random_prob":
        # independently connect with probability connProb
        conn_pairs = set_random_inputs(nInp, nGoC, pConn=connProb, nConn=connGoC)
    elif connType == "random_sample":
        # sample connGoC inputs for each GoC
        conn_pairs = set_random_inputs(nInp, nGoC, pConn=connProb, nConn=connGoC)
    nPairs = conn_pairs.shape[1]

    # prune distal pairs
    if connDist[0] > 0:
        # if connDist is [x,y,z], do separate comparision for each axis
        if len(connDist) == 3:
            for jj in range(3):
                if connDist[jj] <= 0:
                    connDist[jj] = 1e9
            withinbounds = []
            for iPair in range(nPairs):
                iInp = conn_pairs[0, iPair]
                iGoC = conn_pairs[1, iPair]
                xd = abs(Inp_pos[iInp, 0] - GoC_pos[iGoC, 0])
                yd = abs(Inp_pos[iInp, 1] - GoC_pos[iGoC, 1])
                zd = abs(Inp_pos[iInp, 2] - GoC_pos[iGoC, 2])
                inside = xd < connDist[0] and yd < connDist[1] and zd < connDist[2]
                withinbounds.append(inside)
        else:
            # if connDist is scalar, compare net distance
            d2 = connDist[0] ** 2
            withinbounds = []
            for iPair in range(nPairs):
                iInp = conn_pairs[0, iPair]
                iGoC = conn_pairs[1, iPair]
                xd = Inp_pos[iInp, 0] - GoC_pos[iGoC, 0]
                yd = Inp_pos[iInp, 1] - GoC_pos[iGoC, 1]
                zd = Inp_pos[iInp, 2] - GoC_pos[iGoC, 2]
                inside = xd**2 + yd**2 + zd**2 < d2
                withinbounds.append(inside)
        conn_pairs = conn_pairs[:, withinbounds]  # boolean array slicing
        nPairs = conn_pairs.shape[1]

    # --- to add code for weight
    # conn_wt numpy.ndarray size (nPairs)
    conn_wt = get_syn_weights(conn_pairs, conn_wt=connWeight)

    # conn_loc numpy.ndarray size (2, nPairs)
    # conn_loc[0, iPair]: dendrite segment: 0, 1, 2
    # conn_loc[1, iPair]: fraction: 0-1
    conn_loc = np.r_[
        np.random.randint(nGJ_dend, size=[1, nPairs]),
        np.random.random(size=[1, nPairs]),
    ]

    # JSR: reverse order of arrays to be consistent with GJ pairs
    conn_pairs2 = np.zeros([nPairs, 2], dtype=int)
    for iPair in range(nPairs):
        conn_pairs2[iPair, 0] = conn_pairs[0, iPair]
        conn_pairs2[iPair, 1] = conn_pairs[1, iPair]

    conn_loc2 = np.zeros([nPairs, 2], dtype=float)
    for iPair in range(nPairs):
        conn_loc2[iPair, 0] = conn_loc[0, iPair]
        conn_loc2[iPair, 1] = conn_loc[1, iPair]

    return nInp, Inp_pos, conn_pairs2, conn_wt, conn_loc2


def connect_inputs_known(  # NOT USED
    nInp,
    Inp_pos,
    # mult=0,
    loc_type="random",
    connType="random_prob",
    connProb=0.5,
    connGoC=0,
    connWeight=1,
    connDist=[0],
    GoC_pos=[],
    syn_loc="soma",
    nGJ_dend=3,
    seed=-1,
):
    """
    Same as connect_inputs except input locations are previously determined
    (parameters nInp and Inp_pos)
    """

    nGoC = GoC_pos.shape[0]

    if connType == "random_prob":
        conn_pairs = set_random_inputs(nInp, nGoC, pConn=connProb, nConn=connGoC)
    elif connType == "random_sample":
        conn_pairs = set_random_inputs(nInp, nGoC, pConn=connProb, nConn=connGoC)
    if connDist[0] > 0:
        # prune distal pairs
        if len(connDist) == 3:
            for jj in range(3):
                if connDist[jj] <= 0:
                    connDist[jj] = 1e9
            conn_pairs = conn_pairs[
                :,
                [
                    (
                        (
                            abs(
                                Inp_pos[conn_pairs[0, jj], 0]
                                - GoC_pos[conn_pairs[1, jj], 0]
                            )
                            < connDist[0]
                        )
                        & (
                            abs(
                                Inp_pos[conn_pairs[0, jj], 1]
                                - GoC_pos[conn_pairs[1, jj], 1]
                            )
                            < connDist[1]
                        )
                        & (
                            abs(
                                Inp_pos[conn_pairs[0, jj], 2]
                                - GoC_pos[conn_pairs[1, jj], 2]
                            )
                            < connDist[2]
                        )
                    )
                    for jj in range(conn_pairs.shape[1])
                ],
            ]
        else:
            conn_pairs = conn_pairs[
                :,
                [
                    (
                        np.power(
                            Inp_pos[conn_pairs[0, jj], 0]
                            - GoC_pos[conn_pairs[1, jj], 0],
                            2,
                        )
                        + np.power(
                            Inp_pos[conn_pairs[0, jj], 1]
                            - GoC_pos[conn_pairs[1, jj], 1],
                            2,
                        )
                        + np.power(
                            Inp_pos[conn_pairs[0, jj], 2]
                            - GoC_pos[conn_pairs[1, jj], 2],
                            2,
                        )
                    )
                    < connDist[0] ** 2
                    for jj in range(conn_pairs.shape[1])
                ],
            ]

    # --- to add code for weight
    conn_wt = get_syn_weights(conn_pairs, conn_wt=connWeight)
    # col 0,1 = dend, col 2,3 = seg
    conn_loc = np.r_[
        np.random.randint(nGJ_dend, size=[1, conn_pairs.shape[1]]),
        np.random.random(size=[1, conn_pairs.shape[1]]),
    ]

    return conn_pairs, conn_wt, conn_loc


def MF_conn(  # NOT USED
    nMF,
    volume,
    density,
    GoC_pos,
    MF_conntype,
    MF_connprob=0.1,
    MF_connGoC=10,
    # MF_wt_type='mult',
    conn_wt=1,
    seed=-1,
):
    """
    [REDUNDANT]
    Set up connectivity from presynaptic input populations to GoC population
    (can be MF/PF - MF used generically)

    Parameters:
    ===========
    nMF:            number of Inputs. If 0, number calculate from
                    volume and density.
    volume:         length/breadth/height of simulated volume in microns
                    (to generate xyz coordinates for inputs)
    density:        density for inputs: number/mm3
    GoC_pos:        xyz coordinates for all GoCs
    MF_conntype:    'random_prob' (independently connect with fixed prob) or
                    'random_sample' (sample postsynaptic partners)
    MF_connprob:    Connection probability for each input-GoC pair
                    (used if MF_conntype=='random_prob')
    MF_connGoC:     Number of presynaptic partners for each GoC
                    (used if MF_conntype=='random_sample')
    conn_wt:        scale for synaptic strength

    Returns:
    ========
    nMF:        number of inputs
    MF_pos:     coordinates of inputs
    MF_pairs:   list of pre-post connected pairs
    MF_GoC_wt:  synaptic weights (to scale conductance)
    """

    if seed != -1:
        np.random.seed(seed + 4000)
    nMF, MF_pos = set_random_locations(nMF, volume, density)
    nGoC = GoC_pos.shape[0]

    # Set up connectivity from presynaptic to GoCs
    if MF_conntype == "random_prob":
        # each connection is independently determined based on probability
        MF_pairs = set_random_inputs(nMF, nGoC, pConn=MF_connprob, nConn=0)
    elif MF_conntype == "random_sample":
        # fixed number of inputs/GoC - sampled uniformly
        MF_pairs = set_random_inputs(nMF, nGoC, pConn=0, nConn=MF_connGoC)
    MF_GoC_wt = get_syn_weights(MF_pairs, conn_wt)

    return nMF, MF_pos, MF_pairs, MF_GoC_wt


def PF_conn(  # NOT USED
    nPF,
    volume,
    PF_density,
    GoC_pos,
    PF_conntype,
    PF_connprob=0.8,
    PF_connGoC=0,
    PF_conndist=[-1],
    seed=-1,
):
    """
    [REDUNDANT]
    Set up connectivity from presynaptic input populations to GoC population
    Similar to MF_conn, except parameter PF_conndist can be used to specify
    x,y,z connectivity extent (e.g. limited extent in AP axis)
    """
    if seed != -1:
        np.random.seed(seed + 7000)
    nPF, PF_pos = set_random_locations(nPF, volume, PF_density)

    nGoC = GoC_pos.shape[0]
    if PF_conntype == "random_prob":
        PF_pairs = set_random_inputs(nPF, nGoC, pConn=PF_connprob, nConn=0)
    elif PF_conntype == "random_sample":
        PF_pairs = set_random_inputs(nPF, nGoC, pConn=0, nConn=PF_connGoC)

    # prune distal pairs - based on x,y,z distances separately
    # (PF_conndist is [xlim, ylim, zlim] in microns)
    if PF_conndist[0] > 0:
        for jj in range(3):
            if PF_conndist[jj] <= 0:
                PF_conndist[jj] = 1e9
        PF_pairs = PF_pairs[
            :,
            [
                (
                    (
                        abs(PF_pos[PF_pairs[0, jj], 0] - GoC_pos[PF_pairs[1, jj], 0])
                        < PF_conndist[0]
                    )
                    & (
                        abs(PF_pos[PF_pairs[0, jj], 1] - GoC_pos[PF_pairs[1, jj], 1])
                        < PF_conndist[1]
                    )
                    & (
                        abs(PF_pos[PF_pairs[0, jj], 2] - GoC_pos[PF_pairs[1, jj], 2])
                        < PF_conndist[2]
                    )
                )
                for jj in range(PF_pairs.shape[1])
            ],
        ]

    return nPF, PF_pos, PF_pairs
