import os
import sys

import lems.api as lems
import neuroml as nml
from pyneuroml import pynml
from pyneuroml.lems import LEMSSimulation

sys.path.append("../PythonUtils/")
sys.path.append("../Parameters/")
sys.path.append("../Mechanisms/")

import math
import pickle as pkl

import generate_inhom_spikearray as gip
import initialise_bimodal_network_params as inp
import numpy as np


def create_GoC_network(
    duration=1000, dt=0.025, seed=100, runid=0, mf=0.2, pf=0.2, hom=True, run=False
):
    ### ---------- Load Params
    params = get_params(runid=runid, mf=mf, pf=pf, hom=hom)

    hom_str = "hom" if hom else "het"
    # Build network to specify cells and connectivity
    datadir = "../Data_BehNet_" + hom_str + "/"

    if not os.path.exists(datadir):
        os.makedirs(datadir)

    simid = (
        "behNet_"
        + format(runid, "05d")
        + "_mf_"
        + format(int(mf * 100), "03d")
        + "_pf_"
        + format(int(pf * 100), "03d")
        + "_"
        + hom_str
    )

    net = nml.Network(id=simid, type="networkWithTemperature", temperature="23 degC")
    net_doc = nml.NeuroMLDocument(id=net.id)
    net_doc.networks.append(net)

    arr_flnm = "../Mechanisms/SpikeArray_MFON_dur_060_080.nml"
    arr_ = pynml.read_neuroml2_file(arr_flnm)

    ### -------------- Component types ------------------------- ###

    # ---- 1. GoC types
    goc_params = np.unique(np.asarray(params["GoC_ParamID"]))

    ctr = 0
    goc_type = []
    for pid in goc_params:
        gocID = "GoC_" + format(pid, "05d")
        goc_type.append(gocID)
        goc_filename = "../Cells/Golgi/{}.cell.nml".format(goc_type[ctr])
        goc_type[ctr] = pynml.read_neuroml2_file(goc_filename).cells[0]
        net_doc.includes.append(nml.IncludeType(href=goc_filename))
        ctr += 1

    # ---- 2. Inputs
    synapse = {}
    inputGen = {}
    inputBG = {}
    inputBeh = {}

    for input_ in params["Inputs"]["types"]:
        # Load synapse type
        Inp = params["Inputs"][input_]
        net_doc.includes.append(nml.IncludeType(href=Inp["syn_type"][0]))  # filename
        if Inp["syn_type"][1] == "ExpThreeSynapse":
            synapse[input_] = pynml.read_neuroml2_file(
                Inp["syn_type"][0]
            ).exp_three_synapses[0]  # Component Type
        elif Inp["syn_type"][1] == "ExpTwoSynapse":
            synapse[input_] = pynml.read_neuroml2_file(
                Inp["syn_type"][0]
            ).exp_two_synapses[0]

        # Set up spike generators
        if (
            Inp["type"] == "switchPoisson"
        ):  # 2 Rates: from 0 to delay, and delay to delay+duration
            inputGen[input_] = (
                pynml.read_lems_file("../Mechanisms/lems_instances_ON.xml")
                .components["Del_" + format(Inp["rate"][0])]
                .id
            )

        elif (
            Inp["type"] == "MF_step"
        ):  # 2 Rates: from 0 to delay, and delay to delay+duration
            print("PF" + format(Inp["id"]))
            inputGen[input_] = (
                pynml.read_lems_file("../Mechanisms/lems_instances_ON.xml")
                .components["MF" + format(Inp["id"])]
                .id
            )

        elif (
            Inp["type"] == "PF_step"
        ):  # 2 Rates: from 0 to delay, and delay to delay+duration
            inputGen[input_] = (
                pynml.read_lems_file("../Mechanisms/lems_instances_ON.xml")
                .components["PF" + format(Inp["id"])]
                .id
            )

        elif Inp["type"] == "MFON":  # beh - ON
            print("MFON")
            inputBeh[input_] = "MFON"

        elif Inp["type"] == "PFON":  # beh - ON
            print("PFON")
            inputBeh[input_] = "PFON"

        elif Inp["type"] == "transientPoisson":
            inputGen[input_] = (
                pynml.read_lems_file("../Mechanisms/lems_instances_ON.xml")
                .components["Transient_350_2s_100ms"]
                .id
            )

        elif Inp["type"] == "poisson":
            inputBG[input_] = nml.SpikeGeneratorPoisson(
                id=input_, average_rate="{} Hz".format(Inp["rate"][0])
            )
            net_doc.spike_generator_poissons.append(inputBG[input_])

        elif Inp["type"] == "constantRate":
            inputBG[input_] = nml.SpikeGenerator(
                id=input_, period="{} ms".format(1000.0 / (Inp["rate"][0]))
            )
            net_doc.spike_generators.append(inputBG[input_])

    # ---- 3.  Gap Junctions
    gj = nml.GapJunction(id="GJ_0", conductance="426pS")  # GoC synapse
    net_doc.gap_junctions.append(gj)

    ### ------------------ Populations ------------------------- ###

    # Create GoC population
    goc_pop = []
    goc = 0
    for pid in range(params["nPop"]):
        goc_pop.append(
            nml.Population(
                id=goc_type[pid].id + "Pop",
                component=goc_type[pid].id,
                type="populationList",
                size=params["nGoC_per_pop"],
            )
        )
        for ctr in range(params["nGoC_per_pop"]):
            inst = nml.Instance(id=ctr)
            goc_pop[pid].instances.append(inst)
            inst.location = nml.Location(
                x=params["GoC_pos"][goc, 0],
                y=params["GoC_pos"][goc, 1],
                z=params["GoC_pos"][goc, 2],
            )
            goc += 1
        net.populations.append(goc_pop[pid])

    ### Background input_ population
    inputBG_pop = {}
    for input_ in inputBG:
        Inp = params["Inputs"][input_]
        inputBG_pop[input_] = nml.Population(
            id=inputBG[input_].id + "_pop",
            component=inputBG[input_].id,
            type="populationList",
            size=Inp["nInp"],
        )
        for ctr in range(Inp["nInp"]):
            inst = nml.Instance(id=ctr)
            inputBG_pop[input_].instances.append(inst)
            inst.location = nml.Location(
                x=Inp["pos"][ctr, 0], y=Inp["pos"][ctr, 1], z=Inp["pos"][ctr, 2]
            )
        net.populations.append(inputBG_pop[input_])

    # create MF as spike array
    bint = 0.1
    time, rate = gip.create_rate(
        "../Parameters/examplebeh.txt",
        np.random.random(size=(7, 1)),
        60,
        80,
        scale=100,
    )

    rate = np.asarray(rate)
    ctr = 1000
    InpProj = {}

    inputGen_pop = {}
    allID = {}
    # Add spike array populations
    for input_ in inputBeh:
        Inp = params["Inputs"][input_]
        inputGen_pop[input_] = []
        allID[input_] = {}

        for mfii in range(Inp["nInp"]):
            arrid = Inp["sample"][mfii]
            currinp = arr_.spike_arrays[arrid]
            allID[input_][mfii] = currinp.id
            inputGen_pop[input_].append(
                nml.Population(
                    id=input_ + "_" + currinp.id + "_pop",
                    component=currinp.id,
                    type="populationList",
                    size=1,
                )
            )
            inst = nml.Instance(id=0)
            inst.location = nml.Location(
                x=Inp["pos"][mfii, 0], y=Inp["pos"][mfii, 1], z=Inp["pos"][mfii, 2]
            )
            inputGen_pop[input_][mfii].instances.append(inst)
            net.populations.append(inputGen_pop[input_][mfii])

    ### ------------ Connectivity

    ### 1. Background  inputs:
    dend_id = [1, 2, 5]
    nDend = 3

    BG_Proj = {}
    for input_ in inputBG:
        Inp = params["Inputs"][input_]
        BG_Proj[input_] = []

        for jj in range(params["nPop"]):
            BG_Proj[input_].append(
                nml.Projection(
                    id="{}_to_{}".format(input_, goc_type[jj].id),
                    presynaptic_population=inputBG_pop[input_].id,
                    postsynaptic_population=goc_pop[jj].id,
                    synapse=synapse[input_].id,
                )
            )
            net.projections.append(BG_Proj[input_][jj])
            ctr = 0
            nSyn = Inp["conn_pairs"][jj].shape[1]
            for syn in range(nSyn):
                pre, goc = Inp["conn_pairs"][jj][:, syn]
                if Inp["syn_loc"] == "soma":
                    conn = nml.ConnectionWD(
                        id=ctr,
                        pre_cell_id="../{}/{}/{}".format(
                            inputBG_pop[input_].id, pre, inputBG[input_].id
                        ),
                        post_cell_id="../{}/{}/{}".format(
                            goc_pop[jj].id, goc, goc_type[jj].id
                        ),
                        post_segment_id="0",
                        post_fraction_along="0.5",
                        weight=Inp["conn_wt"][jj][syn],
                        delay="0 ms",
                    )  # on soma
                elif Inp["syn_loc"] == "dend":
                    conn = nml.ConnectionWD(
                        id=ctr,
                        pre_cell_id="../{}/{}/{}".format(
                            inputBG_pop[input_].id, pre, inputBG[input_].id
                        ),
                        post_cell_id="../{}/{}/{}".format(
                            goc_pop[jj].id, goc, goc_type[jj].id
                        ),
                        post_segment_id=dend_id[int(Inp["conn_loc"][jj][0, syn])],
                        post_fraction_along=dend_id[int(Inp["conn_loc"][jj][1, syn])],
                        weight=Inp["conn_wt"][jj][syn],
                        delay="0 ms",
                    )  # on dend
                BG_Proj[input_][jj].connection_wds.append(conn)
                ctr += 1

    InputGen_Proj = {}
    for input_ in inputBeh:
        Inp = params["Inputs"][input_]
        InputGen_Proj[input_] = []

        ctr = 0
        for jj in range(params["nPop"]):
            InputGen_Proj[input_].append([])
            # initialise all projections
            for mfii in range(Inp["nInp"]):
                InputGen_Proj[input_][jj].append(
                    nml.Projection(
                        id="{}_{}_to_{}".format(input_, mfii, goc_type[jj].id),
                        presynaptic_population=inputGen_pop[input_][mfii].id,
                        postsynaptic_population=goc_pop[jj].id,
                        synapse=synapse[input_].id,
                    )
                )
                net.projections.append(InputGen_Proj[input_][jj][mfii])

            nSyn = Inp["conn_pairs"][jj].shape[1]
            for syn in range(nSyn):
                pre, goc = Inp["conn_pairs"][jj][:, syn]
                if Inp["syn_loc"] == "soma":
                    conn = nml.ConnectionWD(
                        id=ctr,
                        pre_cell_id="../{}/{}/{}".format(
                            inputGen_pop[input_][pre].id, 0, allID[input_][pre]
                        ),
                        post_cell_id="../{}/{}/{}".format(
                            goc_pop[jj].id, goc, goc_type[jj].id
                        ),
                        post_segment_id="0",
                        post_fraction_along="0.5",
                        weight=Inp["conn_wt"][jj][syn],
                        delay="{} ms".format(np.random.random()),
                    )  # on soma
                elif Inp["syn_loc"] == "dend":
                    conn = nml.ConnectionWD(
                        id=ctr,
                        pre_cell_id="../{}/{}/{}".format(
                            inputGen_pop[input_][pre].id, 0, allID[input_][pre]
                        ),
                        post_cell_id="../{}/{}/{}".format(
                            goc_pop[jj].id, goc, goc_type[jj].id
                        ),
                        post_segment_id=dend_id[int(Inp["conn_loc"][jj][0, syn])],
                        post_fraction_along=dend_id[int(Inp["conn_loc"][jj][1, syn])],
                        weight=Inp["conn_wt"][jj][syn],
                        delay="{} ms".format(np.random.random()),
                    )  # on dend
                InputGen_Proj[input_][jj][pre].connection_wds.append(conn)
                ctr += 1

    ### 3. Electrical coupling between GoCs

    GoCCoupling = []
    ctr = 0
    for pre in range(params["nPop"]):
        for post in range(pre, params["nPop"]):
            GoCCoupling.append(
                nml.ElectricalProjection(
                    id="GJ_{}_{}".format(goc_pop[pre].id, goc_pop[post].id),
                    presynaptic_population=goc_pop[pre].id,
                    postsynaptic_population=goc_pop[post].id,
                )
            )
            net.electrical_projections.append(GoCCoupling[ctr])

            gjParams = params["econn_pop"][pre][post - pre]
            for jj in range(gjParams["GJ_pairs"].shape[0]):
                conn = nml.ElectricalConnectionInstanceW(
                    id=jj,
                    pre_cell="../{}/{}/{}".format(
                        goc_pop[pre].id, gjParams["GJ_pairs"][jj, 0], goc_type[pre].id
                    ),
                    pre_segment=dend_id[int(gjParams["GJ_loc"][jj, 0])],
                    pre_fraction_along="0.5",
                    post_cell="../{}/{}/{}".format(
                        goc_pop[post].id, gjParams["GJ_pairs"][jj, 1], goc_type[post].id
                    ),
                    post_segment=dend_id[int(gjParams["GJ_loc"][jj, 1])],
                    post_fraction_along="0.5",
                    synapse=gj.id,
                    weight=gjParams["GJ_wt"][jj],
                )
                GoCCoupling[ctr].electrical_connection_instance_ws.append(conn)
            ctr += 1

    print("Writing files...")
    ### --------------  Write files

    net_doc.includes.append(nml.IncludeType(href=arr_flnm))

    net_filename = simid + ".net.nml"  # NML2 description of network instance
    pynml.write_neuroml2_file(net_doc, net_filename)

    simid = "sim_" + net.id
    ls = LEMSSimulation(simid, duration=duration, dt=dt, simulation_seed=seed)
    ls.assign_simulation_target(net.id)
    ls.include_neuroml2_file(net_filename)  # Network NML2 file
    ls.include_lems_file("../Mechanisms/inputGenerators.xml")
    ls.include_lems_file("../Mechanisms/lems_instances_ON.xml")

    # Specify outputs
    eof0 = "Events_file"
    ls.create_event_output_file(
        eof0, datadir + "%s.spikes.dat" % simid, format="ID_TIME"
    )
    ctr = 0
    for pid in range(params["nPop"]):
        for jj in range(goc_pop[pid].size):
            ls.add_selection_to_event_output_file(
                eof0,
                ctr,
                "{}/{}/{}".format(goc_pop[pid].id, jj, goc_type[pid].id),
                "spike",
            )
            ctr += 1

    eof1 = "Events_file2"
    ls.create_event_output_file(
        eof1, datadir + "%s.inputs.dat" % simid, format="ID_TIME"
    )
    ctr = 0
    for input_ in inputBeh:
        for pid in range(len(inputGen_pop[input_])):
            ls.add_selection_to_event_output_file(
                eof1,
                ctr,
                "{}/{}/{}".format(inputGen_pop[input_][pid].id, 0, allID[input_][pid]),
                "spike",
            )
            ctr += 1

    of0 = "Volts_file"
    ls.create_output_file(of0, datadir + "%s.v.dat" % simid)
    ctr = 0
    for pid in range(params["nPop"]):
        for jj in range(goc_pop[pid].size):
            ls.add_column_to_output_file(
                of0, ctr, "{}/{}/{}/v".format(goc_pop[pid].id, jj, goc_type[pid].id)
            )
            ctr += 1

    # Create Lems file to run
    lems_simfile = ls.save_to_file()

    if run:
        res = pynml.run_lems_with_jneuroml_neuron(
            lems_simfile, max_memory="2G", nogui=True, plot=False
        )
    else:
        res = pynml.run_lems_with_jneuroml_neuron(
            lems_simfile,
            max_memory="2G",
            only_generate_scripts=True,
            compile_mods=False,
            nogui=True,
            plot=False,
        )

    res = True
    return res


def get_params(runid=0, mf=0.2, pf=0.2, hom=False):
    Input_prob = {
        "MF_Step": 0.2,
        "MF_bg": 0.3,
        "PF_Step": 0.2,
        "PF_bg": 0.5,
        "MFON": 0.2,
        "PFON": 0.2,
    }
    nMF = 60
    nPF = 150

    nPop = 1 if hom else 5

    nInputs_max = {
        "MF_Step": 0,
        "MF_bg": nMF,
        "PF_Step": 0,
        "PF_bg": nPF,
        "MFON": nMF,
        "PFON": nPF,
    }
    nInputs_frac = {
        "MF_Step": mf,
        "MF_bg": 1.0,
        "PF_Step": pf,
        "PF_bg": 1.0,
        "MFON": mf,
        "PFON": pf,
    }

    params = inp.get_simulation_params(
        runid,
        nInputs_max=nInputs_max,
        nInputs_frac=nInputs_frac,
        nGoC_pop=nPop,
        Input_prob=Input_prob,
    )
    return params


def behrate(t, time, rate):
    res = 0 * t
    for jj in len(t):
        x = rate[time >= t[jj]]
        res[jj] = x[1]
    return res


if __name__ == "__main__":
    runid = 0

    print("Generating network using parameters for runid=", runid)
    res = create_GoC_network(duration=20000, dt=0.025, seed=123, runid=runid)
    print(res)
