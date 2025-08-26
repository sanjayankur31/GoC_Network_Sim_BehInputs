# import math
import pickle as pkl

# import numpy as np
import sys
from pathlib import Path

# from scipy.spatial import distance
import neuroml as nml
from pyneuroml import pynml

# from pyneuroml.lems import LEMSSimulation
# import lems.api as lems

sys.path.append("../../PythonUtils")
import initialize_cell_params as icp


def create_GoC(
    runid,
    usefile="cellparams_file.pkl",
    morpho_fname="GoC.cell.nml",
    girk=False,
    # morpho_fname='Golgi_reduced_twoCaPools.cell.nml',
    # morpho_fname="reduced/Golgi_reduced_2CaPools.cell.nml",  # JSR
    # has2Pools=True JSR
    # girk=False JSR
):
    """
    Create GoCs with different channel densities in specified morphology
    with 2 Ca Pools

    Parameters:
    ===========
    usefile: pickle file with selected channel density params
    morpho_fname: filename (NeuroML file) with example cell whose morphology
                  will be imported
    has2Pools:  whether morphology file is of class Cell2CaPools
                (if False, assumes it is Cell)
    girk:     Include girk channel or not (Boolean)
    """

    # load parameters
    # noPar = True
    loadParamsFromFile = False
    if loadParamsFromFile:
        pfile = Path(usefile)
        if pfile.exists():
            print("Reading parameters from file:", usefile)
            file = open(usefile, "rb")
            params_list = pkl.load(file, encoding="bytes")  # JSR
            # print(len(params_list))
            # len(params_list) = 52
            # this file does not contain conductances
            # contains ID numbers for cell file names
            # e.g. GoC_00001.cell.nml, GoC_00025.cell.nml...
            if runid < len(params_list):
                gc_id = params_list[runid]
                gc_file_nml = "Harsha/GoC_" + format(gc_id, "05d") + ".cell.nml"
                print(gc_file_nml)
                pfile = Path(gc_file_nml)
                print(pfile.exists())
            else:
                raise RuntimeError("out of bounds: runid = " + str(runid))
            file.close()
    else:
        p = icp.get_channel_params(runid)

    print(p)

    # return None

    # create NML document for cell
    gocID = "GoC_2Pools_" + format(runid, "05d")  # GoC_2Pools_00000
    # if girk:
    #    gocID = gocID + '_wGIRK'
    cell_doc = nml.NeuroMLDocument(id=gocID)
    goc = nml.Cell2CaPools(id=gocID)
    cell_doc.cell2_ca_poolses.append(goc)

    # import morphology from NML file
    morpho_file = pynml.read_neuroml2_file(morpho_fname)

    # if has2Pools:
    goc.morphology = morpho_file.cell2_ca_poolses[0].morphology

    # else:
    #    goc.morphology= morpho_file.cells[0].morphology
    # cell_doc.includes.append(nml.IncludeType(href=morpho_fname))

    # import channels from NML files
    # dd = '../../'
    dd = "../../../"  # JSR
    na_fname = dd + "Mechanisms/Golgi_Na.channel.nml"  # NaT
    cell_doc.includes.append(nml.IncludeType(href=na_fname))
    nar_fname = dd + "Mechanisms/Golgi_NaR.channel.nml"
    cell_doc.includes.append(nml.IncludeType(href=nar_fname))
    nap_fname = dd + "Mechanisms/Golgi_NaP.channel.nml"
    cell_doc.includes.append(nml.IncludeType(href=nap_fname))

    ka_fname = dd + "Mechanisms/Golgi_KA.channel.nml"
    cell_doc.includes.append(nml.IncludeType(href=ka_fname))
    sk2_fname = dd + "Mechanisms/Golgi_SK2.channel.nml"
    cell_doc.includes.append(nml.IncludeType(href=sk2_fname))
    km_fname = dd + "Mechanisms/Golgi_KM.channel.nml"
    cell_doc.includes.append(nml.IncludeType(href=km_fname))
    kv_fname = dd + "Mechanisms/Golgi_KV.channel.nml"
    cell_doc.includes.append(nml.IncludeType(href=kv_fname))
    bk_fname = dd + "Mechanisms/Golgi_BK.channel.nml"
    cell_doc.includes.append(nml.IncludeType(href=bk_fname))

    cahva_fname = dd + "Mechanisms/Golgi_CaHVA.channel.nml"
    cell_doc.includes.append(nml.IncludeType(href=cahva_fname))
    calva_fname = dd + "Mechanisms/Golgi_CaLVA.channel.nml"
    cell_doc.includes.append(nml.IncludeType(href=calva_fname))

    hcn1f_fname = dd + "Mechanisms/Golgi_HCN1f.channel.nml"
    cell_doc.includes.append(nml.IncludeType(href=hcn1f_fname))
    hcn1s_fname = dd + "Mechanisms/Golgi_HCN1s.channel.nml"
    cell_doc.includes.append(nml.IncludeType(href=hcn1s_fname))
    hcn2f_fname = dd + "Mechanisms/Golgi_HCN2f.channel.nml"
    cell_doc.includes.append(nml.IncludeType(href=hcn2f_fname))
    hcn2s_fname = dd + "Mechanisms/Golgi_HCN2s.channel.nml"
    cell_doc.includes.append(nml.IncludeType(href=hcn2s_fname))

    leak_fname = dd + "Mechanisms/Golgi_lkg.channel.nml"
    cell_doc.includes.append(nml.IncludeType(href=leak_fname))
    calc_fname = dd + "Mechanisms/Golgi_CALC.nml"
    cell_doc.includes.append(nml.IncludeType(href=calc_fname))
    # read_nml = pynml.read_neuroml2_file(calc_fname)
    # calc = read_nml.decaying_pool_concentration_models[0]  # NOT USED??
    calc2_fname = dd + "Mechanisms/Golgi_CALC2.nml"
    cell_doc.includes.append(nml.IncludeType(href=calc2_fname))

    # girk_fname = dd + 'Mechanisms/GIRK.channel.nml'
    # cell_doc.includes.append(nml.IncludeType(href=girk_fname))

    # add biophysical properties
    biophys = nml.BiophysicalProperties2CaPools(id="biophys_" + gocID)
    # goc_2pools_fname = 'Golgi_reduced_twoCaPools.cell.nml'
    goc_2pools_fname = "../Golgi_reduced_twoCaPools.cell.nml"  # JSR
    # same file containing morphology
    # import intracellular properties from this file
    read_nml = pynml.read_neuroml2_file(goc_2pools_fname)
    intracellular = read_nml.cell2_ca_poolses[
        0
    ].biophysical_properties2_ca_pools.intracellular_properties2_ca_pools
    biophys.intracellular_properties2_ca_pools = intracellular

    memb = nml.MembraneProperties2CaPools()
    biophys.membrane_properties2_ca_pools = memb
    goc.biophysical_properties2_ca_pools = biophys

    # pynml.read_neuroml2_file(leak_fname).ion_channel[0].id ->
    # can't read ion channel passive  # WHY??
    idstr = "LeakConductance"  # from nml file
    chan_leak = nml.ChannelDensity(
        ion_channel=idstr,
        cond_density=p["leak_cond"],
        erev="-55 mV",
        ion="non_specific",
        id="Leak",
    )
    memb.channel_densities.append(chan_leak)

    idstr = pynml.read_neuroml2_file(na_fname).ion_channel[0].id
    # reading channel files again to get ID
    chan_nat = nml.ChannelDensity(
        ion_channel=idstr,
        cond_density=p["na_cond"],
        erev="87.39 mV",
        ion="na",
        id="Golgi_Na_soma_group",
        segment_groups="soma_group",
    )
    memb.channel_densities.append(chan_nat)

    idstr = pynml.read_neuroml2_file(nap_fname).ion_channel[0].id
    chan_nap = nml.ChannelDensity(
        ion_channel=idstr,
        cond_density=p["nap_cond"],
        erev="87.39 mV",
        ion="na",
        id="Golgi_NaP_soma_group",
        segment_groups="soma_group",
    )
    memb.channel_densities.append(chan_nap)

    idstr = pynml.read_neuroml2_file(nar_fname).ion_channel[0].id
    chan_nar = nml.ChannelDensity(
        ion_channel=idstr,
        cond_density=p["nar_cond"],
        erev="87.39 mV",
        ion="na",
        id="Golgi_NaR_soma_group",
        segment_groups="soma_group",
    )
    memb.channel_densities.append(chan_nar)

    idstr = pynml.read_neuroml2_file(ka_fname).ion_channel[0].id
    chan_ka = nml.ChannelDensity(
        ion_channel=idstr,
        cond_density=p["ka_cond"],
        erev="-84.69 mV",
        ion="k",
        id="Golgi_KA_soma_group",
        segment_groups="soma_group",
    )
    memb.channel_densities.append(chan_ka)

    idstr = pynml.read_neuroml2_file(sk2_fname).ion_channel_kses[0].id
    chan_sk = nml.ChannelDensity(
        ion_channel=idstr,
        cond_density=p["sk2_cond"],
        erev="-84.69 mV",
        ion="k",
        id="Golgi_KAHP_soma_group",
        segment_groups="soma_group",
    )
    memb.channel_densities.append(chan_sk)

    idstr = pynml.read_neuroml2_file(kv_fname).ion_channel[0].id
    chan_kv = nml.ChannelDensity(
        ion_channel=idstr,
        cond_density=p["kv_cond"],
        erev="-84.69 mV",
        ion="k",
        id="Golgi_KV_soma_group",
        segment_groups="soma_group",
    )
    memb.channel_densities.append(chan_kv)

    idstr = pynml.read_neuroml2_file(km_fname).ion_channel[0].id
    chan_km = nml.ChannelDensity(
        ion_channel=idstr,
        cond_density=p["km_cond"],
        erev="-84.69 mV",
        ion="k",
        id="Golgi_KM_soma_group",
        segment_groups="soma_group",
    )
    memb.channel_densities.append(chan_km)

    idstr = pynml.read_neuroml2_file(bk_fname).ion_channel[0].id
    chan_bk = nml.ChannelDensity(
        ion_channel=idstr,
        cond_density=p["bk_cond"],
        erev="-84.69 mV",
        ion="k",
        id="Golgi_BK_soma_group",
        segment_groups="soma_group",
    )
    memb.channel_densities.append(chan_bk)

    idstr = pynml.read_neuroml2_file(hcn1f_fname).ion_channel[0].id
    chan_h1f = nml.ChannelDensity(
        ion_channel=idstr,
        cond_density=p["hcn1f_cond"],
        erev="-20 mV",
        ion="h",
        id="Golgi_hcn1f_soma_group",
        segment_groups="soma_group",
    )
    memb.channel_densities.append(chan_h1f)

    idstr = pynml.read_neuroml2_file(hcn1s_fname).ion_channel[0].id
    chan_h1s = nml.ChannelDensity(
        ion_channel=idstr,
        cond_density=p["hcn1s_cond"],
        erev="-20 mV",
        ion="h",
        id="Golgi_hcn1s_soma_group",
        segment_groups="soma_group",
    )
    memb.channel_densities.append(chan_h1s)

    idstr = pynml.read_neuroml2_file(hcn2f_fname).ion_channel[0].id
    chan_h2f = nml.ChannelDensity(
        ion_channel=idstr,
        cond_density=p["hcn2f_cond"],
        erev="-20 mV",
        ion="h",
        id="Golgi_hcn2f_soma_group",
        segment_groups="soma_group",
    )
    memb.channel_densities.append(chan_h2f)

    idstr = pynml.read_neuroml2_file(hcn2s_fname).ion_channel[0].id
    chan_h2s = nml.ChannelDensity(
        ion_channel=idstr,
        cond_density=p["hcn2s_cond"],
        erev="-20 mV",
        ion="h",
        id="Golgi_hcn2s_soma_group",
        segment_groups="soma_group",
    )
    memb.channel_densities.append(chan_h2s)

    idstr = pynml.read_neuroml2_file(cahva_fname).ion_channel[0].id
    chan_hva = nml.ChannelDensityNernst(
        ion_channel=idstr,
        cond_density=p["cahva_cond"],
        ion="ca",
        id="Golgi_Ca_HVA_soma_group",
        segment_groups="soma_group",
    )
    memb.channel_density_nernsts.append(chan_hva)

    # connect to other ca pool
    idstr = pynml.read_neuroml2_file(calva_fname).ion_channel[0].id
    chan_lva = nml.ChannelDensityNernstCa2(
        ion_channel=idstr,
        cond_density=p["calva_cond"],
        ion="ca2",
        id="Golgi_Ca_LVA_soma_group",
        segment_groups="soma_group",
    )
    memb.channel_density_nernst_ca2s.append(chan_lva)

    # Add GIRK  -> into apical dendrites!
    """
    if girk:
        girk_density = "0.15 mS_per_cm2"
    else:
        girk_density = "0.006 mS_per_cm2"
    read_nml = pynml.read_neuroml2_file(girk_fname)
    chan_girk = nml.ChannelDensity(ion_channel=read_nml.ion_channel[0].id,
                                    cond_density=girk_density,
                                    erev="-84.69 mV",
                                    ion="k",
                                    id="GIRK_apical_dendrite_group",
                                    segment_groups="apical_dendrite_group"
)
    memb.channel_densities.append(chan_girk)
    """

    memb.spike_threshes.append(nml.SpikeThresh("0 mV"))
    cstr = "1.0 uF_per_cm2"
    memb.specific_capacitances.append(nml.SpecificCapacitance(cstr))
    memb.init_memb_potentials.append(nml.InitMembPotential("-60 mV"))

    goc_filename = "{}.cell.nml".format(gocID)
    pynml.write_neuroml2_file(cell_doc, goc_filename)

    return True


if __name__ == "__main__":
    runid = 1
    usefile = "cellparams_file.pkl"
    if len(sys.argv) > 1:
        runid = int(sys.argv[1])
    if len(sys.argv) > 2:
        usefile = sys.argv[2]
    if len(sys.argv) > 3:
        girk = sys.argv[3]
    print("Generating Golgi cell using parameters for simid =", runid)
    res = create_GoC(runid=runid, usefile=usefile)
    print(res)
