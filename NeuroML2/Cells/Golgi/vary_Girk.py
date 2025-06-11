import pickle as pkl
import sys
from pathlib import Path

import neuroml as nml
from pyneuroml import pynml

sys.path.append("../PythonUtils")
import initialize_cell_params_JSR as icp

# import numpy as np
# from scipy.spatial import distance
# import math
# from pyneuroml.lems import LEMSSimulation
# import lems.api as lems


def create_GoC_nml(
    runid=0,
    girk_cond=0,
    # morpho_fname='Golgi_reduced_twoCaPools.cell.nml',
    add2CaPools=True,
):
    """
    Generate GoC models with different Girk density.
    Code from Harsha

    Parameters:
    ===========
    gChannelFile: pickle file with selected channel density params
    girk_cond:     Density of GIRK channels (0 for no girk)
    morpho_fname: filename (NeuroML file) with example cell
        whose morphology will be imported
    add2CaPools:  (Boolean) whether morphology file is of class Cell2CaPools
        (if False, assumes it is Cell)
    add2CaPools:  (Boolean) whether make class Cell2CaPools or Cell

    <neuroml>
        <include>
        <cell2CaPools>
            <morphology>
            <biophysicalProperties2CaPools>
                <membraneProperties2CaPools>
                    <channelDensity>
                <intracellularProperties2CaPools>
                    <species>
                    <resistivity>
    """

    GIRK = False
    GIRK_Rossi = True

    p = {
        "leak_cond": 0.021 * 2,
        "na_cond": 48.0,  # NaT
        "nap_cond": 0.19,  # NaP
        "nar_cond": 1.7,  # NaR
        "ka_cond": 8.0,  # KA
        "sk2_cond": 38.0,  # KAHP
        "kv_cond": 32.0,  # KV
        "km_cond": 1.0,  # kslow
        "bk_cond": 3.0,  # KC JSR found 9.0 in Golgi_reduced.nml
        "hcn1f_cond": 0.05,
        "hcn1s_cond": 0.05,
        "hcn2f_cond": 0.08,
        "hcn2s_cond": 0.08,
        "cahva_cond": 0.46,
        "calva_cond": 0.25,
    }

    p2 = {}

    # open input file for morphology and intracellularProperties

    # Eugenio Piasini
    # 10 compartments
    # Golgi cell by Solinas

    if add2CaPools:
        input_file = "reduced/Golgi_reduced_2CaPools.cell.nml"
    else:
        input_file = "reduced/Golgi_reduced_1CaPool_JSR.cell.nml"

    # load peak conductances
    g_varability = False
    read_gChannelFile = False

    if read_gChannelFile:
        gChannelFile = "GoC_channel_densities.pkl"
        pfile = Path(gChannelFile)

        if not pfile.exists():
            raise RuntimeError("file does not exist: " + gChannelFile)

        print("Reading GoC ID from file:", gChannelFile)
        file = open(gChannelFile, "rb")
        params_list = pkl.load(file, encoding="bytes")  # JSR
        # print(len(params_list))
        # len(params_list) = 52
        # this file does not contain conductances
        # contains ID numbers for cell file names
        # e.g. GoC_00001.cell.nml, GoC_00025.cell.nml...
        if runid < len(params_list):
            goc_id = params_list[runid]
            goc_file_nml = "Harsha/GoC_" + format(goc_id, "05d") + ".cell.nml"
            pfile = Path(goc_file_nml)
            if not pfile.exists():
                raise RuntimeError("file does not exist: " + goc_file_nml)
            # input_file = gc_file_nml
            # print('opening parameter file: ' + gc_file_nml)
            # add2CaPools = True  # existing files use 2 Ca pools
            # runid = gc_id
            goc_gmax_nml = pynml.read_neuroml2_file(goc_file_nml)
            channel_densities = goc_gmax_nml.cells[0]
            channel_densities = channel_densities.biophysical_properties
            channel_densities = channel_densities.membrane_properties
            channel_densities = channel_densities.channel_densities
            for cd in channel_densities:
                gstr = cd.cond_density.replace(" mS_per_cm2", "")
                g = float(gstr)
                print(cd.id + ": " + str(g))
                p2.update({cd.id: g})
            channel_densities = goc_gmax_nml.cells[0]
            channel_densities = channel_densities.biophysical_properties
            channel_densities = channel_densities.membrane_properties
            channel_densities = channel_densities.channel_density_nernsts
            for cd in channel_densities:
                gstr = cd.cond_density.replace(" mS_per_cm2", "")
                g = float(gstr)
                print(cd.id + ": " + str(g))
                p2.update({cd.id: g})
            runid = goc_id
        else:
            raise RuntimeError("out of bounds: runid = " + str(runid))

        file.close()

    elif g_varability:
        p = icp.get_channel_params(runid)

    input_nml = pynml.read_neuroml2_file(input_file)

    # sim ID
    if add2CaPools:
        gocID = "GoC_2CaPools_" + format(runid, "05d")
        if GIRK:
            gocID += "_GIRK_" + format(int(1e3 * girk_cond))
    else:
        gocID = "GoC_1CaPool_" + format(runid, "05d")
        if GIRK:
            gocID += "_GIRK_" + format(int(1e3 * girk_cond))

    # create nml document
    # <neuroml>
    output_nml = nml.NeuroMLDocument(id=gocID)

    # <include> ion channels
    dd = "../"
    na_fname = dd + "Mechanisms/Golgi_Na.channel.nml"
    include = nml.IncludeType(href=na_fname)
    output_nml.includes.append(include)
    nar_fname = dd + "Mechanisms/Golgi_NaR.channel.nml"
    include = nml.IncludeType(href=nar_fname)
    output_nml.includes.append(include)
    nap_fname = dd + "Mechanisms/Golgi_NaP.channel.nml"
    include = nml.IncludeType(href=nap_fname)
    output_nml.includes.append(include)

    ka_fname = dd + "Mechanisms/Golgi_KA.channel.nml"
    include = nml.IncludeType(href=ka_fname)
    output_nml.includes.append(include)
    sk2_fname = dd + "Mechanisms/Golgi_SK2.channel.nml"
    include = nml.IncludeType(href=sk2_fname)
    output_nml.includes.append(include)
    km_fname = dd + "Mechanisms/Golgi_KM.channel.nml"
    include = nml.IncludeType(href=km_fname)
    output_nml.includes.append(include)
    kv_fname = dd + "Mechanisms/Golgi_KV.channel.nml"
    include = nml.IncludeType(href=kv_fname)
    output_nml.includes.append(include)
    bk_fname = dd + "Mechanisms/Golgi_BK.channel.nml"
    include = nml.IncludeType(href=bk_fname)
    output_nml.includes.append(include)

    cahva_fname = dd + "Mechanisms/Golgi_CaHVA.channel.nml"
    include = nml.IncludeType(href=cahva_fname)
    output_nml.includes.append(include)
    calva_fname = dd + "Mechanisms/Golgi_CaLVA.channel.nml"
    include = nml.IncludeType(href=calva_fname)
    output_nml.includes.append(include)

    hcn1f_fname = dd + "Mechanisms/Golgi_HCN1f.channel.nml"
    include = nml.IncludeType(href=hcn1f_fname)
    output_nml.includes.append(include)
    hcn1s_fname = dd + "Mechanisms/Golgi_HCN1s.channel.nml"
    include = nml.IncludeType(href=hcn1s_fname)
    output_nml.includes.append(include)
    hcn2f_fname = dd + "Mechanisms/Golgi_HCN2f.channel.nml"
    include = nml.IncludeType(href=hcn2f_fname)
    output_nml.includes.append(include)
    hcn2s_fname = dd + "Mechanisms/Golgi_HCN2s.channel.nml"
    include = nml.IncludeType(href=hcn2s_fname)
    output_nml.includes.append(include)

    leak_fname = dd + "Mechanisms/Golgi_lkg.channel.nml"
    # leak_ref = nml.IncludeType(href=leak_fname)
    include = nml.IncludeType(href=leak_fname)
    output_nml.includes.append(include)
    calc_fname = dd + "Mechanisms/Golgi_CALC.nml"
    include = nml.IncludeType(href=calc_fname)
    output_nml.includes.append(include)

    if add2CaPools:
        calc2_fname = dd + "Mechanisms/Golgi_CALC2.nml"
        include = nml.IncludeType(href=calc2_fname)
        output_nml.includes.append(include)

    if GIRK:
        if GIRK_Rossi:
            girk_fname = dd + "Mechanisms/GIRK_Rossi.channel.nml"  # JSR
        else:
            girk_fname = dd + "Mechanisms/GIRK.channel.nml"

        include = nml.IncludeType(href=girk_fname)
        output_nml.includes.append(include)

    # <cell>
    # <cell2CaPools>
    if add2CaPools:
        cell_nml = nml.Cell2CaPools(id=gocID)
        output_nml.cell2_ca_poolses.append(cell_nml)
        """
        Cell2CaPools:
        Variant of Cell with two independent Ca2+ pools.
        Cell with segments specified in a morphology element
        along with details on its biophysicalProperties.
        NOTE: this can only be correctly simulated using jLEMS
        when there is a single segment in the cell,
        and v of this cell represents the membrane potential
        in that isopotential segment.
        """
    else:
        cell_nml = nml.Cell(id=gocID)
        output_nml.cells.append(cell_nml)
        """
        Cell:
        Cell with segments specified in a morphology element
        along with details on its biophysicalProperties.
        NOTE: this can only be correctly simulated using jLEMS
        when there is a single segment in the cell,
        and v of this cell represents the membrane potential
        in that isopotential segment.
        """

    # <morphology>
    if add2CaPools:
        cell_nml.morphology = input_nml.cell2_ca_poolses[0].morphology
    else:
        cell_nml.morphology = input_nml.cells[0].morphology
    # output_nml.includes.append(nml.IncludeType(href=morpho_fname))
    # cell_nml.morphology = morpho_nml

    # <biophysicalProperties>
    # <biophysicalProperties2CaPools>
    if add2CaPools:
        biophys_nml = nml.BiophysicalProperties2CaPools(id="biophys_" + gocID)
        cell_nml.biophysical_properties2_ca_pools = biophys_nml
    else:
        biophys_nml = nml.BiophysicalProperties(id="biophys_" + gocID)
        cell_nml.biophysical_properties = biophys_nml

    # <membraneProperties>
    # <membraneProperties2CaPools>
    if add2CaPools:
        memb_nml = nml.MembraneProperties2CaPools()
        biophys_nml.membrane_properties2_ca_pools = memb_nml
    else:
        memb_nml = nml.MembraneProperties()
        biophys_nml.membrane_properties = memb_nml

    # <intracellularProperties>
    # <intracellularProperties2CaPools>
    # <species>
    # <resistivity>

    if add2CaPools:
        intracellular = input_nml.cell2_ca_poolses[0]
        intracellular = intracellular.biophysical_properties2_ca_pools
        intracellular = intracellular.intracellular_properties2_ca_pools
        # intracellular = fread.cell2_ca_poolses[0].
        # biophysical_properties2_ca_pools.intracellular_properties2_ca_pools
        biophys_nml.intracellular_properties2_ca_pools = intracellular
    else:
        intracellular = input_nml.cells[0]
        intracellular = intracellular.biophysical_properties
        intracellular = intracellular.intracellular_properties
        # intracellular = pynml.read_neuroml2_file(fname).cells[0].
        # biophysical_properties.intracellular_properties
        biophys_nml.intracellular_properties = intracellular

    # <channelDensity>
    # <channelDensityNernst>
    # pynml.read_neuroml2_file(leak_fname).ion_channel[0].id ->
    # can't read ion channel passive
    erev_na = "87.39 mV"
    erev_k = "-84.69 mV"
    erev_h = "-20 mV"
    if p2:
        gstr = "{} mS_per_cm2".format(p2["Leak"])  # JSR
    else:
        gstr = "{} mS_per_cm2".format(p["leak_cond"])  # JSR
    chan_leak = nml.ChannelDensity(
        ion_channel="LeakConductance",
        # cond_density=p['leak_cond'],
        cond_density=gstr,  # JSR
        erev="-55 mV",
        ion="non_specific",
        id="Leak",
    )
    memb_nml.channel_densities.append(chan_leak)

    idstr = pynml.read_neuroml2_file(na_fname).ion_channel[0].id
    if p2:
        gstr = "{} mS_per_cm2".format(p2["Golgi_Na_soma_group"])
    else:
        gstr = "{} mS_per_cm2".format(p["na_cond"])
    chan_na = nml.ChannelDensity(
        ion_channel=idstr,
        # cond_density=p['na_cond'],
        cond_density=gstr,
        erev=erev_na,
        ion="na",
        id="Golgi_Na_soma_group",
        segment_groups="soma_group",
    )
    memb_nml.channel_densities.append(chan_na)

    idstr = pynml.read_neuroml2_file(nap_fname).ion_channel[0].id
    if p2:
        gstr = "{} mS_per_cm2".format(p2["Golgi_NaP_soma_group"])
    else:
        gstr = "{} mS_per_cm2".format(p["nap_cond"])
    chan_nap = nml.ChannelDensity(
        ion_channel=idstr,
        # cond_density=p['nap_cond'],
        cond_density=gstr,
        erev=erev_na,
        ion="na",
        id="Golgi_NaP_soma_group",
        segment_groups="soma_group",
    )
    memb_nml.channel_densities.append(chan_nap)

    idstr = pynml.read_neuroml2_file(nar_fname).ion_channel[0].id
    if p2:
        gstr = "{} mS_per_cm2".format(p2["Golgi_NaR_soma_group"])
    else:
        gstr = "{} mS_per_cm2".format(p["nar_cond"])
    chan_nar = nml.ChannelDensity(
        ion_channel=idstr,
        # cond_density=p['nar_cond'],
        cond_density=gstr,
        erev=erev_na,
        ion="na",
        id="Golgi_NaR_soma_group",
        segment_groups="soma_group",
    )
    memb_nml.channel_densities.append(chan_nar)

    idstr = pynml.read_neuroml2_file(ka_fname).ion_channel[0].id
    if p2:
        gstr = "{} mS_per_cm2".format(p2["Golgi_KA_soma_group"])
    else:
        gstr = "{} mS_per_cm2".format(p["ka_cond"])
    chan_ka = nml.ChannelDensity(
        ion_channel=idstr,
        # cond_density=p['ka_cond'],
        cond_density=gstr,
        erev=erev_k,
        ion="k",
        id="Golgi_KA_soma_group",
        segment_groups="soma_group",
    )
    memb_nml.channel_densities.append(chan_ka)

    idstr = pynml.read_neuroml2_file(sk2_fname).ion_channel_kses[0].id
    if p2:
        gstr = "{} mS_per_cm2".format(p2["Golgi_KAHP_soma_group"])
    else:
        gstr = "{} mS_per_cm2".format(p["sk2_cond"])
    chan_sk = nml.ChannelDensity(
        ion_channel=idstr,
        # cond_density=p['sk2_cond'],
        cond_density=gstr,
        erev=erev_k,
        ion="k",
        id="Golgi_KAHP_soma_group",
        segment_groups="soma_group",
    )
    memb_nml.channel_densities.append(chan_sk)

    idstr = pynml.read_neuroml2_file(kv_fname).ion_channel[0].id
    if p2:
        gstr = "{} mS_per_cm2".format(p2["Golgi_KV_soma_group"])
    else:
        gstr = "{} mS_per_cm2".format(p["kv_cond"])
    chan_kv = nml.ChannelDensity(
        ion_channel=idstr,
        # cond_density=p['kv_cond'],
        cond_density=gstr,
        erev=erev_k,
        ion="k",
        id="Golgi_KV_soma_group",
        segment_groups="soma_group",
    )
    memb_nml.channel_densities.append(chan_kv)

    idstr = pynml.read_neuroml2_file(km_fname).ion_channel[0].id
    if p2:
        gstr = "{} mS_per_cm2".format(p2["Golgi_KM_soma_group"])
    else:
        gstr = "{} mS_per_cm2".format(p["km_cond"])
    chan_km = nml.ChannelDensity(
        ion_channel=idstr,
        # cond_density=p['km_cond'],
        cond_density=gstr,
        erev=erev_k,
        ion="k",
        id="Golgi_KM_soma_group",
        segment_groups="soma_group",
    )
    memb_nml.channel_densities.append(chan_km)

    idstr = pynml.read_neuroml2_file(bk_fname).ion_channel[0].id
    if p2:
        gstr = "{} mS_per_cm2".format(p2["Golgi_BK_soma_group"])
    else:
        gstr = "{} mS_per_cm2".format(p["bk_cond"])
    chan_bk = nml.ChannelDensity(
        ion_channel=idstr,
        # cond_density=p['bk_cond'],
        cond_density=gstr,
        erev=erev_k,
        ion="k",
        id="Golgi_BK_soma_group",
        segment_groups="soma_group",
    )
    memb_nml.channel_densities.append(chan_bk)

    idstr = pynml.read_neuroml2_file(hcn1f_fname).ion_channel[0].id
    if p2:
        gstr = "{} mS_per_cm2".format(p2["Golgi_hcn1f_soma_group"])
    else:
        gstr = "{} mS_per_cm2".format(p["hcn1f_cond"])
    chan_h1f = nml.ChannelDensity(
        ion_channel=idstr,
        # cond_density=p['hcn1f_cond'],
        cond_density=gstr,
        erev=erev_h,
        ion="h",
        id="Golgi_hcn1f_soma_group",
        segment_groups="soma_group",
    )
    memb_nml.channel_densities.append(chan_h1f)

    idstr = pynml.read_neuroml2_file(hcn1s_fname).ion_channel[0].id
    if p2:
        gstr = "{} mS_per_cm2".format(p2["Golgi_hcn1s_soma_group"])
    else:
        gstr = "{} mS_per_cm2".format(p["hcn1s_cond"])
    chan_h1s = nml.ChannelDensity(
        ion_channel=idstr,
        # cond_density=p['hcn1s_cond'],
        cond_density=gstr,
        erev=erev_h,
        ion="h",
        id="Golgi_hcn1s_soma_group",
        segment_groups="soma_group",
    )
    memb_nml.channel_densities.append(chan_h1s)

    idstr = pynml.read_neuroml2_file(hcn2f_fname).ion_channel[0].id
    if p2:
        gstr = "{} mS_per_cm2".format(p2["Golgi_hcn2f_soma_group"])
    else:
        gstr = "{} mS_per_cm2".format(p["hcn2f_cond"])
    chan_h2f = nml.ChannelDensity(
        ion_channel=idstr,
        # cond_density=p['hcn2f_cond'],
        cond_density=gstr,
        erev=erev_h,
        ion="h",
        id="Golgi_hcn2f_soma_group",
        segment_groups="soma_group",
    )
    memb_nml.channel_densities.append(chan_h2f)

    idstr = pynml.read_neuroml2_file(hcn2s_fname).ion_channel[0].id
    if p2:
        gstr = "{} mS_per_cm2".format(p2["Golgi_hcn2s_soma_group"])
    else:
        gstr = "{} mS_per_cm2".format(p["hcn2s_cond"])
    chan_h2s = nml.ChannelDensity(
        ion_channel=idstr,
        # cond_density=p['hcn2s_cond'],
        cond_density=gstr,
        erev=erev_h,
        ion="h",
        id="Golgi_hcn2s_soma_group",
        segment_groups="soma_group",
    )
    memb_nml.channel_densities.append(chan_h2s)

    idstr = pynml.read_neuroml2_file(cahva_fname).ion_channel[0].id
    if p2:
        gstr = "{} mS_per_cm2".format(p2["Golgi_Ca_HVA_soma_group"])
    else:
        gstr = "{} mS_per_cm2".format(p["cahva_cond"])
    chan_hva = nml.ChannelDensityNernst(
        ion_channel=idstr,
        # cond_density=p['cahva_cond'],
        cond_density=gstr,
        ion="ca",
        id="Golgi_Ca_HVA_soma_group",
        segment_groups="soma_group",
    )
    memb_nml.channel_density_nernsts.append(chan_hva)

    if add2CaPools:
        ca_ion = "ca2"
    else:
        ca_ion = "ca"

    idstr = pynml.read_neuroml2_file(calva_fname).ion_channel[0].id
    if p2:
        gstr = "{} mS_per_cm2".format(p2["Golgi_Ca_LVA_soma_group"])
    else:
        gstr = "{} mS_per_cm2".format(p["calva_cond"])
    chan_lva = nml.ChannelDensityNernst(
        ion_channel=idstr,
        # cond_density=p['calva_cond'],
        cond_density=gstr,
        ion=ca_ion,
        id="Golgi_Ca_LVA_soma_group",
        segment_groups="soma_group",
    )
    memb_nml.channel_density_nernsts.append(chan_lva)

    if GIRK:
        # ADDED TO DENDRITES
        idstr = pynml.read_neuroml2_file(girk_fname).ion_channel[0].id
        gstr = "{} mS_per_cm2".format(girk_cond)
        chan_girk = nml.ChannelDensity(
            ion_channel=idstr,
            cond_density=gstr,
            erev=erev_k,
            ion="k",
            id="GIRK_dendrite_group",
            segment_groups="dendrite_group",
        )
        memb_nml.channel_densities.append(chan_girk)

    # <spikeThresh>
    memb_nml.spike_threshes.append(nml.SpikeThresh("0 mV"))

    # <specificCapacitance>
    cstr = "1.0 uF_per_cm2"
    memb_nml.specific_capacitances.append(nml.SpecificCapacitance(cstr))

    # <initMembPotential>
    memb_nml.init_memb_potentials.append(nml.InitMembPotential("-60 mV"))

    goc_filename = "My_{}.cell.nml".format(gocID)
    pynml.write_neuroml2_file(output_nml, goc_filename)
    print("created nml file " + goc_filename)

    return True


if __name__ == "__main__":
    runid = 0
    # gChannelFile = 'cellparams_file.pkl'  # 'GoC_channel_densities.pkl'
    add2CaPools = False
    # girk_cond = 0.006
    # girk_cond = 0.08
    # girk_cond = 0.15
    girk_cond = 0.30
    if len(sys.argv) > 1:
        runid = int(sys.argv[1])
    if len(sys.argv) > 2:
        gChannelFile = sys.argv[2]
    if len(sys.argv) > 3:
        girk_cond = float(sys.argv[3])
    print("Generating Golgi cell using parameters for simid=", runid)
    res = create_GoC_nml(runid=runid, add2CaPools=add2CaPools, girk_cond=girk_cond)
    print(res)
