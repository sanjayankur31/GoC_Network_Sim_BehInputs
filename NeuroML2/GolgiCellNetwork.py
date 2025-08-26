#!/usr/bin/env python3
"""
NeuroML model of cerebellar Golgi cell network

File: code/model/GolgiCellNetwork.py

Copyright 2025 Ankur Sinha
Author: Ankur Sinha <sanjay DOT ankur AT gmail DOT com>
"""

import functools
import glob
import json
import logging
import random
import sys
import typing

import neuroml
import numpy
import scipy
import typer
from neuroml.utils import component_factory
from pyneuroml.annotations import create_annotation
from pyneuroml.io import read_neuroml2_file, write_neuroml2_file
from pyneuroml.lems import generate_lems_file_for_neuroml
from pyneuroml.runners import run_lems_with
from pyneuroml.utils.units import split_nml2_quantity
from pyneuroml.validators import validate_neuroml2_lems_file


class GolgiCellNetwork(object):
    """Cerebellar Golgi Cell network model in NeuroML"""

    network_name = "Golgi_cell_network"
    nml_document = component_factory(neuroml.NeuroMLDocument, id=network_name)
    network = nml_document.add(neuroml.Network, id="Golgi_cell_network", validate=False)

    def __init__(
        self,
    ):
        """Initialise the model"""
        # set up cli
        self.app = typer.Typer(help="Cerebellar Golgi Cell network model in NeuroML")
        self.app.command()(self.create_model)
        self.app.command()(self.create_simulation)
        self.app.command()(self.simulate)
        self.app.callback()(self.configure)

    def configure(
        self,
        code_config_file: str = "parameters/general.json",
        model_parameters_file: str = "parameters/model.json",
        neuroml_file: typing.Optional[str] = None,
        seed: typing.Optional[int] = None,
        label: typing.Optional[str] = None,
        model_variant: typing.Optional[str] = None,
        lems_file: typing.Optional[str] = None,
        logging_level: typing.Optional[str] = None,
    ):
        """Configure model

        :param neuroml_file: name of NeuroML file to serialise model to
        :type neuroml_file: str
        :param seed: model/simulation seed
        :type seed: str
        :param lems_file: name of LEMS simulation file
        :type lems_file: str

        """
        self.code_config_file = code_config_file
        self.model_parameters_file = model_parameters_file
        with open(self.code_config_file) as f:
            self.general_params = json.load(f)

        # load model parameters
        with open(self.model_parameters_file) as f:
            self.model_params = json.load(f)

        if seed:
            self.seed = seed
        else:
            self.seed = self.general_params.get("seed", 1234)

        # set seeds
        random.seed(self.seed)
        numpy.random.seed(self.seed)

        if label:
            provided_label = label
        else:
            provided_label = self.general_params.get("label")
        self.label = f"{provided_label.replace(' ', '_')}" if provided_label else ""

        if model_variant:
            provided_model_variant = model_variant
        else:
            provided_model_variant = self.model_params.get("label")
        self.model_variant = (
            f"{provided_model_variant.replace(' ', '_')}"
            if provided_model_variant
            else ""
        )

        if neuroml_file:
            self.neuroml_file = neuroml.file
        else:
            self.neuroml_file = self.general_params.get(
                "neuroml_file",
                f"{self.network_name}_{self.label}_{self.model_variant}_{self.seed}.net.nml",
            )

        if lems_file:
            self.lems_file = lems_file
        else:
            self.lems_file = self.general_params.get(
                "lems_file",
                f"LEMS_test_Golgi_cells_{self.label}_{self.model_variant}_{self.seed}.xml",
            )

        if logging_level:
            self.logging_level = logging_level
        else:
            self.logging_level = self.general_params.get(
                "logging_level",
                "DEBUG",
            )

        # set up a logger
        self.logger = logging.getLogger(self.network_name)
        self.logger.setLevel(getattr(logging, self.logging_level))
        ch = logging.StreamHandler()
        ch.setLevel(getattr(logging, self.logging_level))
        formatter = logging.Formatter(
            "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        )
        ch.setFormatter(formatter)
        self.logger.addHandler(ch)
        self.logger.propagate = False

        self.logger.info(
            f"CONFIG: General parameters: {self.label}: {self.code_config_file}"
        )
        self.logger.info(
            f"CONFIG: Model parameters: {self.model_variant}: {self.model_parameters_file}"
        )

    def create_model(self):
        """Create the model"""
        self.logger.info("Creating model")

        if self.general_params.get("Annotations", True):
            annotation = create_annotation(
                subject="Golgi_cell_network",
                abstract="Cerebellar Golgi Cell network model",
                title="Cerebellar Golgi Cell network model",
                annotation_style="miriam",
                xml_header=False,
                keywords=["Golgi cell", "Cerebellum"],
                creation_date="2025-06-21",
                authors={
                    "Ankur Sinha": {
                        "ankur.sinha@ucl.ac.uk": "mbox",
                        "https://orcid.org/0000-0001-7568-7167": "orcid",
                    }
                },
                sources={
                    "https://github.com/sanjayankur31/GolgiCellNetwork/": "GitHub"
                },
                is_version_of={
                    "https://github.com/harshagurnani/GoC_Network_Sim_BehInputs": "Gurnani 2021"
                },
                references={
                    "https://doi.org/10.1016/j.neuron.2021.03.027": "Gurnani and Silver, 2021, Neuron 109, 1-15",
                    "https://doi.org/10.1016/j.neuron.2010.06.028": "Vervaeke et al, 2010, Neuron 67, 435 - 451",
                },
            )
            self.nml_document.annotation = neuroml.Annotation([annotation])
        else:
            self.logger.warning("Annotations disabled in params file")

        # do not use dict.get for the final because we want errors to be thrown
        # if these are missing
        self.network_xyz = self.model_params.get("Golgi_cells")["xyz"]
        self.golgi_cell_density = self.model_params.get("Golgi_cells")["density"]
        self.homogeneous_populations = self.model_params.get("Golgi_cells")[
            "homogeneous"
        ]
        self.num_golgi_populations = (
            1
            if self.homogeneous_populations
            else self.model_params.get("Golgi_cells")["num_populations"]
        )
        self.golgi_cell_files = glob.glob("./cells/Golgi/GoC_*.cell.nml")

        self.golgi_cell_variants = list(
            numpy.random.choice(
                self.golgi_cell_files, self.num_golgi_populations, replace=False
            )
        )
        self.golgi_cell_variants.sort()

        self.logger.debug(f"VAR: {self.golgi_cell_variants = }")

        self.__get_golgi_cell_locations()

        self.__create_network()
        if self.model_params.get("Gap_junctions").get("enabled"):
            self.__create_gap_junctions()
        else:
            self.logger.warn("Gap junctions disabled in model params")

        # write to file
        self.logger.info(f"Writing network to {self.neuroml_file}")
        write_neuroml2_file(self.nml_document, self.neuroml_file)

    def __get_golgi_cell_locations(self):
        """Get locations of Golgi cells that fit in the given volume

        Also calculate the total number of Golgi cells, and the number of cells
        per variant.
        """
        x, y, z = list(map(lambda x: split_nml2_quantity(x)[0], self.network_xyz))
        self.num_golgi_cells = int(
            split_nml2_quantity(self.golgi_cell_density)[0] * (x * y * z)
        )
        # round off to nearest multiple of num_golgi_populations so that we can
        # distribute the different Golgi variants uniformly
        self.num_golgi_cells = self.num_golgi_populations * round(
            self.num_golgi_cells / self.num_golgi_populations
        )
        self.logger.debug(f"VAR: {self.num_golgi_cells = }")

        self.golgi_cell_locations: typing.List[typing.List[float]] = (
            numpy.random.random_sample((self.num_golgi_cells, 3)) * [x, y, z]
        )

        self.num_cells_per_golgi_cell_variant = int(
            self.num_golgi_cells / self.num_golgi_populations
        )
        self.logger.debug(f"VAR: {self.num_cells_per_golgi_cell_variant = }")

    def __create_network(self):
        """Create network"""
        # track components in case we need them later
        self.golgi_cell_components = {}
        location_ctr = 0
        for pnum in range(0, self.num_golgi_populations):
            golgi_cell_variant = self.golgi_cell_variants[pnum]

            golgi_cell_component = read_neuroml2_file(golgi_cell_variant).cells[0]
            self.golgi_cell_components[golgi_cell_component.id] = golgi_cell_component

            self.nml_document.add(neuroml.IncludeType, href=golgi_cell_variant)
            pop = self.network.add(
                neuroml.Population,
                id=f"{golgi_cell_component.id}_pop",
                component=golgi_cell_component.id,
                type="populationList",
            )

            for ins in range(0, self.num_cells_per_golgi_cell_variant):
                x = self.golgi_cell_locations[location_ctr][0]
                y = self.golgi_cell_locations[location_ctr][1]
                z = self.golgi_cell_locations[location_ctr][2]

                pop.add(
                    neuroml.Instance, id=ins, location=neuroml.Location(x=x, y=y, z=z)
                )
                location_ctr += 1

    def __create_gap_junctions(self):
        """Create gap junctions between cells"""
        gap_junction_model = self.nml_document.add(
            neuroml.GapJunction,
            id="Gap_junction",
            conductance=self.model_params.get("Gap_junctions")["conductance"],
        )
        self.logger.debug(f"VAR: {self.golgi_cell_locations.shape = }")
        # returns the matrix in condensed form where it is read in row major
        # form
        distance_bw_cells = scipy.spatial.distance.pdist(
            self.golgi_cell_locations, metric="euclidean"
        )
        self.logger.debug(f"VAR: {distance_bw_cells = }")

        # Only convert to square form when required, but until then, use
        # condensed form. The matrix will remain symmetric from here.
        # distance_bw_cells_matrix = scipy.spatial.distance.squareform(distance_bw_cells)
        # self.logger.debug(f"VAR: {distance_bw_cells_matrix.shape = }")

        # Vervaeke et al 2010, Figure 7
        connection_probability = 1e-2 * (
            -1745 + 1836 / (1 + numpy.exp((distance_bw_cells - 267) / 39))
        )

        random_matrix = numpy.random.random(size=distance_bw_cells.shape)
        connection_probability -= random_matrix

        # ensure minimum probability is 0
        connection_probability = numpy.maximum(connection_probability, 0)
        self.logger.debug(f"VAR: {connection_probability = }")
        # get ones that are connected, print number of connected pairs
        connected_cells = connection_probability > 0
        self.logger.debug(f"VAR: {connected_cells = }")
        num_connected = len(numpy.nonzero(connected_cells)[0])
        self.logger.debug(f"VAR: {num_connected = }")

        gap_junction_weight_type = self.model_params.get("Gap_junctions")["weight_type"]
        self.logger.debug(f"VAR: {gap_junction_weight_type = }")

        if gap_junction_weight_type == "Vervaeke2010":
            weights_matrix = self.__get_gap_junction_weights_vervaeke2010(
                distance_bw_cells
            )
        elif gap_junction_weight_type == "Szobozlay2016":
            weights_matrix = self.__get_gap_junction_weights_szobozlay2016(
                distance_bw_cells
            )
        elif gap_junction_weight_type == "constant":
            gap_junction_weight = self.model_params.get("Gap_junctions")[
                "constant_weight"
            ]
            weights_matrix = self.__get_gap_junction_weights_constant(
                distance_bw_cells, gap_junction_weight
            )
        else:
            raise ValueError(
                f"Invalid weight type for gap junctions: {gap_junction_weight_type}"
            )

        self.logger.debug(f"VAR: {weights_matrix = }")

        # index in the condensed matrix
        k = 0

        # keep track of connected pops
        projection_ids = {}
        # iterate over the top triangle of the square matrices, which
        # corresponds to the condensed matrices we have
        for i in range(self.num_golgi_cells):
            pre_cell_pop = int(i / self.num_cells_per_golgi_cell_variant)
            pre_cell_index = i % self.num_cells_per_golgi_cell_variant
            # self.logger.debug(f"VAR: {pre_cell_pop = }")
            # self.logger.debug(f"VAR: {pre_cell_index = }")

            pre_cell_pop_component = self.network.populations[pre_cell_pop]
            pre_cell_component_id = pre_cell_pop_component.component

            for j in range(i + 1, self.num_golgi_cells):
                connected = connected_cells[k]

                if connected:
                    post_cell_pop = int(j / self.num_cells_per_golgi_cell_variant)
                    post_cell_index = j % self.num_cells_per_golgi_cell_variant

                    post_cell_pop_component = self.network.populations[post_cell_pop]
                    post_cell_component_id = post_cell_pop_component.component
                    # self.logger.debug(f"VAR: {post_cell_pop = }")
                    # self.logger.debug(f"VAR: {post_cell_index = }")

                    weight = weights_matrix[k]
                    dendritic_id_pre = random.choice(
                        self.__get_dendritic_ids(pre_cell_component_id)
                    )
                    dendritic_id_post = random.choice(
                        self.__get_dendritic_ids(post_cell_component_id)
                    )

                    # create projection
                    # track so that we don't add multiple projections
                    # between the same populations
                    projection_id = f"GJ_{pre_cell_pop}_{post_cell_pop}"
                    try:
                        projection_index = projection_ids[projection_id]
                        projection = self.network.electrical_projections[
                            projection_index
                        ]
                    except KeyError:
                        projection = self.network.add(
                            neuroml.ElectricalProjection,
                            id=f"GJ_{pre_cell_pop}_{post_cell_pop}",
                            presynaptic_population=pre_cell_pop_component.id,
                            postsynaptic_population=post_cell_pop_component.id,
                        )
                        projection_ids[projection_id] = (
                            len(self.network.electrical_projections) - 1
                        )

                    projection.add(
                        neuroml.ElectricalConnectionInstanceW,
                        id=k,
                        pre_cell=f"../{pre_cell_pop_component.id}/{pre_cell_index}/{pre_cell_component_id}",
                        pre_segment=f"{dendritic_id_pre}",
                        post_cell=f"../{post_cell_pop_component.id}/{post_cell_index}/{post_cell_component_id}",
                        post_segment=f"{dendritic_id_post}",
                        synapse=gap_junction_model.id,
                        weight=weight,
                    )

                k += 1

    @functools.cache
    def __get_dendritic_ids(self, cell_component_id):
        """Get number of dendrites in cell with given component_id

        :param cell_component_id: component id of cell
        :returns: number of dendrites in this cell component

        """
        cell: neuroml.Cell = self.golgi_cell_components[cell_component_id]
        dendrites = cell.get_all_segments_in_group("dendrite_group")
        return dendrites

    def __get_gap_junction_weights_constant(self, dist_matrix, weight=1):
        """Get constant weights for gap junctions.

        :param dist_matrix: condensed matrix of distances between cells
        :param weight: weight to use
        :returns: matrix of weights, with 0 as minimum

        """
        weights = numpy.empty(shape=dist_matrix.shape)
        weights.fill(weight)

        return weights

    def __get_gap_junction_weights_szobozlay2016(self, dist_matrix, dist_k=1):
        """Get constant weights for gap junctions.

        :param dist_matrix: condensed matrix of distances between cells
        :param dist_k: scaling factor
        :returns: matrix of weights, with 0 as minimum

        Reference: Szobozlay2016
        """
        coupling_coefficient = self.__get_gap_junction_coupling_coefficient(
            dist_matrix, dist_k
        )
        weights = 2 * coupling_coefficient / 5.0

        weights[weights < 0] = 0

        return weights

    def __get_gap_junction_weights_vervaeke2010(self, dist_matrix, dist_k=1):
        """Get weights of gap junctions as a function of distances between the
        cell somas.

        The minimum weight is 0.

        :param dist_matrix: condensed matrix of distances between cells
        :param dist_k: scaling factor
        :returns: matrix of weights, with 0 as minimum

        Reference: Vervaeke 2010
        """
        coupling_coefficient = self.__get_gap_junction_coupling_coefficient(
            dist_matrix, dist_k
        )
        weights = (
            0.576 * numpy.exp(coupling_coefficient / 12.4)
            + 0.00059 * numpy.exp(coupling_coefficient / 2.79)
            - 0.564
        )

        weights[weights < 0] = 0

        return weights

    def __get_gap_junction_coupling_coefficient(self, dist_matrix, dist_k=1):
        """Get coupling coefficients for Gap Junctions.

        :param dist_matrix: condensed matrix of distances between cells
        :param dist_k: scaling factor
        :returns: matrix of coupling coefficients

        Reference: Vervaeke 2010
        """
        coupling_coefficient = -2.3 + 29.7 * numpy.exp(
            (-1 * dist_matrix) / (70.4 * dist_k)
        )

        return coupling_coefficient

    def create_simulation(self, lems_file: typing.Optional[str] = None):
        """Create simulation

        :param lems_file: name of LEMS file to serialise simulation to
        :type lems_file: str
        """
        if lems_file:
            self.lems_file = lems_file
        else:
            # if not already set, use a default
            if not hasattr(self, "lems_file"):
                self.logger.error(
                    "No file name set for lems_file before, please pass a value"
                )
                return

        self.logger.info(f"Saving LEMS simulation file {self.lems_file}")
        quantities, sim = generate_lems_file_for_neuroml(
            sim_id="test_golgi_cells",
            neuroml_file=self.neuroml_file,
            target=self.network.id,
            duration="1500 ms",
            dt="0.01",
            lems_file_name=self.lems_file,
            target_dir=".",
            gen_plots_for_all_v=True,
            gen_saves_for_all_v=True,
            # gen_saves_for_only_populations=["GoC_00177_pop"],
            simulation_seed=self.seed,
        )

        validate_neuroml2_lems_file(self.lems_file)

    def simulate(
        self,
        lems_file: typing.Optional[str] = None,
        skip_run: bool = False,
        only_generate_scripts: bool = False,
    ):
        """Simulate the model

        :param skip_run: only parse file but do not generate scripts or run
        :type skip_run: bool
        :param only_generate_scripts: toggle whether only the runner script
            should be generated
        :type only_generate_scripts: bool

        """
        self.simulator = self.general_params.get("simulator", "jneuroml_neuron")
        if lems_file:
            self.lems_file = lems_file
        else:
            # if not already set, use a default
            if not hasattr(self, "lems_file"):
                self.logger.error(
                    "No file name set for lems_file before, please pass a value"
                )
                return

        # https://pyneuroml.readthedocs.io/en/development/pyneuroml.runners.html#pyneuroml.runners.run_lems_with
        # you can also use `pynml ..` from the command line to do this

        kwargs = {
            "engine": self.simulator,
            "lems_file_name": self.lems_file,
            "skip_run": skip_run,
        }

        # disable gui for neuron runs
        if "neuron" in self.simulator:
            kwargs["nogui"] = True

        run_lems_with(**kwargs)


if __name__ == "__main__":
    model = GolgiCellNetwork()
    model.app()
