#!/usr/bin/env python3
# encoding: utf-8

"""
This module contains unit tests of the arc.job.adapters.ts.heuristics module
"""

import copy
import os
import unittest
import shutil

from rmgpy.molecule.molecule import Molecule
from rmgpy.reaction import Reaction
from rmgpy.species import Species

import arc.rmgdb as rmgdb
from arc.common import ARC_PATH
from arc.job.adapters.ts.heuristics import (HeuristicsAdapter,
                                            combine_coordinates_with_redundant_atoms,
                                            determine_glue_params,
                                            find_distant_neighbor,
                                            generate_the_two_constrained_zmats,
                                            get_modified_params_from_zmat_2,
                                            get_new_zmat_2_map,
                                            sort_xyz,
                                            stretch_zmat_bond,
                                            )
from arc.reaction import ARCReaction
from arc.species.converter import str_to_xyz, zmat_to_xyz
from arc.species.species import ARCSpecies

# Todo: test NCO + NH2 <=> HNCO + NH


class TestHeuristicsAdapter(unittest.TestCase):
    """
    Contains unit tests for the HeuristicsAdapter class.
    """

    @classmethod
    def setUpClass(cls):
        """
        A method that is run before all unit tests in this class.
        """
        cls.maxDiff = None
        cls.rmgdb = rmgdb.rmg_database_instance_only_fams
        if cls.rmgdb is None:
            cls.rmgdb = rmgdb.make_rmg_database_object()
            rmgdb.load_families_only(cls.rmgdb)
        cls.ccooh_xyz = {'symbols': ('C', 'C', 'O', 'O', 'H', 'H', 'H', 'H', 'H', 'H'),
                         'isotopes': (12, 12, 16, 16, 1, 1, 1, 1, 1, 1),
                         'coords': ((-1.34047, -0.03188, 0.16703), (0.07658, -0.19298, -0.34334),
                                    (0.27374, 0.70670, -1.43275), (1.64704, 0.49781, -1.86879),
                                    (-2.06314, -0.24194, -0.62839), (-1.53242, -0.70687, 1.00574),
                                    (-1.51781, 0.99794, 0.49424), (0.24018, -1.21958, -0.68782),
                                    (0.79344, 0.03863, 0.45152), (1.95991, 1.39912, -1.67215))}
        cls.ccooh = ARCSpecies(label='CCOOH', smiles='CCOO', xyz=cls.ccooh_xyz)
        cls.c2h6_xyz = {'symbols': ('C', 'C', 'H', 'H', 'H', 'H', 'H', 'H'),
                        'isotopes': (12, 12, 1, 1, 1, 1, 1, 1),
                        'coords': ((0.75560, 0.02482, 0.00505), (-0.75560, -0.02482, -0.00505),
                                   (1.17380, -0.93109, -0.32410), (1.11891, 0.80903, -0.66579),
                                   (1.12656, 0.23440, 1.01276), (-1.17380, 0.93109, 0.32410),
                                   (-1.11891, -0.809039, 0.66579), (-1.12656, -0.23440, -1.01276))}
        cls.c2h6 = ARCSpecies(label='C2H6', smiles='CC', xyz=cls.c2h6_xyz)
        cls.zmat_1 = {'symbols': ('C', 'C', 'O', 'O', 'H', 'H', 'H', 'H', 'H', 'H'),
                      'coords': ((None, None, None), ('R_1_0', None, None), ('R_2_1', 'A_2_1_0', None),
                                 ('R_3_2', 'A_3_2_1', 'D_3_2_1_0'), ('R_4_0', 'A_4_0_1', 'D_4_0_1_3'),
                                 ('R_5_0', 'A_5_0_1', 'D_5_0_1_4'), ('R_6_0', 'A_6_0_1', 'D_6_0_1_5'),
                                 ('R_7_1', 'A_7_1_0', 'D_7_1_0_6'), ('R_8_1', 'A_8_1_0', 'D_8_1_0_7'),
                                 ('R_9_3', 'A_9_3_2', 'D_9_3_2_1')),
                      'vars': {'R_1_0': 1.5147479951212197, 'R_2_1': 1.4265728986680748,
                               'A_2_1_0': 108.63387152978416, 'R_3_2': 1.4559254886404387,
                               'A_3_2_1': 105.58023544826183, 'D_3_2_1_0': 179.9922243050821,
                               'R_4_0': 1.0950205915944824, 'A_4_0_1': 110.62463321031589,
                               'D_4_0_1_3': 59.13545080998071, 'R_5_0': 1.093567969297245,
                               'A_5_0_1': 110.91425998596507, 'D_5_0_1_4': 120.87266977773987,
                               'R_6_0': 1.0950091062890002, 'A_6_0_1': 110.62270362433773,
                               'D_6_0_1_5': 120.87301274044218, 'R_7_1': 1.0951433842986755,
                               'A_7_1_0': 110.20822115119915, 'D_7_1_0_6': 181.16392677464265,
                               'R_8_1': 1.0951410439636102, 'A_8_1_0': 110.20143800025897,
                               'D_8_1_0_7': 239.4199964284852, 'R_9_3': 0.9741224704818748,
                               'A_9_3_2': 96.30065819269021, 'D_9_3_2_1': 242.3527063196313},
                      'map': {0: 0, 1: 1, 2: 2, 3: 3, 4: 4, 5: 5, 6: 6, 7: 7, 8: 8, 9: 9}}
        cls.zmat_2 = {'symbols': ('H', 'C', 'C', 'H', 'H', 'H', 'H', 'H'),
                      'coords': ((None, None, None), ('R_1_0', None, None), ('R_2_1', 'A_2_1_0', None),
                                 ('R_3_1', 'A_3_1_2', 'D_3_1_2_0'), ('R_4_1', 'A_4_1_2', 'D_4_1_2_3'),
                                 ('R_5_2', 'A_5_2_1', 'D_5_2_1_4'), ('R_6_2', 'A_6_2_1', 'D_6_2_1_5'),
                                 ('R_7_2', 'A_7_2_1', 'D_7_2_1_6')),
                      'vars': {'R_1_0': 1.0940775789443724, 'R_2_1': 1.5120487296562577,
                               'A_2_1_0': 110.56801921096591, 'R_3_1': 1.0940725668318991,
                               'A_3_1_2': 110.56890700195424, 'D_3_1_2_0': 239.99938309284212,
                               'R_4_1': 1.0940817193677925, 'A_4_1_2': 110.56754686774481,
                               'D_4_1_2_3': 239.9997190582892, 'R_5_2': 1.0940725668318991,
                               'A_5_2_1': 110.56890700195424, 'D_5_2_1_4': 59.99971758419434,
                               'R_6_2': 1.0940840619688397, 'A_6_2_1': 110.56790845138725,
                               'D_6_2_1_5': 239.99905123159166, 'R_7_2': 1.0940817193677925,
                               'A_7_2_1': 110.56754686774481, 'D_7_2_1_6': 240.00122783407815},
                      'map': {0: 3, 1: 0, 2: 1, 3: 2, 4: 4, 5: 5, 6: 6, 7: 7}}
        cls.h2_xyz = {'coords': ((0, 0, 0.3736550), (0, 0, -0.3736550)),
                      'isotopes': (1, 1), 'symbols': ('H', 'H')}
        cls.h2_mol = ARCSpecies(label='H2', smiles='[H][H]', xyz=cls.h2_xyz).mol
        cls.ch3_xyz = """C       0.00000000    0.00000001   -0.00000000
                         H       1.06690511   -0.17519582    0.05416493
                         H      -0.68531716   -0.83753536   -0.02808565
                         H      -0.38158795    1.01273118   -0.02607927"""
        cls.ch4_xyz = {'symbols': ('C', 'H', 'H', 'H', 'H'), 'isotopes': (12, 1, 1, 1, 1),
                       'coords': ((-5.45906343962835e-10, 4.233517924761169e-10, 2.9505240956083194e-10),
                                  (-0.6505520089868748, -0.7742801979689132, -0.4125187934483119),
                                  (-0.34927557824779626, 0.9815958255612931, -0.3276823191685369),
                                  (-0.022337921721882443, -0.04887374527620588, 1.0908766524267022),
                                  (1.0221655095024578, -0.15844188273952128, -0.350675540104908))}
        cls.ch4_mol = ARCSpecies(label='CH4', smiles='C', xyz=cls.ch4_xyz).mol
        cls.n2h4_xyz = str_to_xyz("""N      -0.67026921   -0.02117571   -0.25636419
                          N       0.64966276    0.05515705    0.30069593
                          H      -1.27787600    0.74907557    0.03694453
                          H      -1.14684483   -0.88535632    0.02014513
                          H       0.65472168    0.28979031    1.29740292
                          H       1.21533718    0.77074524   -0.16656810""")
        cls.n2h4_mol = ARCSpecies(label='N2H4', smiles='NN', xyz=cls.n2h4_xyz).mol
        cls.nh3_xyz = str_to_xyz("""N       0.00064924   -0.00099698    0.29559292
                         H      -0.41786606    0.84210396   -0.09477452
                         H      -0.52039228   -0.78225292   -0.10002797
                         H       0.93760911   -0.05885406   -0.10079043""")
        cls.nh3_mol = ARCSpecies(label='NH3', smiles='N', xyz=cls.nh3_xyz).mol
        cls.ch3ch2oh = ARCSpecies(label='CH3CH2OH', smiles='CCO', xyz="""C      -0.97459464    0.29181710    0.10303882
                                                                         C       0.39565894   -0.35143697    0.10221676
                                                                         O       0.30253309   -1.63748710   -0.49196889
                                                                         H      -1.68942501   -0.32359616    0.65926091
                                                                         H      -0.93861751    1.28685508    0.55523033
                                                                         H      -1.35943743    0.38135479   -0.91822428
                                                                         H       0.76858330   -0.46187184    1.12485643
                                                                         H       1.10301149    0.25256708   -0.47388355
                                                                         H       1.19485981   -2.02360458   -0.47786539""")
        cls.ch3ooh = ARCSpecies(label='CH3OOH', smiles='COO', xyz="""C      -0.76039072    0.01483858   -0.00903344
                                                                     O       0.44475333    0.76952102    0.02291303
                                                                     O       0.16024511    1.92327904    0.86381800
                                                                     H      -1.56632337    0.61401630   -0.44251282
                                                                     H      -1.02943316   -0.30449156    1.00193709
                                                                     H      -0.60052507   -0.86954495   -0.63086438
                                                                     H       0.30391344    2.59629139    0.17435159""")

    # def test_heuristics_for_h_abstraction(self):
    #     """
    #     Test that ARC can generate TS guesses based on heuristics for H Abstraction reactions.
    #     """
    #     ch4_xyz = """C  0.0000000  0.0000000  0.0000000
    #                  H  0.6279670  0.6279670  0.6279670
    #                  H -0.6279670 -0.6279670  0.6279670
    #                  H -0.6279670  0.6279670 -0.6279670
    #                  H  0.6279670 -0.6279670 -0.6279670"""
    #     h_xyz = """H  0.0  0.0  0.0"""
    #     ch3_xyz = """C  0.0000000  0.0000000  0.0000000
    #                  H  0.0000000  1.0922900  0.0000000
    #                  H  0.9459510 -0.5461450  0.0000000
    #                  H -0.9459510 -0.5461450  0.0000000"""
    #     h2_xyz = """H 0.0000000  0.0000000  0.3714780
    #                 H 0.0000000  0.0000000 -0.3714780"""
    #     ch4 = ARCSpecies(label='CH4', smiles='C', xyz=ch4_xyz)
    #     h = ARCSpecies(label='H', smiles='[H]', xyz=h_xyz)
    #     ch3 = ARCSpecies(label='CH3', smiles='[CH3]', xyz=ch3_xyz)
    #     h2 = ARCSpecies(label='H2', smiles='[H][H]', xyz=h2_xyz)
    #     rxn1 = ARCReaction(reactants=['CH4', 'H'], products=['CH3', 'H2'],
    #                        r_species=[ch4, h], p_species=[ch3, h2],
    #                        rmg_reaction=Reaction(reactants=[Species().from_smiles('C'),
    #                                                         Species().from_smiles('[H]')],
    #                                              products=[Species().from_smiles('[CH3]'),
    #                                                        Species().from_smiles('[H][H]')]))
    #     rxn1.determine_family(rmg_database=self.rmgdb)
    #     self.assertEqual(rxn1.family.label, 'H_Abstraction')
    #     self.assertEqual(rxn1.atom_map[0], 0)
    #     for index in [1, 2, 3, 4]:
    #         self.assertIn(rxn1.atom_map[index], [1, 2, 3, 4, 5])
    #     self.assertIn(rxn1.atom_map[5], [4, 5])
    #     heuristics1 = HeuristicsAdapter(job_type='tsg',
    #                                     reactions=[rxn1],
    #                                     testing=True,
    #                                     project='test',
    #                                     project_directory=os.path.join(ARC_PATH, 'arc', 'testing', 'heuristics'),
    #                                     )
    #     heuristics1.execute_incore()
    #     self.assertTrue(rxn1.ts_species.is_ts)
    #     self.assertEqual(rxn1.ts_species.charge, 0)
    #     self.assertEqual(rxn1.ts_species.multiplicity, 2)
    #     self.assertEqual(len(rxn1.ts_species.ts_guesses), 1)
    #     self.assertEqual(rxn1.ts_species.ts_guesses[0].initial_xyz['symbols'], ('C', 'H', 'H', 'H', 'H', 'H'))
    #     self.assertEqual(len(rxn1.ts_species.ts_guesses[0].initial_xyz['coords']), 6)
    #     self.assertTrue(rxn1.ts_species.ts_guesses[0].success)
    #
    #     c3h8_xyz = """C	0.0000000 0.0000000 0.5949240
    #                   C 0.0000000 1.2772010 -0.2630030
    #                   C 0.0000000 -1.2772010 -0.2630030
    #                   H 0.8870000 0.0000000 1.2568980
    #                   H -0.8870000 0.0000000 1.2568980
    #                   H 0.0000000 2.1863910 0.3643870
    #                   H 0.0000000 -2.1863910 0.3643870
    #                   H 0.8933090 1.3136260 -0.9140200
    #                   H -0.8933090 1.3136260 -0.9140200
    #                   H -0.8933090 -1.3136260 -0.9140200
    #                   H 0.8933090 -1.3136260 -0.9140200"""
    #     ho2_xyz = """O 0.0553530 -0.6124600 0.0000000
    #                  O 0.0553530 0.7190720 0.0000000
    #                  H -0.8856540 -0.8528960 0.0000000"""
    #     c3h7_xyz = """C 1.3077700 -0.2977690 0.0298660
    #                   C 0.0770610 0.5654390 -0.0483740
    #                   C -1.2288150 -0.2480100 0.0351080
    #                   H -2.1137100 0.4097560 -0.0247200
    #                   H -1.2879330 -0.9774520 -0.7931500
    #                   H -1.2803210 -0.8079420 0.9859990
    #                   H 0.1031750 1.3227340 0.7594170
    #                   H 0.0813910 1.1445730 -0.9987260
    #                   H 2.2848940 0.1325040 0.2723890
    #                   H 1.2764100 -1.3421290 -0.3008110"""
    #     h2o2_xyz = """O 0.0000000 0.7275150 -0.0586880
    #                   O 0.0000000 -0.7275150 -0.0586880
    #                   H 0.7886440 0.8942950 0.4695060
    #                   H -0.7886440 -0.8942950 0.4695060"""
    #     c3h8 = ARCSpecies(label='C3H8', smiles='CCC', xyz=c3h8_xyz)
    #     ho2 = ARCSpecies(label='HO2', smiles='O[O]', xyz=ho2_xyz)
    #     c3h7 = ARCSpecies(label='C3H7', smiles='[CH2]CC', xyz=c3h7_xyz)
    #     h2o2 = ARCSpecies(label='H2O2', smiles='OO', xyz=h2o2_xyz)
    #     rxn2 = ARCReaction(reactants=['C3H8', 'HO2'], products=['C3H7', 'H2O2'],
    #                        r_species=[c3h8, ho2], p_species=[c3h7, h2o2],
    #                        rmg_reaction=Reaction(reactants=[Species().from_smiles('CCC'),
    #                                                         Species().from_smiles('O[O]')],
    #                                              products=[Species().from_smiles('[CH2]CC'),
    #                                                        Species().from_smiles('OO')]))
    #     rxn2.determine_family(rmg_database=self.rmgdb)
    #     self.assertEqual(rxn2.family.label, 'H_Abstraction')
    #     heuristics2 = HeuristicsAdapter(job_type='tsg',
    #                                     reactions=[rxn2],
    #                                     testing=True,
    #                                     project='test',
    #                                     project_directory=os.path.join(ARC_PATH, 'arc', 'testing', 'heuristics'),
    #                                     )
    #     heuristics2.execute_incore()
    #     self.assertTrue(rxn2.ts_species.is_ts)
    #     self.assertEqual(rxn2.ts_species.charge, 0)
    #     self.assertEqual(rxn2.ts_species.multiplicity, 2)
    #     self.assertEqual(len(rxn2.ts_species.ts_guesses), 18)
    #     self.assertEqual(rxn2.ts_species.ts_guesses[0].initial_xyz['symbols'],
    #                      ('C', 'C', 'C', 'H', 'H', 'H', 'H', 'H', 'H', 'H', 'H', 'O', 'O', 'H'))
    #     self.assertEqual(rxn2.ts_species.ts_guesses[1].initial_xyz['symbols'],
    #                      ('C', 'C', 'C', 'H', 'H', 'H', 'H', 'H', 'H', 'H', 'H', 'O', 'O', 'H'))
    #     self.assertEqual(len(rxn2.ts_species.ts_guesses[1].initial_xyz['coords']), 14)
    #     self.assertTrue(rxn2.ts_species.ts_guesses[0].success)
    #     self.assertTrue(rxn2.ts_species.ts_guesses[1].success)
    #     self.assertTrue(rxn2.ts_species.ts_guesses[2].success)
    #     self.assertTrue(rxn2.ts_species.ts_guesses[3].success)
    #     self.assertTrue(rxn2.ts_species.ts_guesses[4].success)
    #     self.assertTrue(rxn2.ts_species.ts_guesses[5].success)
    #     self.assertTrue(rxn2.ts_species.ts_guesses[6].success)
    #     self.assertTrue(rxn2.ts_species.ts_guesses[7].success)
    #     self.assertTrue(rxn2.ts_species.ts_guesses[8].success)
    #     self.assertTrue(rxn2.ts_species.ts_guesses[9].success)
    #     self.assertTrue(rxn2.ts_species.ts_guesses[10].success)
    #     self.assertTrue(rxn2.ts_species.ts_guesses[11].success)
    #     self.assertTrue(rxn2.ts_species.ts_guesses[12].success)
    #     self.assertTrue(rxn2.ts_species.ts_guesses[13].success)
    #     self.assertTrue(rxn2.ts_species.ts_guesses[14].success)
    #     self.assertTrue(rxn2.ts_species.ts_guesses[15].success)
    #     self.assertTrue(rxn2.ts_species.ts_guesses[16].success)
    #     self.assertTrue(rxn2.ts_species.ts_guesses[17].success)
    #
    #     cccoh_xyz = """C -1.4562640 1.2257490 0.0000000
    #                    C 0.0000000 0.7433860 0.0000000
    #                    C 0.1008890 -0.7771710 0.0000000
    #                    O 1.4826600 -1.1256940 0.0000000
    #                    H -1.5081640 2.3212940 0.0000000
    #                    H -1.9909330 0.8624630 0.8882620
    #                    H -1.9909330 0.8624630 -0.8882620
    #                    H 0.5289290 1.1236530 0.8845120
    #                    H 0.5289290 1.1236530 -0.8845120
    #                    H -0.4109400 -1.1777970 0.8923550
    #                    H -0.4109400 -1.1777970 -0.8923550
    #                    H 1.5250230 -2.0841670 0.0000000"""
    #     oh_xyz = """O 0.0000000 0.0000000 0.1078170
    #                 H 0.0000000 0.0000000 -0.8625320"""
    #     ccco_xyz = """C      -1.22579665    0.34157501   -0.08330600
    #                   C      -0.04626439   -0.57243496    0.22897599
    #                   C      -0.11084721   -1.88672335   -0.59040103
    #                   O       0.94874959   -2.60335587   -0.24842497
    #                   H      -2.17537216   -0.14662734    0.15781317
    #                   H      -1.15774972    1.26116047    0.50644174
    #                   H      -1.23871523    0.61790236   -1.14238547
    #                   H       0.88193016   -0.02561912    0.01201028
    #                   H      -0.05081615   -0.78696747    1.30674288
    #                   H      -1.10865982   -2.31155703   -0.39617740
    #                   H      -0.21011639   -1.57815495   -1.64338139"""
    #     h2o_xyz = """O      -0.00032832    0.39781490    0.00000000
    #                  H      -0.76330345   -0.19953755    0.00000000
    #                  H       0.76363177   -0.19827735    0.00000000"""
    #     cccoh = ARCSpecies(label='CCCOH', smiles='CCCO', xyz=cccoh_xyz)
    #     oh = ARCSpecies(label='OH', smiles='[OH]', xyz=oh_xyz)
    #     ccco = ARCSpecies(label='CCCO', smiles='CCC[O]', xyz=ccco_xyz)
    #     h2o = ARCSpecies(label='H2O', smiles='O', xyz=h2o_xyz)
    #     rxn3 = ARCReaction(reactants=['CCCOH', 'OH'], products=['CCCO', 'H2O'],
    #                        r_species=[cccoh, oh], p_species=[ccco, h2o],
    #                        rmg_reaction=Reaction(reactants=[Species().from_smiles('CCCO'),
    #                                                         Species().from_smiles('[OH]')],
    #                                              products=[Species().from_smiles('CCC[O]'),
    #                                                        Species().from_smiles('O')]))
    #     rxn3.determine_family(rmg_database=self.rmgdb)
    #     self.assertEqual(rxn3.family.label, 'H_Abstraction')
    #     heuristics3 = HeuristicsAdapter(job_type='tsg',
    #                                     reactions=[rxn3],
    #                                     testing=True,
    #                                     project='test',
    #                                     project_directory=os.path.join(ARC_PATH, 'arc', 'testing', 'heuristics'),
    #                                     )
    #     heuristics3.execute_incore()
    #     self.assertTrue(rxn3.ts_species.is_ts)
    #     self.assertEqual(rxn3.ts_species.charge, 0)
    #     self.assertEqual(rxn3.ts_species.multiplicity, 2)
    #     self.assertEqual(len(rxn3.ts_species.ts_guesses), 18)
    #     self.assertEqual(rxn3.ts_species.ts_guesses[0].initial_xyz['symbols'],
    #                      ('C', 'C', 'C', 'O', 'H', 'H', 'H', 'H', 'H', 'H', 'H', 'H', 'O', 'H'))
    #     self.assertEqual(len(rxn3.ts_species.ts_guesses[1].initial_xyz['coords']), 14)
    #
    #     cdcoh_xyz = """C      -0.80601307   -0.11773769    0.32792128
    #                    C       0.23096883    0.47536513   -0.26437348
    #                    O       1.44620485   -0.11266560   -0.46339257
    #                    H      -1.74308628    0.41660480    0.45016601
    #                    H      -0.75733964   -1.13345488    0.70278513
    #                    H       0.21145717    1.48838416   -0.64841675
    #                    H       1.41780836   -1.01649567   -0.10468897"""
    #     cdco_xyz = """C      -0.68324480   -0.04685539   -0.10883672
    #                   C       0.63642204    0.05717653    0.10011041
    #                   O       1.50082619   -0.82476680    0.32598015
    #                   H      -1.27691852    0.84199331   -0.29048852
    #                   H      -1.17606821   -1.00974165   -0.10030145
    #                   H       0.99232452    1.08896899    0.06242974"""
    #     cdcoh = ARCSpecies(label='C=COH', smiles='C=CO', xyz=cdcoh_xyz)
    #     cdco = ARCSpecies(label='C=CO', smiles='C=C[O]', xyz=cdco_xyz)
    #     rxn4 = ARCReaction(reactants=['C=COH', 'H'], products=['C=CO', 'H2'],
    #                        r_species=[cdcoh, h], p_species=[cdco, h2],
    #                        rmg_reaction=Reaction(reactants=[Species().from_smiles('C=CO'),
    #                                                         Species().from_smiles('[H]')],
    #                                              products=[Species().from_smiles('C=C[O]'),
    #                                                        Species().from_smiles('[H][H]')]))
    #     rxn4.determine_family(rmg_database=self.rmgdb)
    #     self.assertEqual(rxn4.family.label, 'H_Abstraction')
    #     heuristics4 = HeuristicsAdapter(job_type='tsg',
    #                                     reactions=[rxn4],
    #                                     testing=True,
    #                                     project='test',
    #                                     project_directory=os.path.join(ARC_PATH, 'arc', 'testing', 'heuristics'),
    #                                     )
    #     heuristics4.execute_incore()
    #     self.assertTrue(rxn4.ts_species.is_ts)
    #     self.assertEqual(rxn4.ts_species.charge, 0)
    #     self.assertEqual(rxn4.ts_species.multiplicity, 2)
    #     self.assertEqual(len(rxn4.ts_species.ts_guesses), 1)
    #     self.assertEqual(rxn4.ts_species.ts_guesses[0].initial_xyz['symbols'], ('C', 'C', 'O', 'H', 'H', 'H', 'H', 'H'))
    #
    # def test_keeping_atom_order_in_ts(self):
    #     """Test that the generated TS has the same atom order as in the reactants"""
    #     ccooj_xyz = {'symbols': ('C', 'C', 'O', 'O', 'H', 'H', 'H', 'H', 'H'),
    #                  'isotopes': (12, 12, 16, 16, 1, 1, 1, 1, 1),
    #                  'coords': ((-1.10653, -0.06552, 0.042602), (0.385508, 0.205048, 0.049674),
    #                             (0.759622, 1.114927, -1.032928), (0.675395, 0.525342, -2.208593),
    #                             (-1.671503, 0.860958, 0.166273), (-1.396764, -0.534277, -0.898851),
    #                             (-1.36544, -0.740942, 0.862152), (0.97386, -0.704577, -0.082293),
    #                             (0.712813, 0.732272, 0.947293))}
    #     rxn_1 = ARCReaction(r_species=[ARCSpecies(label='CCOOj', smiles='CCO[O]', xyz=ccooj_xyz),
    #                                    self.c2h6],
    #                         p_species=[self.ccooh,
    #                                    ARCSpecies(label='C2H5', smiles='[CH2]C')])
    #     rxn_1.determine_family(rmg_database=self.rmgdb)
    #     heuristics_1 = HeuristicsAdapter(job_type='tsg',
    #                                      reactions=[rxn_1],
    #                                      testing=True,
    #                                      project='test_1',
    #                                      project_directory=os.path.join(ARC_PATH, 'arc', 'testing', 'heuristics_1'),
    #                                      )
    #     heuristics_1.execute_incore()
    #     for tsg in rxn_1.ts_species.ts_guesses:
    #         self.assertEqual(tsg.initial_xyz['symbols'],
    #                          ('C', 'C', 'O', 'O', 'H', 'H', 'H', 'H', 'H', 'C', 'C', 'H', 'H', 'H', 'H', 'H', 'H'))

    def test_combine_coordinates_with_redundant_atoms(self):
        """Test the combine_coordinates_with_redundant_atoms() function."""
        ts_xyz = combine_coordinates_with_redundant_atoms(xyz_1=self.ccooh_xyz,
                                                          xyz_2=self.c2h6_xyz,
                                                          mol_1=self.ccooh.mol,
                                                          mol_2=self.c2h6.mol,
                                                          h1=9,
                                                          h2=2,
                                                          c=2,
                                                          d=1,
                                                          d3=0,
                                                          d2=20,
                                                          atom_map=list(range(18)),
                                                          reactants_reversed=True,
                                                          )
        expected_xyz = {'symbols': ('C', 'C', 'H', 'H', 'H', 'H', 'H', 'C', 'C', 'O', 'O', 'H', 'H', 'H', 'H', 'H', 'H'),
                        'isotopes': (12, 12, 1, 1, 1, 1, 1, 12, 12, 16, 16, 1, 1, 1, 1, 1, 1),
                        'coords': ((-1.4505800233745982, 0.8347365064420571, 1.541962348783862),
                                   (-2.111858939479215, 1.0479903850216523, 0.19900876088837327),
                                   (-1.6181638579176663, 1.6970235688774484, 2.194175833085717),
                                   (-1.857568557786189, -0.05364361338628676, 2.034055738503809),
                                   (-3.1907094596969467, 1.184535658519712, 0.31914087485216847),
                                   (-1.9442680336898222, 0.18569978322531044, -0.45320910225601274),
                                   (-1.7048704471449168, 1.9363705385693595, -0.293084602757991),
                                   (0.8734510413197367, -1.134943164014587, -2.183313525798534),
                                   (0.8734510413197367, -1.134943164014587, -0.6685655306773144),
                                   (0.8734510413197367, 0.21684833439420115, -0.21274760645322632),
                                   (0.8732607157563287, 0.13929005726338484, 1.2411106073687335),
                                   (1.7530761206622243, -0.6090590094519608, -2.569028020972885),
                                   (0.8734588393757856, -2.1564621284094656, -2.5736850267994464),
                                   (-0.006180841756312172, -0.6090660244940078, -2.568989461347503),
                                   (1.766092970686615, -1.644275724891227, -0.2902670234899649),
                                   (-0.019203876234738626, -1.6443388828432262, -0.2903895055533352),
                                   (-0.15595939613160326, 0.6708822000485979, 1.3978038140214841))}
        self.assertEqual(ts_xyz, expected_xyz)

    def test_get_new_zmat2_map(self):
        """Test the get_new_zmat_2_map() function."""
        zmat_1 = {'symbols': ('C', 'C', 'O', 'O', 'H', 'H', 'H', 'H', 'H', 'H', 'X'),
                  'coords': ((None, None, None), ('R_1_0', None, None), ('R_2_1', 'A_2_1_0', None),
                             ('R_3_2', 'A_3_2_1', 'D_3_2_1_0'), ('R_4_0', 'A_4_0_1', 'D_4_0_1_3'),
                             ('R_5_0', 'A_5_0_1', 'D_5_0_1_4'), ('R_6_0', 'A_6_0_1', 'D_6_0_1_5'),
                             ('R_7_1', 'A_7_1_0', 'D_7_1_0_6'), ('R_8_1', 'A_8_1_0', 'D_8_1_0_7'),
                             ('R_9_3', 'A_9_3_2', 'D_9_3_2_1'), ('RX_10_9', 'AX_10_9_3', 'DX_10_9_3_2')),
                  'vars': {'R_1_0': 1.5147479951212197, 'R_2_1': 1.4265728986680748, 'A_2_1_0': 108.63387152978416,
                           'R_3_2': 1.4559254886404387, 'A_3_2_1': 105.58023544826183, 'D_3_2_1_0': 179.9922243050821,
                           'R_4_0': 1.0950205915944824, 'A_4_0_1': 110.62463321031589, 'D_4_0_1_3': 59.13545080998071,
                           'R_5_0': 1.093567969297245, 'A_5_0_1': 110.91425998596507, 'D_5_0_1_4': 120.87266977773987,
                           'R_6_0': 1.0950091062890002, 'A_6_0_1': 110.62270362433773, 'D_6_0_1_5': 120.87301274044218,
                           'R_7_1': 1.0951433842986755, 'A_7_1_0': 110.20822115119915, 'D_7_1_0_6': 181.16392677464265,
                           'R_8_1': 1.0951410439636102, 'A_8_1_0': 110.20143800025897, 'D_8_1_0_7': 239.4199964284852,
                           'R_9_3': 1.1689469645782498, 'A_9_3_2': 96.30065819269021, 'D_9_3_2_1': 242.3527063196313,
                           'RX_10_9': 1.0, 'AX_10_9_3': 90.0, 'DX_10_9_3_2': 0},
                  'map': {0: 0, 1: 1, 2: 2, 3: 3, 4: 4, 5: 5, 6: 6, 7: 7, 8: 8, 9: 9, 10: 'X10'}}
        zmat_2 = {'symbols': ('H', 'C', 'C', 'H', 'H', 'H', 'H', 'H'),
                  'coords': ((None, None, None), ('R_1_0', None, None), ('R_2_1', 'A_2_1_0', None),
                             ('R_3_1', 'A_3_1_2', 'D_3_1_2_0'), ('R_4_1', 'A_4_1_2', 'D_4_1_2_3'),
                             ('R_5_2', 'A_5_2_1', 'D_5_2_1_4'), ('R_6_2', 'A_6_2_1', 'D_6_2_1_5'),
                             ('R_7_2', 'A_7_2_1', 'D_7_2_1_6')),
                  'vars': {'R_1_0': 1.3128870801982788, 'R_2_1': 1.5120487296562577, 'A_2_1_0': 110.56890700195424,
                           'R_3_1': 1.0940775789443724, 'A_3_1_2': 110.56801921096591, 'D_3_1_2_0': 120.00061587714492,
                           'R_4_1': 1.0940817193677925, 'A_4_1_2': 110.56754686774481, 'D_4_1_2_3': 119.99910067703652,
                           'R_5_2': 1.0940725668318991, 'A_5_2_1': 110.56890700195424, 'D_5_2_1_4': 59.99971758419434,
                           'R_6_2': 1.0940840619688397, 'A_6_2_1': 110.56790845138725, 'D_6_2_1_5': 239.99905123159166,
                           'R_7_2': 1.0940817193677925, 'A_7_2_1': 110.56754686774481, 'D_7_2_1_6': 240.00122783407815},
                  'map': {0: 2, 1: 0, 2: 1, 3: 3, 4: 4, 5: 5, 6: 6, 7: 7}}
        new_map = get_new_zmat_2_map(zmat_1=zmat_1,
                                     zmat_2=zmat_2,
                                     atom_map=list(range(18)),
                                     reactants_reversed=False,
                                     )
        expected_new_map = {0: 0, 1: 1, 2: 2, 3: 3, 4: 4, 5: 5, 6: 6, 7: 7, 8: 8, 9: 9, 10: 'X10',
                            11: 11, 12: 12, 13: 13, 14: 14, 15: 15, 16: 16, 17: 17}
        self.assertEqual(new_map, expected_new_map)

        new_map = get_new_zmat_2_map(zmat_1=zmat_1,
                                     zmat_2=zmat_2,
                                     atom_map=list(range(18)),
                                     reactants_reversed=True,
                                     )
        expected_new_map = {0: 7, 1: 8, 2: 9, 3: 10, 4: 11, 5: 12, 6: 13, 7: 14, 8: 15, 9: 16, 10: 'X17',
                            11: 0, 12: 1, 13: 2, 14: 3, 15: 4, 16: 5, 17: 6}
        self.assertEqual(new_map, expected_new_map)

    def test_generate_the_two_constrained_zmats(self):
        """Test the generate_the_two_constrained_zmats() function."""
        zmat_1, zmat_2 = generate_the_two_constrained_zmats(xyz_1=self.ccooh_xyz,
                                                            xyz_2=self.c2h6_xyz,
                                                            mol_1=self.ccooh.mol,
                                                            mol_2=self.c2h6.mol,
                                                            h1=9,
                                                            h2=3,
                                                            a=3,
                                                            b=0,
                                                            c=2,
                                                            d=1,
                                                            )
        self.assertEqual(zmat_1, self.zmat_1)
        self.assertEqual(zmat_2, self.zmat_2)

        zmat_1, zmat_2 = generate_the_two_constrained_zmats(xyz_1=self.ch4_xyz,
                                                            xyz_2=self.h2_xyz,
                                                            mol_1=self.ch4_mol,
                                                            mol_2=self.h2_mol,
                                                            h1=2,
                                                            h2=0,
                                                            a=0,
                                                            b=1,
                                                            c=None,
                                                            d=None,
                                                            )
        expected_zmat_1 = {'symbols': ('C', 'H', 'H', 'H', 'H'),
                           'coords': ((None, None, None), ('R_1_0', None, None), ('R_2_0', 'A_2_0_1', None),
                                      ('R_3_0', 'A_3_0_1', 'D_3_0_1_2'), ('R_4_0', 'A_4_0_1', 'D_4_0_1_3')),  # R_4_0
                           'vars': {'R_1_0': 1.092199370793132, 'R_2_0': 1.0921994253661749,
                                    'A_2_0_1': 109.47122156965536, 'R_3_0': 1.092199370793132,
                                    'A_3_0_1': 109.47122278898594, 'D_3_0_1_2': 120.00000135665665,
                                    'R_4_0': 1.0921994253661749, 'A_4_0_1': 109.47121850997881,
                                    'D_4_0_1_3': 120.00000068116007},
                           'map': {0: 0, 1: 1, 2: 3, 3: 4, 4: 2}}
        expected_zmat_2 = {'symbols': ('H', 'H'), 'coords': ((None, None, None), ('R_1_0', None, None)),  # R_1_0
                           'vars': {'R_1_0': 0.7473099866382779}, 'map': {0: 0, 1: 1}}
        self.assertEqual(zmat_1, expected_zmat_1)
        self.assertEqual(zmat_2, expected_zmat_2)

        zmat_1, zmat_2 = generate_the_two_constrained_zmats(xyz_1=self.n2h4_xyz,
                                                            xyz_2=self.nh3_xyz,
                                                            mol_1=self.n2h4_mol,
                                                            mol_2=self.nh3_mol,
                                                            h1=2,
                                                            h2=1,
                                                            a=0,
                                                            b=0,
                                                            c=1,
                                                            d=None,
                                                            )
        expected_zmat_1 = {'symbols': ('N', 'N', 'H', 'H', 'H', 'H'),
                           'coords': ((None, None, None), ('R_1_0', None, None), ('R_2_1', 'A_2_1_0', None),  # A_2_1_0
                                      ('R_3_1', 'A_3_1_0', 'D_3_1_0_2'), ('R_4_0', 'A_4_0_1', 'D_4_0_1_3'),
                                      ('R_5_0', 'A_5_0_1', 'D_5_0_1_4')),
                           'vars': {'R_1_0': 1.4346996064735746, 'R_2_1': 1.023964433208832,
                                    'A_2_1_0': 113.24587551512498, 'R_3_1': 1.0248853619364922,
                                    'A_3_1_0': 111.58697299955385, 'D_3_1_0_2': 240.07077704046898,
                                    'R_4_0': 1.0239645496281908, 'A_4_0_1': 113.24586240810203,
                                    'D_4_0_1_3': 284.3887260014507, 'R_5_0': 1.024885187464345,
                                    'A_5_0_1': 111.5869758085818, 'D_5_0_1_4': 240.07079516383422},
                           'map': {0: 1, 1: 0, 2: 2, 3: 3, 4: 4, 5: 5}}
        expected_zmat_2 = {'symbols': ('H', 'N', 'H', 'H'),
                           'coords': ((None, None, None), ('R_1_0', None, None), ('R_2_1', 'A_2_1_0', None),  # R_1_0
                                      ('R_3_1', 'A_3_1_0', 'D_3_1_0_2')),
                           'vars': {'R_1_0': 1.0190000355938578, 'R_2_1': 1.0189999771005855,
                                    'A_2_1_0': 105.99799962283844, 'R_3_1': 1.0190000940871264,
                                    'A_3_1_0': 105.99799852287603, 'D_3_1_0_2': 112.36218461898632},
                           'map': {0: 1, 1: 0, 2: 2, 3: 3}}
        self.assertEqual(zmat_1, expected_zmat_1)
        self.assertEqual(zmat_2, expected_zmat_2)

        zmat_1, zmat_2 = generate_the_two_constrained_zmats(xyz_2=self.n2h4_xyz,
                                                            xyz_1=self.nh3_xyz,
                                                            mol_2=self.n2h4_mol,
                                                            mol_1=self.nh3_mol,
                                                            h2=2,
                                                            h1=1,
                                                            b=0,
                                                            a=0,
                                                            d=1,
                                                            c=None,
                                                            )
        expected_zmat_1 = {'symbols': ('N', 'H', 'H', 'H'),
                           'coords': ((None, None, None), ('R_1_0', None, None), ('R_2_0', 'A_2_0_1', None),
                                      ('R_3_0', 'A_3_0_1', 'D_3_0_1_2')),
                           'vars': {'R_1_0': 1.0189999771005855, 'R_2_0': 1.0190000940871264,
                                    'A_2_0_1': 105.9980011877756, 'R_3_0': 1.0190000355938578,  # R_3_0
                                    'A_3_0_1': 105.99799962283844, 'D_3_0_1_2': 112.36217876566015},
                           'map': {0: 0, 1: 2, 2: 3, 3: 1}}
        expected_zmat_2 = {'symbols': ('H', 'N', 'N', 'H', 'H', 'H'),
                           'coords': ((None, None, None), ('R_1_0', None, None), ('R_2_1', 'A_2_1_0', None),  # A_2_1_0
                                      ('R_3_1', 'A_3_1_2', 'D_3_1_2_0'), ('R_4_2', 'A_4_2_1', 'D_4_2_1_3'),
                                      ('R_5_2', 'A_5_2_1', 'D_5_2_1_4')),
                           'vars': {'R_1_0': 1.023964433208832, 'R_2_1': 1.4346996064735746,
                                    'A_2_1_0': 113.24587551512498, 'R_3_1': 1.0248853619364922,
                                    'A_3_1_2': 111.58697299955385, 'D_3_1_2_0': 240.07077704046898,
                                    'R_4_2': 1.0239645496281908, 'A_4_2_1': 113.24586240810203,
                                    'D_4_2_1_3': 284.3887260014507, 'R_5_2': 1.024885187464345,
                                    'A_5_2_1': 111.5869758085818, 'D_5_2_1_4': 240.07079516383422},
                           'map': {0: 2, 1: 0, 2: 1, 3: 3, 4: 4, 5: 5}}
        self.assertEqual(zmat_1, expected_zmat_1)
        self.assertEqual(zmat_2, expected_zmat_2)

        zmat_1, zmat_2 = generate_the_two_constrained_zmats(xyz_1=self.ch3ch2oh.get_xyz(),
                                                            xyz_2=self.ch3ooh.get_xyz(),
                                                            mol_1=self.ch3ch2oh.mol,
                                                            mol_2=self.ch3ooh.mol,
                                                            h1=8,
                                                            a=2,
                                                            c=1,
                                                            h2=6,
                                                            b=2,
                                                            d=1,
                                                            )
        expected_zmat_1 = {'symbols': ('C', 'C', 'O', 'H', 'H', 'H', 'H', 'H', 'H'),
                           'coords': ((None, None, None), ('R_1_0', None, None), ('R_2_1', 'A_2_1_0', None),
                                      ('R_3_0', 'A_3_0_1', 'D_3_0_1_2'), ('R_4_0', 'A_4_0_1', 'D_4_0_1_3'),
                                      ('R_5_0', 'A_5_0_1', 'D_5_0_1_4'), ('R_6_1', 'A_6_1_0', 'D_6_1_0_5'),
                                      ('R_7_1', 'A_7_1_0', 'D_7_1_0_6'), ('R_8_2', 'A_8_2_1', 'D_8_2_1_0')),  # A_8_2_1
                           'vars': {'R_1_0': 1.5137276325416074, 'R_2_1': 1.4197372463410514,
                                    'A_2_1_0': 109.01303538097567, 'R_3_0': 1.0950337097344136,
                                    'A_3_0_1': 110.63258593497066, 'D_3_0_1_2': 59.12026293645304,
                                    'R_4_0': 1.0935594120185885, 'A_4_0_1': 110.92258531860486,
                                    'D_4_0_1_3': 120.87939658650379, 'R_5_0': 1.095033981893329,
                                    'A_5_0_1': 110.63254968567193, 'D_5_0_1_4': 120.8793955156551,
                                    'R_6_1': 1.0941026391623285, 'A_6_1_0': 110.5440977183771,
                                    'D_6_1_0_5': 181.34310124526917, 'R_7_1': 1.0941023667717409,
                                    'A_7_1_0': 110.54410081124645, 'D_7_1_0_6': 239.07189759901027,
                                    'R_8_2': 0.9723850776119742, 'A_8_2_1': 107.06334992092434,
                                    'D_8_2_1_0': 179.99951584936258},
                           'map': {0: 0, 1: 1, 2: 2, 3: 3, 4: 4, 5: 5, 6: 6, 7: 7, 8: 8}}
        expected_zmat_2 = {'symbols': ('H', 'O', 'O', 'C', 'H', 'H', 'H'),
                           'coords': ((None, None, None), ('R_1_0', None, None), ('R_2_1', 'A_2_1_0', None),  # A_2_1_0
                                      ('R_3_2', 'A_3_2_1', 'D_3_2_1_0'), ('R_4_3', 'A_4_3_2', 'D_4_3_2_1'),
                                      ('R_5_3', 'A_5_3_2', 'D_5_3_2_4'), ('R_6_3', 'A_6_3_2', 'D_6_3_2_5')),
                           'vars': {'R_1_0': 0.9741406737537205, 'R_2_1': 1.4557553347542735,
                                    'A_2_1_0': 96.30924284405943, 'R_3_2': 1.4223004218216138,
                                    'A_3_2_1': 105.53988198924208, 'D_3_2_1_0': 242.22744613021646,
                                    'R_4_3': 1.093821659465454, 'A_4_3_2': 110.03248558165065,
                                    'D_4_3_2_1': 60.84271231265853, 'R_5_3': 1.0938084178037755,
                                    'A_5_3_2': 110.03299489037433, 'D_5_3_2_4': 238.4134975536592,
                                    'R_6_3': 1.0928700313199922, 'A_6_3_2': 108.55511996651099,
                                    'D_6_3_2_5': 240.7911024479184},
                           'map': {0: 6, 1: 2, 2: 1, 3: 0, 4: 3, 5: 4, 6: 5}}
        self.assertEqual(zmat_1, expected_zmat_1)
        self.assertEqual(zmat_2, expected_zmat_2)

        zmat_1, zmat_2 = generate_the_two_constrained_zmats(xyz_2=self.ch3ch2oh.get_xyz(),
                                                            xyz_1=self.ch3ooh.get_xyz(),
                                                            mol_2=self.ch3ch2oh.mol,
                                                            mol_1=self.ch3ooh.mol,
                                                            h2=8,
                                                            b=2,
                                                            d=1,
                                                            h1=6,
                                                            a=2,
                                                            c=1,
                                                            )
        expected_zmat_1 = {'symbols': ('C', 'O', 'O', 'H', 'H', 'H', 'H'),
                           'coords': ((None, None, None), ('R_1_0', None, None), ('R_2_1', 'A_2_1_0', None),
                                      ('R_3_0', 'A_3_0_1', 'D_3_0_1_2'), ('R_4_0', 'A_4_0_1', 'D_4_0_1_3'),
                                      ('R_5_0', 'A_5_0_1', 'D_5_0_1_4'), ('R_6_2', 'A_6_2_1', 'D_6_2_1_0')),  # A_6_2_1
                           'vars': {'R_1_0': 1.4223004218216138, 'R_2_1': 1.4557553347542735,
                                    'A_2_1_0': 105.53988198924208, 'R_3_0': 1.093821659465454,
                                    'A_3_0_1': 110.03248558165065, 'D_3_0_1_2': 60.84271231265853,
                                    'R_4_0': 1.0938084178037755, 'A_4_0_1': 110.03299489037433,
                                    'D_4_0_1_3': 238.4134975536592, 'R_5_0': 1.0928700313199922,
                                    'A_5_0_1': 108.55511996651099, 'D_5_0_1_4': 240.7911024479184,
                                    'R_6_2': 0.9741406737537205, 'A_6_2_1': 96.30924284405943,
                                    'D_6_2_1_0': 242.22744613021646},
                           'map': {0: 0, 1: 1, 2: 2, 3: 3, 4: 4, 5: 5, 6: 6}}
        expected_zmat_2 = {'symbols': ('H', 'O', 'C', 'C', 'H', 'H', 'H', 'H', 'H'),
                           'coords': ((None, None, None), ('R_1_0', None, None), ('R_2_1', 'A_2_1_0', None),  # A_2_1_0
                                      ('R_3_2', 'A_3_2_1', 'D_3_2_1_0'), ('R_4_3', 'A_4_3_2', 'D_4_3_2_1'),
                                      ('R_5_3', 'A_5_3_2', 'D_5_3_2_4'), ('R_6_3', 'A_6_3_2', 'D_6_3_2_5'),
                                      ('R_7_2', 'A_7_2_3', 'D_7_2_3_6'), ('R_8_2', 'A_8_2_3', 'D_8_2_3_7')),
                           'vars': {'R_1_0': 0.9723850776119742, 'R_2_1': 1.4197372463410514,
                                    'A_2_1_0': 107.06334992092434, 'R_3_2': 1.5137276325416074,
                                    'A_3_2_1': 109.01303538097567, 'D_3_2_1_0': 179.99951584936258,
                                    'R_4_3': 1.0950337097344136, 'A_4_3_2': 110.63258593497066,
                                    'D_4_3_2_1': 59.12026293645304, 'R_5_3': 1.0935594120185885,
                                    'A_5_3_2': 110.92258531860486, 'D_5_3_2_4': 120.87939658650379,
                                    'R_6_3': 1.095033981893329, 'A_6_3_2': 110.63254968567193,
                                    'D_6_3_2_5': 120.8793955156551, 'R_7_2': 1.0941026391623285,
                                    'A_7_2_3': 110.5440977183771, 'D_7_2_3_6': 181.34310124526917,
                                    'R_8_2': 1.0941023667717409, 'A_8_2_3': 110.54410081124645,
                                    'D_8_2_3_7': 239.07189759901027},
                           'map': {0: 8, 1: 2, 2: 1, 3: 0, 4: 3, 5: 4, 6: 5, 7: 6, 8: 7}}
        self.assertEqual(zmat_1, expected_zmat_1)
        self.assertEqual(zmat_2, expected_zmat_2)

        zmat_1, zmat_2 = generate_the_two_constrained_zmats(xyz_1=self.ch3ch2oh.get_xyz(),
                                                            xyz_2=self.ch3ooh.get_xyz(),
                                                            mol_1=self.ch3ch2oh.mol,
                                                            mol_2=self.ch3ooh.mol,
                                                            h1=7,
                                                            a=4,
                                                            c=2,
                                                            h2=3,
                                                            b=0,
                                                            d=1,
                                                            )
        expected_zmat_1 = {'symbols': ('C', 'C', 'O', 'H', 'H', 'H', 'H', 'H', 'H'),
                           'coords': ((None, None, None), ('R_1_0', None, None), ('R_2_1', 'A_2_1_0', None),
                                      ('R_3_0', 'A_3_0_1', 'D_3_0_1_2'), ('R_4_0', 'A_4_0_1', 'D_4_0_1_3'),
                                      ('R_5_0', 'A_5_0_1', 'D_5_0_1_4'), ('R_6_1', 'A_6_1_0', 'D_6_1_0_5'),
                                      ('R_7_4', 'A_7_4_2', 'D_7_4_2_1'), ('R_8_2', 'A_8_2_1', 'D_8_2_1_0')),  # A_7_4_2
                           'vars': {'R_1_0': 1.5137276325416074, 'R_2_1': 1.4197372463410514,
                                    'A_2_1_0': 109.01303538097567, 'R_3_0': 1.0950337097344136,
                                    'A_3_0_1': 110.63258593497066, 'D_3_0_1_2': 59.12026293645304,
                                    'R_4_0': 1.0935594120185885, 'A_4_0_1': 110.92258531860486,
                                    'D_4_0_1_3': 120.87939658650379, 'R_5_0': 1.095033981893329,
                                    'A_5_0_1': 110.63254968567193, 'D_5_0_1_4': 120.8793955156551,
                                    'R_6_1': 1.0941026391623285, 'A_6_1_0': 110.5440977183771,
                                    'D_6_1_0_5': 181.34310124526917, 'R_7_4': 2.509397582146114,
                                    'A_7_4_2': 37.75756623354705, 'D_7_4_2_1': 35.45850610925192,
                                    'R_8_2': 0.9723850776119742, 'A_8_2_1': 107.06334992092434,
                                    'D_8_2_1_0': 179.99951584936258},
                           'map': {0: 0, 1: 1, 2: 2, 3: 3, 4: 4, 5: 5, 6: 6, 7: 7, 8: 8}}
        expected_zmat_2 = {'symbols': ('H', 'C', 'O', 'O', 'H', 'H', 'H'),
                           'coords': ((None, None, None), ('R_1_0', None, None), ('R_2_1', 'A_2_1_0', None),  # A_2_1_0
                                      ('R_3_2', 'A_3_2_1', 'D_3_2_1_0'), ('R_4_1', 'A_4_1_2', 'D_4_1_2_3'),
                                      ('R_5_1', 'A_5_1_2', 'D_5_1_2_4'), ('R_6_3', 'A_6_3_2', 'D_6_3_2_1')),
                           'vars': {'R_1_0': 1.093821659465454, 'R_2_1': 1.4223004218216138,
                                    'A_2_1_0': 110.03248558165065, 'R_3_2': 1.4557553347542735,
                                    'A_3_2_1': 105.53988198924208, 'D_3_2_1_0': 60.84271231265853,
                                    'R_4_1': 1.0938084178037755, 'A_4_1_2': 110.03299489037433,
                                    'D_4_1_2_3': 299.2562112207745, 'R_5_1': 1.0928700313199922,
                                    'A_5_1_2': 108.55511996651099, 'D_5_1_2_4': 240.7911024479184,
                                    'R_6_3': 0.9741406737537205, 'A_6_3_2': 96.30924284405943,
                                    'D_6_3_2_1': 242.22744613021646},
                           'map': {0: 3, 1: 0, 2: 1, 3: 2, 4: 4, 5: 5, 6: 6}}
        self.assertEqual(zmat_1, expected_zmat_1)
        self.assertEqual(zmat_2, expected_zmat_2)

    def test_stretch_zmat_bond(self):
        """Test the stretch_zmat_bond function."""
        zmat2_copy = copy.deepcopy(self.zmat_2)
        stretch_zmat_bond(zmat=zmat2_copy, indices=(1, 0), stretch=1.5)
        self.assertEqual(zmat2_copy['vars']['R_2_1'], self.zmat_2['vars']['R_2_1'] * 1.5)

    def test_determine_glue_params(self):
        """Test the determine_glue_params() function."""
        zmat_1_copy = copy.deepcopy(self.zmat_1)
        param_a2, param_d2, param_d3 = determine_glue_params(zmat=zmat_1_copy,
                                                             is_a2_linear=True,
                                                             a=3,
                                                             c=2,
                                                             d=0,
                                                             d3=30,
                                                             )
        self.assertEqual(param_a2, 'A_11_9_10')
        self.assertEqual(param_d2, 'D_11_9_10_3')
        self.assertEqual(param_d3, 'D_12_11_9_10')
        self.assertEqual(zmat_1_copy['symbols'][-1], 'X')
        self.assertEqual(zmat_1_copy['coords'][-1], ('RX_10_9', 'AX_10_9_3', 'DX_10_9_3_2'))
        self.assertEqual(zmat_1_copy['vars']['RX_10_9'], 1.0)
        self.assertEqual(zmat_1_copy['vars']['AX_10_9_3'], 90)
        self.assertEqual(zmat_1_copy['vars']['DX_10_9_3_2'], 0)
        self.assertEqual(zmat_1_copy['map'][10], 'X10')
        expected_xyz_1 = {'symbols': ('C', 'C', 'O', 'O', 'H', 'H', 'H', 'H', 'H', 'H', 'X'),
                          'isotopes': (12, 12, 16, 16, 1, 1, 1, 1, 1, 1, None),
                          'coords': ((0.01398594476849108, -0.6889055930112338, -1.784376061009113),
                                     (0.01398594476849108, -0.6889055930112338, -0.26962806588789334),
                                     (0.01398594476849108, 0.6628859053975544, 0.18618985833619472),
                                     (0.013795619205083086, 0.5853276282667381, 1.6400480721581545),
                                     (0.8936110241109786, -0.1630214384486075, -2.1700905561834642),
                                     (0.013993742824539971, -1.7104245574061123, -2.174747562010025),
                                     (-0.8656459383075578, -0.16302845349065453, -2.170051996558082),
                                     (0.9066278741353695, -1.1982381538878737, 0.10867044129945613),
                                     (-0.8786689727859842, -1.1983013118398729, 0.10854795923608584),
                                     (-0.8438878073681937, 1.028321080587749, 1.77062574436878),
                                     (-0.9409710712603131, 1.1321270125843694, 0.7807776912038682))}
        self.assertEqual(zmat_to_xyz(zmat_1_copy, keep_dummy=True), expected_xyz_1)

        zmat_1_copy = copy.deepcopy(self.zmat_1)
        param_a2, param_d2, param_d3 = determine_glue_params(zmat=zmat_1_copy.copy(),
                                                             is_a2_linear=True,
                                                             a=3,
                                                             c=2,
                                                             d=0,
                                                             d3=None,
                                                             )
        self.assertEqual(param_a2, 'A_11_9_10')
        self.assertEqual(param_d2, 'D_11_9_10_3')
        self.assertIsNone(param_d3)

        zmat_1_copy = copy.deepcopy(self.zmat_1)
        param_a2, param_d2, param_d3 = determine_glue_params(zmat=zmat_1_copy.copy(),
                                                             is_a2_linear=True,
                                                             a=3,
                                                             c=None,
                                                             d=None,
                                                             d3=None,
                                                             )
        self.assertEqual(param_a2, 'A_11_9_10')
        self.assertIsNone(param_d2)
        self.assertIsNone(param_d3)

    def test_get_modified_params_from_zmat_2(self):
        """Test the get_modified_params_from_zmat_2() function."""
        zmat_1 = {'symbols': ('C', 'C', 'H', 'H', 'H', 'H', 'H', 'H', 'X'),
                  'coords': ((None, None, None), ('R_1_0', None, None), ('R_2_0', 'A_2_0_1', None),
                             ('R_3_0', 'A_3_0_1', 'D_3_0_1_2'), ('R_4_1', 'A_4_1_0', 'D_4_1_0_3'),
                             ('R_5_1', 'A_5_1_0', 'D_5_1_0_4'), ('R_6_1', 'A_6_1_0', 'D_6_1_0_5'),
                             ('R_7_0', 'A_7_0_1', 'D_7_0_1_6'), ('RX_8_7', 'AX_8_7_0', 'DX_8_7_0_1')),
                  'vars': {'R_1_0': 1.5120487296562577, 'R_2_0': 1.0940775789443724, 'A_2_0_1': 110.56801921096591,
                           'R_3_0': 1.0940817193677925, 'A_3_0_1': 110.56754686774481, 'D_3_0_1_2': 119.99910067703652,
                           'R_4_1': 1.0940725668318991, 'A_4_1_0': 110.56890700195424, 'D_4_1_0_3': 59.99971758419434,
                           'R_5_1': 1.0940840619688397, 'A_5_1_0': 110.56790845138725, 'D_5_1_0_4': 239.99905123159166,
                           'R_6_1': 1.0940817193677925, 'A_6_1_0': 110.56754686774481, 'D_6_1_0_5': 240.00122783407815,
                           'R_7_0': 1.3128870801982788, 'A_7_0_1': 110.56890700195424, 'D_7_0_1_6': 300.00028241580566,
                           'RX_8_7': 1.0, 'AX_8_7_0': 90.0, 'DX_8_7_0_1': 0},
                  'map': {0: 0, 1: 1, 2: 3, 3: 4, 4: 5, 5: 6, 6: 7, 7: 2, 8: 'X8'}}
        zmat_2 = {'symbols': ('H', 'O', 'O', 'C', 'C', 'H', 'H', 'H', 'H', 'H'),
                  'coords': ((None, None, None), ('R_1_0', None, None), ('R_2_1', 'A_2_1_0', None),
                             ('R_3_2', 'A_3_2_1', 'D_3_2_1_0'), ('R_4_3', 'A_4_3_2', 'D_4_3_2_1'),
                             ('R_5_4', 'A_5_4_3', 'D_5_4_3_2'), ('R_6_4', 'A_6_4_3', 'D_6_4_3_5'),
                             ('R_7_4', 'A_7_4_3', 'D_7_4_3_6'), ('R_8_3', 'A_8_3_4', 'D_8_3_4_7'),
                             ('R_9_3', 'A_9_3_4', 'D_9_3_4_8')),
                  'vars': {'R_1_0': 1.1689469645782498, 'R_2_1': 1.4559254886404387, 'A_2_1_0': 96.30065819269021,
                           'R_3_2': 1.4265728986680748, 'A_3_2_1': 105.58023544826183, 'D_3_2_1_0': 242.3527063196313,
                           'R_4_3': 1.5147479951212197, 'A_4_3_2': 108.63387152978416, 'D_4_3_2_1': 179.9922243050821,
                           'R_5_4': 1.0950205915944824, 'A_5_4_3': 110.62463321031589, 'D_5_4_3_2': 59.1268942923763,
                           'R_6_4': 1.093567969297245, 'A_6_4_3': 110.91425998596507, 'D_6_4_3_5': 120.87266977773987,
                           'R_7_4': 1.0950091062890002, 'A_7_4_3': 110.62270362433773, 'D_7_4_3_6': 120.87301274044218,
                           'R_8_3': 1.0951433842986755, 'A_8_3_4': 110.20822115119915, 'D_8_3_4_7': 181.16392677464265,
                           'R_9_3': 1.0951410439636102, 'A_9_3_4': 110.20143800025897, 'D_9_3_4_8': 239.4199964284852},
                  'map': {0: 9, 1: 3, 2: 2, 3: 1, 4: 0, 5: 4, 6: 5, 7: 6, 8: 7, 9: 8}}
        new_symbols, new_coords, new_vars, new_map = get_modified_params_from_zmat_2(zmat_1=zmat_1,
                                                                                     zmat_2=zmat_2,
                                                                                     is_a2_linear=True,
                                                                                     glue_params=('A_9_7_8',
                                                                                                  'D_9_7_8_0',
                                                                                                  'D_10_9_7_0'),
                                                                                     a2=100,
                                                                                     c=1,
                                                                                     d2=0,
                                                                                     d3=0,
                                                                                     atom_map=list(range(18)),
                                                                                     )
        expected_new_symbols = ('C', 'C', 'H', 'H', 'H', 'H', 'H', 'H', 'X', 'O', 'O', 'C', 'C', 'H', 'H', 'H', 'H', 'H')
        expected_new_coords = ((None, None, None), ('R_1_0', None, None), ('R_2_0', 'A_2_0_1', None),
                               ('R_3_0', 'A_3_0_1', 'D_3_0_1_2'), ('R_4_1', 'A_4_1_0', 'D_4_1_0_3'),
                               ('R_5_1', 'A_5_1_0', 'D_5_1_0_4'), ('R_6_1', 'A_6_1_0', 'D_6_1_0_5'),
                               ('R_7_0', 'A_7_0_1', 'D_7_0_1_6'), ('RX_8_7', 'AX_8_7_0', 'DX_8_7_0_1'),
                               ('R_9_7', 'A_9_7_8', 'D_9_7_8_0'), ('R_10_9', 'A_10_9_8', 'D_10_9_7_0'),
                               ('R_11_10', 'A_11_10_9', 'D_11_10_9_8'), ('R_12_11', 'A_12_11_10', 'D_12_11_10_9'),
                               ('R_13_12', 'A_13_12_11', 'D_13_12_11_10'), ('R_14_12', 'A_14_12_11', 'D_14_12_11_13'),
                               ('R_15_12', 'A_15_12_11', 'D_15_12_11_14'), ('R_16_11', 'A_16_11_12', 'D_16_11_12_15'),
                               ('R_17_11', 'A_17_11_12', 'D_17_11_12_16'))
        expected_new_vars = {'R_1_0': 1.5120487296562577, 'R_2_0': 1.0940775789443724, 'A_2_0_1': 110.56801921096591,
                             'R_3_0': 1.0940817193677925, 'A_3_0_1': 110.56754686774481, 'D_3_0_1_2': 119.99910067703652,
                             'R_4_1': 1.0940725668318991, 'A_4_1_0': 110.56890700195424, 'D_4_1_0_3': 59.99971758419434,
                             'R_5_1': 1.0940840619688397, 'A_5_1_0': 110.56790845138725, 'D_5_1_0_4': 239.99905123159166,
                             'R_6_1': 1.0940817193677925, 'A_6_1_0': 110.56754686774481, 'D_6_1_0_5': 240.00122783407815,
                             'R_7_0': 1.3128870801982788, 'A_7_0_1': 110.56890700195424, 'D_7_0_1_6': 300.00028241580566,
                             'RX_8_7': 1.0, 'AX_8_7_0': 90.0, 'DX_8_7_0_1': 0, 'R_9_7': 1.1689469645782498, 'A_9_7_8': 190,
                             'D_9_7_8_0': 0, 'R_10_9': 1.4559254886404387, 'A_10_9_8': 96.30065819269021, 'D_10_9_7_0': 0,
                             'R_11_10': 1.4265728986680748, 'A_11_10_9': 105.58023544826183, 'D_11_10_9_8': 242.3527063196313,
                             'R_12_11': 1.5147479951212197, 'A_12_11_10': 108.63387152978416, 'D_12_11_10_9': 179.9922243050821,
                             'R_13_12': 1.0950205915944824, 'A_13_12_11': 110.62463321031589, 'D_13_12_11_10': 59.1268942923763,
                             'R_14_12': 1.093567969297245, 'A_14_12_11': 110.91425998596507, 'D_14_12_11_13': 120.87266977773987,
                             'R_15_12': 1.0950091062890002, 'A_15_12_11': 110.62270362433773, 'D_15_12_11_14': 120.87301274044218,
                             'R_16_11': 1.0951433842986755, 'A_16_11_12': 110.20822115119915, 'D_16_11_12_15': 181.16392677464265,
                             'R_17_11': 1.0951410439636102, 'A_17_11_12': 110.20143800025897, 'D_17_11_12_16': 239.4199964284852}
        expected_new_map = {0: 0, 1: 1, 2: 3, 3: 4, 4: 5, 5: 6, 6: 7, 7: 2, 8: 'X8', 9: 12, 10: 11, 11: 10, 12: 9,
                            13: 13, 14: 14, 15: 15, 16: 16, 17: 17}
        self.assertEqual(new_symbols, expected_new_symbols)
        self.assertEqual(new_coords, expected_new_coords)
        self.assertEqual(new_vars, expected_new_vars)
        self.assertEqual(new_map, expected_new_map)

        zmat_1 = {'coords': ((None, None, None), ('R_1_0', None, None), ('R_2_1', 'A_2_1_0', None),
                             ('R_3_2', 'A_3_2_1', 'D_3_2_1_0'), ('R_4_0', 'A_4_0_1', 'D_4_0_1_3'),
                             ('R_5_0', 'A_5_0_1', 'D_5_0_1_4'),
                             ('R_6_0', 'A_6_0_1', 'D_6_0_1_5'), ('R_7_1', 'A_7_1_0', 'D_7_1_0_6'),
                             ('R_8_1', 'A_8_1_0', 'D_8_1_0_7'),
                             ('R_9_3', 'A_9_3_2', 'D_9_3_2_1'), ('RX_10_9', 'AX_10_9_3', 'DX_10_9_3_2')),
                  'map': {0: 0, 1: 1, 2: 2, 3: 3, 4: 4, 5: 5, 6: 6, 7: 7, 8: 8, 9: 9, 10: 'X10'},
                  'symbols': ('C', 'C', 'O', 'O', 'H', 'H', 'H', 'H', 'H', 'H', 'X'),
                  'vars': {'AX_10_9_3': 90.0, 'A_2_1_0': 108.63387152978416, 'A_3_2_1': 105.58023544826183,
                           'A_4_0_1': 110.62463321031589, 'A_5_0_1': 110.91425998596507, 'A_6_0_1': 110.62270362433773,
                           'A_7_1_0': 110.20822115119915, 'A_8_1_0': 110.20143800025897, 'A_9_3_2': 96.30065819269021,
                           'DX_10_9_3_2': 0, 'D_3_2_1_0': 179.9922243050821, 'D_4_0_1_3': 59.13545080998071,
                           'D_5_0_1_4': 120.87266977773987, 'D_6_0_1_5': 120.87301274044218,
                           'D_7_1_0_6': 181.16392677464265,
                           'D_8_1_0_7': 239.4199964284852, 'D_9_3_2_1': 242.3527063196313, 'RX_10_9': 1.0,
                           'R_1_0': 1.5147479951212197, 'R_2_1': 1.4265728986680748, 'R_3_2': 1.4559254886404387,
                           'R_4_0': 1.0950205915944824, 'R_5_0': 1.093567969297245, 'R_6_0': 1.0950091062890002,
                           'R_7_1': 1.0951433842986755, 'R_8_1': 1.0951410439636102, 'R_9_3': 1.1689469645782498}}
        zmat_2 = {'coords': ((None, None, None), ('R_1_0', None, None), ('R_2_1', 'A_2_1_0', None),
                             ('R_3_1', 'A_3_1_2', 'D_3_1_2_0'), ('R_4_1', 'A_4_1_2', 'D_4_1_2_3'),
                             ('R_5_2', 'A_5_2_1', 'D_5_2_1_4'),
                             ('R_6_2', 'A_6_2_1', 'D_6_2_1_5'), ('R_7_2', 'A_7_2_1', 'D_7_2_1_6')),
                  'map': {0: 2, 1: 0, 2: 1, 3: 3, 4: 4, 5: 5, 6: 6, 7: 7},
                  'symbols': ('H', 'C', 'C', 'H', 'H', 'H', 'H', 'H'),
                  'vars': {'A_2_1_0': 110.56890700195424, 'A_3_1_2': 110.56801921096591, 'A_4_1_2': 110.56754686774481,
                           'A_5_2_1': 110.56890700195424, 'A_6_2_1': 110.56790845138725, 'A_7_2_1': 110.56754686774481,
                           'D_3_1_2_0': 120.00061587714492, 'D_4_1_2_3': 119.99910067703652,
                           'D_5_2_1_4': 59.99971758419434,
                           'D_6_2_1_5': 239.99905123159166, 'D_7_2_1_6': 240.00122783407815,
                           'R_1_0': 1.3128870801982788,
                           'R_2_1': 1.5120487296562577, 'R_3_1': 1.0940775789443724, 'R_4_1': 1.0940817193677925,
                           'R_5_2': 1.0940725668318991, 'R_6_2': 1.0940840619688397, 'R_7_2': 1.0940817193677925}}
        new_symbols, new_coords, new_vars, new_map = get_modified_params_from_zmat_2(zmat_1=zmat_1,
                                                                                     zmat_2=zmat_2,
                                                                                     is_a2_linear=True,
                                                                                     glue_params=('A_11_9_10',
                                                                                                  'D_11_9_10_3',
                                                                                                  'D_12_11_9_3'),
                                                                                     a2=100,
                                                                                     c=2,
                                                                                     d2=None,
                                                                                     d3=0,
                                                                                     atom_map=list(range(18)),
                                                                                     )
        expected_new_symbols = ('C', 'C', 'O', 'O', 'H', 'H', 'H', 'H', 'H', 'H', 'X', 'C', 'C', 'H', 'H', 'H', 'H', 'H')
        expected_new_coords = ((None, None, None), ('R_1_0', None, None), ('R_2_1', 'A_2_1_0', None),
                               ('R_3_2', 'A_3_2_1', 'D_3_2_1_0'), ('R_4_0', 'A_4_0_1', 'D_4_0_1_3'),
                               ('R_5_0', 'A_5_0_1', 'D_5_0_1_4'), ('R_6_0', 'A_6_0_1', 'D_6_0_1_5'),
                               ('R_7_1', 'A_7_1_0', 'D_7_1_0_6'), ('R_8_1', 'A_8_1_0', 'D_8_1_0_7'),
                               ('R_9_3', 'A_9_3_2', 'D_9_3_2_1'), ('RX_10_9', 'AX_10_9_3', 'DX_10_9_3_2'),
                               ('R_11_9', 'A_11_9_10', 'D_11_9_10_3'), ('R_12_11', 'A_12_11_10', 'D_12_11_9_3'),
                               ('R_13_11', 'A_13_11_12', 'D_13_11_12_10'), ('R_14_11', 'A_14_11_12', 'D_14_11_12_13'),
                               ('R_15_12', 'A_15_12_11', 'D_15_12_11_14'), ('R_16_12', 'A_16_12_11', 'D_16_12_11_15'),
                               ('R_17_12', 'A_17_12_11', 'D_17_12_11_16'))
        expected_new_vars = {'AX_10_9_3': 90.0, 'A_2_1_0': 108.63387152978416, 'A_3_2_1': 105.58023544826183,
                             'A_4_0_1': 110.62463321031589, 'A_5_0_1': 110.91425998596507, 'A_6_0_1': 110.62270362433773,
                             'A_7_1_0': 110.20822115119915, 'A_8_1_0': 110.20143800025897, 'A_9_3_2': 96.30065819269021,
                             'DX_10_9_3_2': 0, 'D_3_2_1_0': 179.9922243050821, 'D_4_0_1_3': 59.13545080998071,
                             'D_5_0_1_4': 120.87266977773987, 'D_6_0_1_5': 120.87301274044218, 'D_7_1_0_6': 181.16392677464265,
                             'D_8_1_0_7': 239.4199964284852, 'D_9_3_2_1': 242.3527063196313, 'RX_10_9': 1.0,
                             'R_1_0': 1.5147479951212197, 'R_2_1': 1.4265728986680748, 'R_3_2': 1.4559254886404387,
                             'R_4_0': 1.0950205915944824, 'R_5_0': 1.093567969297245, 'R_6_0': 1.0950091062890002,
                             'R_7_1': 1.0951433842986755, 'R_8_1': 1.0951410439636102, 'R_9_3': 1.1689469645782498,
                             'R_11_9': 1.3128870801982788, 'A_11_9_10': 190, 'D_11_9_10_3': None, 'R_12_11': 1.5120487296562577,
                             'A_12_11_10': 110.56890700195424, 'D_12_11_9_3': 0, 'R_13_11': 1.0940775789443724,
                             'A_13_11_12': 110.56801921096591, 'D_13_11_12_10': 120.00061587714492, 'R_14_11': 1.0940817193677925,
                             'A_14_11_12': 110.56754686774481, 'D_14_11_12_13': 119.99910067703652, 'R_15_12': 1.0940725668318991,
                             'A_15_12_11': 110.56890700195424, 'D_15_12_11_14': 59.99971758419434, 'R_16_12': 1.0940840619688397,
                             'A_16_12_11': 110.56790845138725, 'D_16_12_11_15': 239.99905123159166, 'R_17_12': 1.0940817193677925,
                             'A_17_12_11': 110.56754686774481, 'D_17_12_11_16': 240.00122783407815}
        expected_new_map = {0: 0, 1: 1, 2: 2, 3: 3, 4: 4, 5: 5, 6: 6, 7: 7, 8: 8, 9: 9, 10: 'X10',
                            11: 11, 12: 12, 13: 13, 14: 14, 15: 15, 16: 16, 17: 17}
        self.assertEqual(new_symbols, expected_new_symbols)
        self.assertEqual(new_coords, expected_new_coords)
        self.assertEqual(new_vars, expected_new_vars)
        self.assertEqual(new_map, expected_new_map)

        zmat_1 = {'symbols': ('C', 'C', 'O', 'O', 'H', 'H', 'H', 'H', 'H', 'H', 'X'),
                  'coords': ((None, None, None), ('R_1_0', None, None), ('R_2_1', 'A_2_1_0', None),
                             ('R_3_2', 'A_3_2_1', 'D_3_2_1_0'), ('R_4_0', 'A_4_0_1', 'D_4_0_1_3'),
                             ('R_5_0', 'A_5_0_1', 'D_5_0_1_4'), ('R_6_0', 'A_6_0_1', 'D_6_0_1_5'),
                             ('R_7_1', 'A_7_1_0', 'D_7_1_0_6'), ('R_8_1', 'A_8_1_0', 'D_8_1_0_7'),
                             ('R_9_3', 'A_9_3_2', 'D_9_3_2_1'), ('RX_10_9', 'AX_10_9_3', 'DX_10_9_3_2')),
                  'vars': {'R_1_0': 1.5147479951212197, 'R_2_1': 1.4265728986680748, 'A_2_1_0': 108.63387152978416,
                           'R_3_2': 1.4559254886404387, 'A_3_2_1': 105.58023544826183, 'D_3_2_1_0': 179.9922243050821,
                           'R_4_0': 1.0950205915944824, 'A_4_0_1': 110.62463321031589, 'D_4_0_1_3': 59.13545080998071,
                           'R_5_0': 1.093567969297245, 'A_5_0_1': 110.91425998596507, 'D_5_0_1_4': 120.87266977773987,
                           'R_6_0': 1.0950091062890002, 'A_6_0_1': 110.62270362433773, 'D_6_0_1_5': 120.87301274044218,
                           'R_7_1': 1.0951433842986755, 'A_7_1_0': 110.20822115119915, 'D_7_1_0_6': 181.16392677464265,
                           'R_8_1': 1.0951410439636102, 'A_8_1_0': 110.20143800025897, 'D_8_1_0_7': 239.4199964284852,
                           'R_9_3': 1.1689469645782498, 'A_9_3_2': 96.30065819269021, 'D_9_3_2_1': 242.3527063196313,
                           'RX_10_9': 1.0, 'AX_10_9_3': 90.0, 'DX_10_9_3_2': 0},
                  'map': {0: 0, 1: 1, 2: 2, 3: 3, 4: 4, 5: 5, 6: 6, 7: 7, 8: 8, 9: 9, 10: 'X10'}}
        zmat_2 = {'symbols': ('H', 'C', 'C', 'H', 'H', 'H', 'H', 'H'),
                  'coords': ((None, None, None), ('R_1_0', None, None), ('R_2_1', 'A_2_1_0', None),
                             ('R_3_1', 'A_3_1_2', 'D_3_1_2_0'), ('R_4_1', 'A_4_1_2', 'D_4_1_2_3'),
                             ('R_5_2', 'A_5_2_1', 'D_5_2_1_4'), ('R_6_2', 'A_6_2_1', 'D_6_2_1_5'),
                             ('R_7_2', 'A_7_2_1', 'D_7_2_1_6')),
                  'vars': {'R_1_0': 1.3128870801982788, 'R_2_1': 1.5120487296562577, 'A_2_1_0': 110.56890700195424,
                           'R_3_1': 1.0940775789443724, 'A_3_1_2': 110.56801921096591, 'D_3_1_2_0': 120.00061587714492,
                           'R_4_1': 1.0940817193677925, 'A_4_1_2': 110.56754686774481, 'D_4_1_2_3': 119.99910067703652,
                           'R_5_2': 1.0940725668318991, 'A_5_2_1': 110.56890700195424, 'D_5_2_1_4': 59.99971758419434,
                           'R_6_2': 1.0940840619688397, 'A_6_2_1': 110.56790845138725, 'D_6_2_1_5': 239.99905123159166,
                           'R_7_2': 1.0940817193677925, 'A_7_2_1': 110.56754686774481, 'D_7_2_1_6': 240.00122783407815},
                  'map': {0: 2, 1: 0, 2: 1, 3: 3, 4: 4, 5: 5, 6: 6, 7: 7}}
        new_symbols, new_coords, new_vars, new_map = get_modified_params_from_zmat_2(zmat_1=zmat_1,
                                                                                     zmat_2=zmat_2,
                                                                                     is_a2_linear=True,
                                                                                     glue_params=('A_11_9_10',
                                                                                                  'D_11_9_10_3',
                                                                                                  'D_12_11_9_10'),
                                                                                     a2=180,
                                                                                     c=2,
                                                                                     d2=None,
                                                                                     d3=0,
                                                                                     atom_map=list(range(18)),
                                                                                     reactants_reversed=True,
                                                                                     )
        expected_new_symbols = ('C', 'C', 'O', 'O', 'H', 'H', 'H', 'H', 'H', 'H', 'X', 'C', 'C', 'H', 'H', 'H', 'H', 'H')
        expected_new_coords = ((None, None, None), ('R_1_0', None, None), ('R_2_1', 'A_2_1_0', None),
                               ('R_3_2', 'A_3_2_1', 'D_3_2_1_0'), ('R_4_0', 'A_4_0_1', 'D_4_0_1_3'),
                               ('R_5_0', 'A_5_0_1', 'D_5_0_1_4'), ('R_6_0', 'A_6_0_1', 'D_6_0_1_5'),
                               ('R_7_1', 'A_7_1_0', 'D_7_1_0_6'), ('R_8_1', 'A_8_1_0', 'D_8_1_0_7'),
                               ('R_9_3', 'A_9_3_2', 'D_9_3_2_1'), ('RX_10_9', 'AX_10_9_3', 'DX_10_9_3_2'),
                               ('R_11_9', 'A_11_9_10', 'D_11_9_10_3'), ('R_12_11', 'A_12_11_10', 'D_12_11_9_10'),
                               ('R_13_11', 'A_13_11_12', 'D_13_11_12_10'), ('R_14_11', 'A_14_11_12', 'D_14_11_12_13'),
                               ('R_15_12', 'A_15_12_11', 'D_15_12_11_14'), ('R_16_12', 'A_16_12_11', 'D_16_12_11_15'),
                               ('R_17_12', 'A_17_12_11', 'D_17_12_11_16'))
        expected_new_vars = {'R_1_0': 1.5147479951212197, 'R_2_1': 1.4265728986680748, 'A_2_1_0': 108.63387152978416,
                             'R_3_2': 1.4559254886404387, 'A_3_2_1': 105.58023544826183, 'D_3_2_1_0': 179.9922243050821,
                             'R_4_0': 1.0950205915944824, 'A_4_0_1': 110.62463321031589, 'D_4_0_1_3': 59.13545080998071,
                             'R_5_0': 1.093567969297245, 'A_5_0_1': 110.91425998596507, 'D_5_0_1_4': 120.87266977773987,
                             'R_6_0': 1.0950091062890002, 'A_6_0_1': 110.62270362433773, 'D_6_0_1_5': 120.87301274044218,
                             'R_7_1': 1.0951433842986755, 'A_7_1_0': 110.20822115119915, 'D_7_1_0_6': 181.16392677464265,
                             'R_8_1': 1.0951410439636102, 'A_8_1_0': 110.20143800025897, 'D_8_1_0_7': 239.4199964284852,
                             'R_9_3': 1.1689469645782498, 'A_9_3_2': 96.30065819269021, 'D_9_3_2_1': 242.3527063196313,
                             'RX_10_9': 1.0, 'AX_10_9_3': 90.0, 'DX_10_9_3_2': 0, 'R_11_9': 1.3128870801982788,
                             'A_11_9_10': 270, 'D_11_9_10_3': None, 'R_12_11': 1.5120487296562577, 'A_12_11_10': 110.56890700195424,
                             'D_12_11_9_10': 0, 'R_13_11': 1.0940775789443724, 'A_13_11_12': 110.56801921096591,
                             'D_13_11_12_10': 120.00061587714492, 'R_14_11': 1.0940817193677925, 'A_14_11_12': 110.56754686774481,
                             'D_14_11_12_13': 119.99910067703652, 'R_15_12': 1.0940725668318991, 'A_15_12_11': 110.56890700195424,
                             'D_15_12_11_14': 59.99971758419434, 'R_16_12': 1.0940840619688397, 'A_16_12_11': 110.56790845138725,
                             'D_16_12_11_15': 239.99905123159166, 'R_17_12': 1.0940817193677925, 'A_17_12_11': 110.56754686774481,
                             'D_17_12_11_16': 240.00122783407815}
        expected_new_map = {0: 7, 1: 8, 2: 9, 3: 10, 4: 11, 5: 12, 6: 13, 7: 14, 8: 15, 9: 16, 10: 'X17',
                            11: 0, 12: 1, 13: 2, 14: 3, 15: 4, 16: 5, 17: 6}
        self.assertEqual(new_symbols, expected_new_symbols)
        self.assertEqual(new_coords, expected_new_coords)
        self.assertEqual(new_vars, expected_new_vars)
        self.assertEqual(new_map, expected_new_map)

    def test_sort_xyz(self):
        """Test the sort_xyz() function."""
        # Test OH + NH3
        xyz_1 = {'symbols': ('O', 'H', 'N', 'H', 'H', 'H'), 'isotopes': (8, 1, 7, 1, 1, 1),
                 'coords': ((0, 0, 0), (1, 0, 0), (2, 0, 0), (3, 0, 0), (4, 0, 0), (5, 0, 0))}
        xyz = sort_xyz(xyz=xyz_1, reactants_reversed=True, rmg_reactant_mol=Molecule(smiles='N'))
        expected_xyz_1 = {'symbols': ('N', 'H', 'H', 'H', 'O', 'H'), 'isotopes': (7, 1, 1, 1, 8, 1),
                          'coords': ((2, 0, 0), (3, 0, 0), (4, 0, 0), (5, 0, 0), (0, 0, 0), (1, 0, 0))}
        self.assertEqual(xyz, expected_xyz_1)
        xyz = sort_xyz(xyz=xyz_1, reactants_reversed=False, rmg_reactant_mol=Molecule(smiles='N'))
        self.assertEqual(xyz, xyz_1)

    def test_find_distant_neighbor(self):
        """Test the find_distant_neighbor() function."""
        xyz_1 = {'symbols': ('S', 'O', 'O', 'N', 'C', 'H', 'H', 'H', 'H', 'H'),
                 'isotopes': (32, 16, 16, 14, 12, 1, 1, 1, 1, 1),
                 'coords': ((-0.06618943, -0.12360663, -0.07631983),
                            (-0.79539707, 0.86755487, 1.02675668),
                            (-0.68919931, 0.25421823, -1.34830853),
                            (0.01546439, -1.54297548, 0.44580391),
                            (1.59721519, 0.47861334, 0.00711),
                            (1.94428095, 0.40772394, 1.03719428),
                            (2.20318015, -0.14715186, -0.64755729),
                            (1.59252246, 1.5117895, -0.33908352),
                            (-0.8785689, -2.02453514, 0.38494433),
                            (-1.34135876, 1.49608206, 0.53295071))}
        mol_1 = ARCSpecies(label='CS(=O)(O)[NH]', xyz=xyz_1).mol
        self.assertEqual(find_distant_neighbor(rmg_mol=mol_1, start=8), 0)
        self.assertEqual(find_distant_neighbor(rmg_mol=mol_1, start=9), 0)
        self.assertEqual(find_distant_neighbor(rmg_mol=mol_1, start=5), 0)
        self.assertIn(find_distant_neighbor(rmg_mol=mol_1, start=2), [1, 3, 4])
        self.assertIn(find_distant_neighbor(rmg_mol=mol_1, start=4), [1, 2, 3])
        self.assertIn(find_distant_neighbor(rmg_mol=mol_1, start=3), [1, 2, 4])
        self.assertIn(find_distant_neighbor(rmg_mol=mol_1, start=0), [5, 6, 7, 8, 9])

        xyz_2 = {'symbols': ('S', 'H', 'O', 'N', 'C', 'H', 'H', 'H', 'H'),
                 'isotopes': (32, 1, 16, 14, 12, 1, 1, 1, 1),
                 'coords': ((-0.06618943, -0.12360663, -0.07631983),
                            (-0.79539707, 0.86755487, 1.02675668),
                            (-0.68919931, 0.25421823, -1.34830853),
                            (0.01546439, -1.54297548, 0.44580391),
                            (1.59721519, 0.47861334, 0.00711),
                            (1.94428095, 0.40772394, 1.03719428),
                            (2.20318015, -0.14715186, -0.64755729),
                            (1.59252246, 1.5117895, -0.33908352),
                            (-0.8785689, -2.02453514, 0.38494433))}
        mol_2 = ARCSpecies(label='C[SH](=O)[NH]', xyz=xyz_2).mol
        self.assertEqual(find_distant_neighbor(rmg_mol=mol_2, start=8), 0)
        self.assertEqual(find_distant_neighbor(rmg_mol=mol_2, start=5), 0)
        self.assertIn(find_distant_neighbor(rmg_mol=mol_2, start=2), [3, 4])
        self.assertIn(find_distant_neighbor(rmg_mol=mol_2, start=4), [2, 3])
        self.assertIn(find_distant_neighbor(rmg_mol=mol_2, start=3), [2, 4])
        self.assertIn(find_distant_neighbor(rmg_mol=mol_2, start=0), [5, 6, 7, 8])

        xyz_3 = """C                 -2.27234259   -0.78101274   -0.00989219
                   H                 -1.94047502   -1.78971513   -0.14135849
                   H                 -1.87026865   -0.16873290   -0.78986011
                   H                 -3.34123881   -0.74952096   -0.04689398
                   C                 -1.79025824   -0.25578524    1.35514655
                   H                 -2.12212581    0.75291715    1.48661285
                   H                 -2.19233218   -0.86806508    2.13511447
                   N                 -0.32177464   -0.29904964    1.40598078
                   H                  0.05399540    0.27317450    0.67703880
                   H                 -0.00873286    0.04200717    2.29236958"""
        mol_3 = ARCSpecies(label='EA', smiles='NCC', xyz=xyz_3).mol
        self.assertEqual(find_distant_neighbor(rmg_mol=mol_3, start=9), 4)
        self.assertEqual(find_distant_neighbor(rmg_mol=mol_3, start=8), 4)
        self.assertIn(find_distant_neighbor(rmg_mol=mol_3, start=5), [0, 7])
        self.assertIn(find_distant_neighbor(rmg_mol=mol_3, start=6), [0, 7])
        self.assertEqual(find_distant_neighbor(rmg_mol=mol_3, start=1), 4)
        self.assertEqual(find_distant_neighbor(rmg_mol=mol_3, start=2), 4)
        self.assertEqual(find_distant_neighbor(rmg_mol=mol_3, start=2), 4)

    @classmethod
    def tearDownClass(cls):
        """
        A function that is run ONCE after all unit tests in this class.
        Delete all project directories created during these unit tests.
        """
        shutil.rmtree(os.path.join(ARC_PATH, 'arc', 'testing', 'heuristics'), ignore_errors=True)
        shutil.rmtree(os.path.join(ARC_PATH, 'arc', 'testing', 'heuristics_1'), ignore_errors=True)


if __name__ == '__main__':
    unittest.main(testRunner=unittest.TextTestRunner(verbosity=2))
