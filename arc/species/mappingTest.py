#!/usr/bin/env python3
# encoding: utf-8

"""
This module contains unit tests of the arc.species.mapping module
"""

import unittest
from typing import List

from qcelemental.models.molecule import Molecule as QCMolecule

from rmgpy.molecule import Molecule
from rmgpy.reaction import Reaction
from rmgpy.species import Species

import arc.species.mapping as mapping
from arc.common import check_r_n_p_symbols_between_rmg_and_arc_rxns
from arc.rmgdb import determine_family, load_families_only, make_rmg_database_object, rmg_database_instance_only_fams
from arc.reaction import ARCReaction
from arc.species.species import ARCSpecies
from arc.species.vectors import calculate_dihedral_angle


class TestMapping(unittest.TestCase):
    """
    Contains unit tests for the mapping module.
    """

    @classmethod
    def setUpClass(cls):
        """
        A method that is run before all unit tests in this class.
        """
        cls.maxDiff = None
        cls.rmgdb = rmg_database_instance_only_fams
        if cls.rmgdb is None:
            cls.rmgdb = make_rmg_database_object()
            load_families_only(cls.rmgdb)
        cls.ch4_xyz = {'symbols': ('C', 'H', 'H', 'H', 'H'), 'isotopes': (12, 1, 1, 1, 1),
                       'coords': ((-5.45906343962835e-10, 4.233517924761169e-10, 2.9505240956083194e-10),
                                  (-0.6505520089868748, -0.7742801979689132, -0.4125187934483119),
                                  (-0.34927557824779626, 0.9815958255612931, -0.3276823191685369),
                                  (-0.022337921721882443, -0.04887374527620588, 1.0908766524267022),
                                  (1.0221655095024578, -0.15844188273952128, -0.350675540104908))}
        cls.ch4_xyz_diff_order = """H      -0.65055201   -0.77428020   -0.41251879
                                    H      -0.34927558    0.98159583   -0.32768232
                                    C      -0.00000000    0.00000000    0.00000000
                                    H      -0.02233792   -0.04887375    1.09087665
                                    H       1.02216551   -0.15844188   -0.35067554"""
        cls.oh_xyz = """O       0.48890387    0.00000000    0.00000000
                        H      -0.48890387    0.00000000    0.00000000"""
        cls.ch3_xyz = """C       0.00000000    0.00000001   -0.00000000
                         H       1.06690511   -0.17519582    0.05416493
                         H      -0.68531716   -0.83753536   -0.02808565
                         H      -0.38158795    1.01273118   -0.02607927"""
        cls.h2o_xyz = """O      -0.00032832    0.39781490    0.00000000
                         H      -0.76330345   -0.19953755    0.00000000
                         H       0.76363177   -0.19827735    0.00000000"""
        cls.nh2_xyz = """N       0.00022972    0.40059496    0.00000000
                     H      -0.83174214   -0.19982058    0.00000000
                     H       0.83151242   -0.20077438    0.00000000"""
        cls.n2h4_xyz = """N      -0.67026921   -0.02117571   -0.25636419
                          N       0.64966276    0.05515705    0.30069593
                          H      -1.27787600    0.74907557    0.03694453
                          H      -1.14684483   -0.88535632    0.02014513
                          H       0.65472168    0.28979031    1.29740292
                          H       1.21533718    0.77074524   -0.16656810"""
        cls.nh3_xyz = """N       0.00064924   -0.00099698    0.29559292
                         H      -0.41786606    0.84210396   -0.09477452
                         H      -0.52039228   -0.78225292   -0.10002797
                         H       0.93760911   -0.05885406   -0.10079043"""
        cls.n2h3_xyz = """N      -0.46371338    0.04553420    0.30600516
                          N       0.79024530   -0.44272936   -0.27090857
                          H      -1.18655934   -0.63438343    0.06795859
                          H      -0.71586186    0.90189070   -0.18800765
                          H       1.56071894    0.18069099    0.00439608"""
        cls.arc_reaction_1 = ARCReaction(label='CH4 + OH <=> CH3 + H2O',
                                         r_species=[ARCSpecies(label='CH4', smiles='C', xyz=cls.ch4_xyz),
                                                    ARCSpecies(label='OH', smiles='[OH]', xyz=cls.oh_xyz)],
                                         p_species=[ARCSpecies(label='CH3', smiles='[CH3]', xyz=cls.ch3_xyz),
                                                    ARCSpecies(label='H2O', smiles='O', xyz=cls.h2o_xyz)])
        cls.arc_reaction_2 = ARCReaction(label='C3H8 + NH2 <=> nC3H7 + NH3',
                                         r_species=[ARCSpecies(label='C3H8', smiles='CCC',
                                                               xyz="""C      -1.26511392    0.18518050   -0.19976825
                                                                      C       0.02461113   -0.61201635   -0.29700643
                                                                      C       0.09902018   -1.69054887    0.77051392
                                                                      H      -1.34710559    0.68170095    0.77242199
                                                                      H      -2.12941774   -0.47587010   -0.31761654
                                                                      H      -1.31335400    0.95021638   -0.98130653
                                                                      H       0.88022594    0.06430231   -0.19248282
                                                                      H       0.09389171   -1.07422931   -1.28794952
                                                                      H      -0.73049348   -2.39807515    0.67191015
                                                                      H       1.03755706   -2.24948851    0.69879172
                                                                      H       0.04615234   -1.24181601    1.76737952"""),
                                                    ARCSpecies(label='NH2', smiles='[NH2]', xyz=cls.nh2_xyz)],
                                         p_species=[ARCSpecies(label='nC3H7', smiles='[CH2]CC',
                                                               xyz="""C       1.37804355    0.27791700   -0.19511840
                                                                      C       0.17557158   -0.34036318    0.43265003
                                                                      C      -0.83187173    0.70418067    0.88324591
                                                                      H       2.32472110   -0.25029805   -0.17789388
                                                                      H       1.28332450    1.14667614   -0.83695597
                                                                      H      -0.29365298   -1.02042821   -0.28596734
                                                                      H       0.48922284   -0.93756983    1.29560539
                                                                      H      -1.19281782    1.29832390    0.03681748
                                                                      H      -1.69636720    0.21982441    1.34850246
                                                                      H      -0.39178710    1.38838724    1.61666119"""),
                                                    ARCSpecies(label='NH3', smiles='N', xyz=cls.nh3_xyz)])
        cls.arc_reaction_4 = ARCReaction(label='CH2CH2NH2 <=> CH3CH2NH',
                                         r_species=[ARCSpecies(label='CH2CH2NH2', smiles='[CH2]CN',
                                                               xyz="""C      -1.24450121    0.17451352    0.00786829
                                                                      C       0.09860657   -0.41192142   -0.18691029
                                                                      N       0.39631461   -0.45573259   -1.60474376
                                                                      H      -2.04227601   -0.03772349   -0.69530380
                                                                      H      -1.50683666    0.61023628    0.96427405
                                                                      H       0.11920004   -1.42399272    0.22817674
                                                                      H       0.85047586    0.18708096    0.33609395
                                                                      H       0.46736985    0.49732569   -1.95821046
                                                                      H       1.31599714   -0.87204344   -1.73848831""")],
                                         p_species=[ARCSpecies(label='CH3CH2NH', smiles='CC[NH]',
                                                               xyz="""C      -1.03259818   -0.08774861    0.01991495
                                                                      C       0.48269985   -0.19939835    0.09039740
                                                                      N       0.94816502   -1.32096642   -0.71111614
                                                                      H      -1.51589318   -0.99559128    0.39773163
                                                                      H      -1.37921961    0.75694189    0.62396315
                                                                      H      -1.37189872    0.07009088   -1.00987126
                                                                      H       0.78091492   -0.31605120    1.13875203
                                                                      H       0.92382278    0.74158978   -0.25822764
                                                                      H       1.97108857   -1.36649904   -0.64094836""")])
        cls.rmg_reaction_1 = Reaction(reactants=[Species(smiles='C'), Species(smiles='[OH]')],
                                      products=[Species(smiles='[CH3]'), Species(smiles='O')])
        cls.rmg_reaction_2 = Reaction(reactants=[Species(smiles='[OH]'), Species(smiles='C')],
                                      products=[Species(smiles='[CH3]'), Species(smiles='O')])
        cls.arc_reaction_3 = ARCReaction(label='CH3 + CH3 <=> C2H6',
                                         r_species=[ARCSpecies(label='CH3', smiles='[CH3]')],
                                         p_species=[ARCSpecies(label='C2H6', smiles='CC')])
        cls.rmg_reaction_3 = Reaction(reactants=[Species(smiles='[CH3]'), Species(smiles='[CH3]')],
                                      products=[Species(smiles='CC')])

        cls.r_xyz_2a = """C                  0.50180491   -0.93942231   -0.57086745
        C                  0.01278145    0.13148427    0.42191407
        C                 -0.86874485    1.29377369   -0.07163907
        H                  0.28549447    0.06799101    1.45462711
        H                  1.44553946   -1.32386345   -0.24456986
        H                  0.61096295   -0.50262210   -1.54153222
        H                 -0.24653265    2.11136864   -0.37045418
        H                 -0.21131163   -1.73585284   -0.61629002
        H                 -1.51770930    1.60958621    0.71830245
        H                 -1.45448167    0.96793094   -0.90568876"""
        cls.r_xyz_2b = """C                  0.50180491   -0.93942231   -0.57086745
        C                  0.01278145    0.13148427    0.42191407
        H                  0.28549447    0.06799101    1.45462711
        H                  1.44553946   -1.32386345   -0.24456986
        H                  0.61096295   -0.50262210   -1.54153222
        H                 -0.24653265    2.11136864   -0.37045418
        C                 -0.86874485    1.29377369   -0.07163907
        H                 -0.21131163   -1.73585284   -0.61629002
        H                 -1.51770930    1.60958621    0.71830245
        H                 -1.45448167    0.96793094   -0.90568876"""
        cls.p_xyz_2 = """C                  0.48818717   -0.94549701   -0.55196729
        C                  0.35993708    0.29146456    0.35637075
        C                 -0.91834764    1.06777042   -0.01096751
        H                  0.30640232   -0.02058840    1.37845537
        H                  1.37634603   -1.48487836   -0.29673876
        H                  0.54172192   -0.63344406   -1.57405191
        H                  1.21252186    0.92358349    0.22063264
        H                 -0.36439762   -1.57761595   -0.41622918
        H                 -1.43807526    1.62776079    0.73816131
        H                 -1.28677889    1.04716138   -1.01532486"""
        cls.ts_xyz_2 = """C       0.52123900   -0.93806900   -0.55301700
        C       0.15387500    0.18173100    0.37122900
        C      -0.89554000    1.16840700   -0.01362800
        H       0.33997700    0.06424800    1.44287100
        H       1.49602200   -1.37860200   -0.29763200
        H       0.57221700   -0.59290500   -1.59850500
        H       0.39006800    1.39857900   -0.01389600
        H      -0.23302200   -1.74751100   -0.52205400
        H      -1.43670700    1.71248300    0.76258900
        H      -1.32791000    1.11410600   -1.01554900"""  # C[CH]C <=> [CH2]CC
        cls.ts_spc_2 = ARCSpecies(label='TS', is_ts=True, xyz=cls.ts_xyz_2)
        cls.ts_spc_2.mol_from_xyz()
        cls.reactant_2a = ARCSpecies(label='C[CH]C', smiles='C[CH]C', xyz=cls.r_xyz_2a)
        cls.reactant_2b = ARCSpecies(label='C[CH]C', smiles='C[CH]C',
                                     xyz=cls.r_xyz_2b)  # same as 2a, only one C atom shifted place in the reactant xyz
        cls.product_2 = ARCSpecies(label='[CH2]CC', smiles='[CH2]CC', xyz=cls.p_xyz_2)
        cls.rxn_2a = ARCReaction(r_species=[cls.reactant_2a], p_species=[cls.product_2])
        cls.rxn_2a.ts_species = cls.ts_spc_2
        cls.rxn_2b = ARCReaction(r_species=[cls.reactant_2b], p_species=[cls.product_2])
        cls.rxn_2b.ts_species = cls.ts_spc_2

        cls.spc1 = ARCSpecies(label='[CH2]C(C)CO_a', smiles='[CH2]C(C)CO',
                              xyz={'symbols': ('C', 'H', 'H', 'C', 'C', 'C', 'O', 'H', 'H', 'H', 'H', 'H', 'H', 'H'),
                                   'isotopes': (12, 1, 1, 12, 12, 12, 16, 1, 1, 1, 1, 1, 1, 1),
                                   'coords': ((0.025711531222639566, 1.5002469234994276, -0.018809721320361607),
                                              (-0.2501237905589279, 2.283276320160058, 0.6795778782867752),
                                              (0.21710649528235348, 1.7701501165266882, -1.0518607878262018),
                                              (-0.1296127183749531, 0.05931626777072968, 0.3829802045651552),
                                              (-1.5215969202773243, -0.4341372833972907, -0.0024458040153687616),
                                              (0.954275466146204, -0.8261822387409435, -0.2512878552942834),
                                              (2.238645869558612, -0.5229077195628998, 0.2868843893740711),
                                              (-0.022719509344805086, 0.012299638536749403, 1.47391586262432),
                                              (-1.6734988982808552, -1.4656213151526711, 0.3333615031669381),
                                              (-1.6708084550075688, -0.40804497485420527, -1.0879383468423085),
                                              (-2.3005261427143897, 0.18308085969254126, 0.45923715033920876),
                                              (0.7583076310662862, -1.882720433150506, -0.04089782108496264),
                                              (0.9972006722528377, -0.7025586995487184, -1.3391950754631268),
                                              (2.377638769033351, 0.43380253822255727, 0.17647842348371048))})
        cls.spc2 = ARCSpecies(label='[CH2]C(C)CO_b', smiles='[CH2]C(C)CO',
                              xyz={'symbols': ('C', 'C', 'C', 'C', 'O', 'H', 'H', 'H', 'H', 'H', 'H', 'H', 'H', 'H'),
                                   'isotopes': (12, 12, 12, 12, 16, 1, 1, 1, 1, 1, 1, 1, 1, 1),
                                   'coords': ((-1.3857811794963277, -1.3882629357157468, -0.09505562903985151),
                                              (-0.48149615440373633, -0.18843419821506932, -0.36867730403761334),
                                              (-1.1615061896768615, 1.1047002102194075, 0.08616180242702906),
                                              (0.8755815877704686, -0.37530244696805926, 0.3151370087166933),
                                              (1.7499930104893404, 0.685479589154504, -0.04657790660423845),
                                              (-1.5824305690669607, -1.5021839148592626, 0.9764743462618697),
                                              (-0.9236829644275987, -2.313486599576571, -0.4551984262103633),
                                              (-0.31894490259166897, -0.11004093787895164, -1.45123483619259),
                                              (-0.551069637667873, 1.9799864363130495, -0.1585701723917383),
                                              (-2.1300920179099943, 1.2282601298258158, -0.4100421867371722),
                                              (-1.3337649482883558, 1.102945655452365, 1.1678836912458532),
                                              (0.7709290243761263, -0.38422053705817527, 1.4054470816682596),
                                              (1.337910696892115, -1.3171321272490044, 0.001256204546378134),
                                              (2.595292874972368, 0.5254618254772234, 0.4066018700054956))})
        cls.h2_xyz = {'coords': ((0, 0, 0.3736550), (0, 0, -0.3736550)),
                      'isotopes': (1, 1), 'symbols': ('H', 'H')}
        cls.c3h6o_xyz = {'coords': ((-1.0614352911982476, -0.35086070951203013, 0.3314546936475969),
                                    (0.08232694092180896, 0.5949821397504677, 0.020767511136565348),
                                    (1.319643623472743, -0.1238222051358961, -0.4579284002686819),
                                    (1.4145501246584122, -1.339374145335546, -0.5896335370976351),
                                    (-0.7813545474862899, -1.0625754884160945, 1.1151404910689675),
                                    (-1.3481804813952152, -0.9258389945508673, -0.5552942813558058),
                                    (-1.9370566523150816, 0.2087367432207233, 0.6743848589525232),
                                    (-0.2162279757671984, 1.3021306884228383, -0.7596873819624604),
                                    (0.35220978385921775, 1.1650050778348893, 0.9154971248602527),
                                    (2.1755244752498673, 0.5316168937214946, -0.6947010789813145)),
                         'isotopes': (12, 12, 12, 16, 1, 1, 1, 1, 1, 1),
                         'symbols': ('C', 'C', 'C', 'O', 'H', 'H', 'H', 'H', 'H', 'H')}
        cls.c4h9o_xyz = cls.spc1.get_xyz()
        cls.c3h5o_xyz = {'coords': ((-1.1339526749599567, -0.11366348271898848, -0.17361178233231772),
                                    (0.1315989608873882, 0.19315012600914244, 0.5375291058021542),
                                    (0.12186476447223683, 0.5479023323381329, 1.5587521800625246),
                                    (1.435623589506148, 0.026762256080503182, -0.11697684942586563),
                                    (1.5559845484585495, -0.3678359306766861, -1.2677014903374604),
                                    (-1.6836994309836657, -0.8907558916446712, 0.3657463577153353),
                                    (-1.7622426221647125, 0.7810307051429465, -0.21575166529131876),
                                    (-0.9704526962734873, -0.4619573344933834, -1.1970278328709658),
                                    (2.3052755610575106, 0.2853672199629854, 0.5090419766779545)),
                         'isotopes': (12, 12, 1, 12, 16, 1, 1, 1, 1),
                         'symbols': ('C', 'C', 'H', 'C', 'O', 'H', 'H', 'H', 'H')}
        cls.c4h10o_xyz = {'coords': ((-1.0599869990613344, -1.2397714287161459, 0.010871360821665921),
                                     (-0.15570197396874313, -0.0399426912154684, -0.2627503141760959),
                                     (-0.8357120092418682, 1.2531917172190083, 0.1920887922885465),
                                     (1.2013757682054618, -0.22681093996845836, 0.42106399857821075),
                                     (2.0757871909243337, 0.8339710961541049, 0.05934908325727899),
                                     (-1.2566363886319676, -1.3536924078596617, 1.082401336123387),
                                     (-0.5978887839926055, -2.1649950925769703, -0.3492714363488459),
                                     (-2.0220571570609596, -1.1266512469159389, -0.4999630281827645),
                                     (0.0068492778433242255, 0.03845056912064928, -1.3453078463310726),
                                     (-0.22527545723287978, 2.1284779433126504, -0.05264318253022085),
                                     (-1.804297837475001, 1.3767516368254167, -0.30411519687565475),
                                     (-1.0079707678533625, 1.2514371624519658, 1.2738106811073706),
                                     (1.0967232048111195, -0.23572903005857432, 1.511374071529777),
                                     (1.6637048773271081, -1.1686406202494035, 0.10718319440789557),
                                     (2.9210870554073614, 0.6739533324768243, 0.512528859867013)),
                          'isotopes': (12, 12, 12, 12, 16, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1),
                          'symbols': ('C', 'C', 'C', 'C', 'O', 'H', 'H', 'H', 'H', 'H', 'H', 'H', 'H', 'H', 'H')}
        cls.cccoj = ARCSpecies(label='CCCOj', smiles='CC=C[O]',
                               xyz={'symbols': ('C', 'C', 'C', 'O', 'H', 'H', 'H', 'H', 'H'),
                                    'isotopes': (12, 12, 12, 16, 1, 1, 1, 1, 1),
                                    'coords': ((-1.5025952162720535, 0.0009167580572484457, 0.49334397442432165),
                                               (-0.358832984151997, 0.9467596073197462, 0.18265679191329012),
                                               (0.8784836983989371, 0.2279552624333825, -0.29603911949195716),
                                               (0.9733901995846063, -0.9875966777662675, -0.42774425632091034),
                                               (-1.2225144725600958, -0.7107980208468159, 1.2770297718456922),
                                               (-1.7893404064690213, -0.5740615269815887, -0.39340500057908107),
                                               (-2.3782165773888875, 0.5605142107900019, 0.8362741397292479),
                                               (-0.08895014121458822, 1.516782545404168, 1.0773864056369775),
                                               (1.7343645501760614, 0.8833943612907731, -0.5328117982045897))})
        cls.ccjco = ARCSpecies(label='CCjCO', smiles='CC=C[O]', xyz=cls.c3h5o_xyz)
        cls.chiral_spc_1 = ARCSpecies(label='chiral_1', xyz="""C                 -0.81825240   -0.04911020   -0.14065159
                                                               H                 -1.34163466   -0.39900096   -1.00583797
                                                               C                  0.51892324    0.16369053   -0.19760928
                                                               H                  1.05130979   -0.01818286   -1.10776676
                                                               O                 -1.52975971    0.19395459    1.07572699
                                                               H                 -2.43039815    0.45695722    0.87255216
                                                               C                  1.27220245    0.66727126    1.04761235
                                                               H                  1.28275235    1.73721734    1.04963240
                                                               N                  0.59593728    0.18162740    2.25910542
                                                               H                  0.58607755   -0.81832221    2.25721751
                                                               H                  1.08507962    0.50862787    3.06769089
                                                               S                  2.94420440    0.05748035    1.01655601
                                                               H                  3.58498087    0.48585096    2.07580298""")
        cls.chiral_spc_1_b = ARCSpecies(label='chiral_1b', xyz="""C                 -0.81825240   -0.04911020   -0.14065159
                                                                  S                  2.94420440    0.05748035    1.01655601
                                                                  H                 -1.34163466   -0.39900096   -1.00583797
                                                                  N                  0.59593728    0.18162740    2.25910542
                                                                  H                  0.58607755   -0.81832221    2.25721751
                                                                  H                  1.05130979   -0.01818286   -1.10776676
                                                                  O                 -1.52975971    0.19395459    1.07572699
                                                                  H                 -2.43039815    0.45695722    0.87255216
                                                                  C                  0.51892324    0.16369053   -0.19760928
                                                                  C                  1.27220245    0.66727126    1.04761235
                                                                  H                  1.28275235    1.73721734    1.04963240
                                                                  H                  1.08507962    0.50862787    3.06769089
                                                                  H                  3.58498087    0.48585096    2.07580298""")  # same as chiral_spc_1, different atom order
        cls.chiral_spc_2 = ARCSpecies(label='chiral_2', xyz="""C                 -0.87981815   -0.20807053    0.19322984
                                                               C                  0.42332778    0.13088820    0.03998264
                                                               H                  0.86277627    0.13934039   -0.93557545
                                                               C                  1.27169817    0.50390374    1.26991235
                                                               H                  1.20358506    1.55745467    1.44395558
                                                               S                  0.66715728   -0.37334195    2.69587529
                                                               H                 -0.58281416   -0.04639937    2.91216198
                                                               N                  2.67433788    0.13702924    1.02720897
                                                               H                  3.01396759    0.62986390    0.22610619
                                                               H                  3.22522774    0.37924712    1.82586462
                                                               O                 -1.66759006   -0.55444424   -0.94884750
                                                               H                 -2.58359979   -0.31485264   -0.79034816
                                                               H                 -1.31926707   -0.21652112    1.16878775""")
        cls.fingerprint_1 = {0: {'self': 'C', 'C': [3], 'H': [1, 2]},
                             3: {'self': 'C', 'chirality': 'S', 'C': [0, 4, 5], 'H': [7]},
                             4: {'self': 'C', 'C': [3], 'H': [8, 9, 10]},
                             5: {'self': 'C', 'C': [3], 'O': [6], 'H': [11, 12]},
                             6: {'self': 'O', 'C': [5], 'H': [13]}}
        cls.fingerprint_2 = {0: {'self': 'C', 'C': [1], 'H': [5, 6]},
                             1: {'self': 'C', 'chirality': 'S', 'C': [0, 2, 3], 'H': [7]},
                             2: {'self': 'C', 'C': [1], 'H': [8, 9, 10]},
                             3: {'self': 'C', 'C': [1], 'O': [4], 'H': [11, 12]},
                             4: {'self': 'O', 'C': [3], 'H': [13]}}

    def test_map_h_abstraction(self):
        """Test the map_h_abstraction() function."""
        # H + CH4 <=> H2 + CH3
        r_1 = ARCSpecies(label='H', smiles='[H]', xyz={'coords': ((0, 0, 0),), 'isotopes': (1,), 'symbols': ('H',)})
        r_2 = ARCSpecies(label='CH4', smiles='C', xyz=self.ch4_xyz)
        p_1 = ARCSpecies(label='H2', smiles='[H][H]', xyz=self.h2_xyz)
        p_2 = ARCSpecies(label='CH3', smiles='[CH3]', xyz=self.ch3_xyz)
        rxn = ARCReaction(r_species=[r_1, r_2], p_species=[p_1, p_2])
        atom_map = mapping.map_h_abstraction(rxn=rxn, db=self.rmgdb, backend='ARC')
        self.assertIn(atom_map[0], [0, 1])
        self.assertEqual(atom_map[1], 2)
        for index in [2, 3, 4, 5]:
            self.assertIn(atom_map[index], [0, 1, 3, 4, 5])
        self.assertTrue(any(atom_map[r_index] in [0, 1] for r_index in [2, 3, 4, 5]))

        # H + CH4 <=> CH3 + H2 (different order)
        rxn = ARCReaction(r_species=[r_1, r_2], p_species=[p_2, p_1])
        atom_map = mapping.map_h_abstraction(rxn=rxn, db=self.rmgdb)
        self.assertIn(atom_map[0], [4, 5])
        self.assertEqual(atom_map[1], 0)
        for index in [2, 3, 4, 5]:
            self.assertIn(atom_map[index], [1, 2, 3, 4, 5])
        self.assertTrue(any(atom_map[r_index] in [4, 5] for r_index in [2, 3, 4, 5]))

        # CH4 + H <=> H2 + CH3 (different order)
        rxn = ARCReaction(r_species=[r_2, r_1], p_species=[p_1, p_2])
        atom_map = mapping.map_h_abstraction(rxn=rxn, db=self.rmgdb)
        self.assertEqual(atom_map[0], 2)
        for index in [1, 2, 3, 4]:
            self.assertIn(atom_map[index], [0, 1, 3, 4, 5])
        self.assertTrue(any(atom_map[r_index] in [0, 1] for r_index in [1, 2, 3, 4]))
        self.assertIn(atom_map[5], [0, 1])

        # CH4 + H <=> CH3 + H2 (different order)
        rxn = ARCReaction(r_species=[r_2, r_1], p_species=[p_2, p_1])
        atom_map = mapping.map_h_abstraction(rxn=rxn, db=self.rmgdb)
        self.assertEqual(atom_map[0], 0)
        for index in [1, 2, 3, 4]:
            self.assertIn(atom_map[index], [1, 2, 3, 4, 5])
        self.assertTrue(any(atom_map[r_index] in [4, 5] for r_index in [1, 2, 3, 4]))
        self.assertIn(atom_map[5], [4, 5])

        # H + CH4 <=> H2 + CH3 using QCElemental as the backend.
        rxn = ARCReaction(r_species=[r_1, r_2], p_species=[p_1, p_2])
        atom_map = mapping.map_h_abstraction(rxn=rxn, db=self.rmgdb, backend='QCElemental')
        self.assertIn(atom_map[0], [0, 1])
        self.assertEqual(atom_map[1], 2)
        for index in [2, 3, 4, 5]:
            self.assertIn(atom_map[index], [0, 1, 3, 4, 5])
        self.assertTrue(any(atom_map[r_index] in [0, 1] for r_index in [2, 3, 4, 5]))

        # H + CH3NH2 <=> H2 + CH2NH2
        ch3nh2_xyz = {'coords': ((-0.5734111454228507, 0.0203516083213337, 0.03088703933770556),
                                 (0.8105595891860601, 0.00017446498908627427, -0.4077728757313545),
                                 (-1.1234549667791063, -0.8123899006368857, -0.41607711106038836),
                                 (-0.6332220120842996, -0.06381791823047896, 1.1196983583774054),
                                 (-1.053200912106195, 0.9539501896695028, -0.27567270246542575),
                                 (1.3186422395164141, 0.7623906284020254, 0.038976118645639976),
                                 (1.2540872076899663, -0.8606590725145833, -0.09003882710357966)),
                      'isotopes': (12, 14, 1, 1, 1, 1, 1),
                      'symbols': ('C', 'N', 'H', 'H', 'H', 'H', 'H')}
        ch2nh2_xyz = {'coords': ((0.6919493009211066, 0.054389375309083846, 0.02065422596281878),
                                 (1.3094508022837807, -0.830934909576592, 0.14456347719459348),
                                 (1.1649142139806816, 1.030396183273415, 0.08526955368597328),
                                 (-0.7278194451655412, -0.06628299353512612, -0.30657582460750543),
                                 (-1.2832757211903472, 0.7307667658607352, 0.00177732009031573),
                                 (-1.155219150829674, -0.9183344213315149, 0.05431124767380799)),
                      'isotopes': (12, 1, 1, 14, 1, 1),
                      'symbols': ('C', 'H', 'H', 'N', 'H', 'H')}
        r_1 = ARCSpecies(label='H', smiles='[H]', xyz={'coords': ((0, 0, 0),), 'isotopes': (1,), 'symbols': ('H',)})
        r_2 = ARCSpecies(label='CH3NH2', smiles='CN', xyz=ch3nh2_xyz)
        p_1 = ARCSpecies(label='H2', smiles='[H][H]', xyz=self.h2_xyz)
        p_2 = ARCSpecies(label='CH2NH2', smiles='[CH2]N', xyz=ch2nh2_xyz)
        rxn = ARCReaction(r_species=[r_1, r_2], p_species=[p_1, p_2])
        atom_map = mapping.map_h_abstraction(rxn=rxn, db=self.rmgdb)
        self.assertEqual(atom_map[0], 0)
        self.assertEqual(atom_map[1], 2)
        self.assertEqual(atom_map[2], 5)
        self.assertIn(atom_map[3], [0, 1, 3, 4])
        self.assertIn(atom_map[4], [0, 1, 3, 4])
        self.assertIn(atom_map[5], [0, 1, 3, 4])
        self.assertTrue(any(atom_map[r_index] in [0, 1] for r_index in [3, 4, 5]))
        self.assertIn(atom_map[6], [6, 7])
        self.assertIn(atom_map[7], [6, 7])

        # CH4 + OH <=> CH3 + H2O
        r_1 = ARCSpecies(label='CH4', smiles='C', xyz=self.ch4_xyz)
        r_2 = ARCSpecies(label='OH', smiles='[OH]', xyz=self.oh_xyz)
        p_1 = ARCSpecies(label='CH3', smiles='[CH3]', xyz=self.ch3_xyz)
        p_2 = ARCSpecies(label='H2O', smiles='O', xyz=self.h2o_xyz)
        rxn = ARCReaction(r_species=[r_1, r_2], p_species=[p_1, p_2])
        atom_map = mapping.map_h_abstraction(rxn=rxn, db=self.rmgdb)
        self.assertEqual(atom_map[0], 0)
        self.assertIn(atom_map[1], [1, 2, 3, 5, 6])
        self.assertIn(atom_map[2], [1, 2, 3, 5, 6])
        self.assertIn(atom_map[3], [1, 2, 3, 5, 6])
        self.assertIn(atom_map[4], [1, 2, 3, 5, 6])
        self.assertEqual(atom_map[5], 4)
        self.assertIn(atom_map[6], [5, 6])
        self.assertTrue(any(atom_map[r_index] in [5, 6] for r_index in [1, 2, 3, 4]))

        # NH2 + N2H4 <=> NH3 + N2H3
        r_1 = ARCSpecies(label='NH2', smiles='[NH2]', xyz=self.nh2_xyz)
        r_2 = ARCSpecies(label='N2H4', smiles='NN', xyz=self.n2h4_xyz)
        p_1 = ARCSpecies(label='NH3', smiles='N', xyz=self.nh3_xyz)
        p_2 = ARCSpecies(label='N2H3', smiles='N[NH]', xyz=self.n2h3_xyz)
        rxn = ARCReaction(r_species=[r_1, r_2], p_species=[p_1, p_2])
        atom_map = mapping.map_h_abstraction(rxn=rxn, db=self.rmgdb)
        self.assertEqual(atom_map[0], 0)
        self.assertIn(atom_map[1], [1, 2, 3])
        self.assertIn(atom_map[2], [1, 2, 3])
        self.assertIn(atom_map[3], [4, 5])
        self.assertIn(atom_map[4], [4, 5])
        self.assertTrue(any(atom_map[r_index] in [1, 2, 3] for r_index in [5, 6, 7, 8]))

        # NH2 + N2H4 <=> N2H3 + NH3 (reversed product order compared to the above reaction)
        rxn = ARCReaction(r_species=[r_1, r_2], p_species=[p_2, p_1])
        atom_map = mapping.map_h_abstraction(rxn=rxn, db=self.rmgdb)
        self.assertEqual(atom_map[0], 5)
        self.assertIn(atom_map[1], [6, 7, 8])
        self.assertIn(atom_map[2], [6, 7, 8])
        self.assertIn(atom_map[3], [0, 1])
        self.assertIn(atom_map[4], [0, 1])
        self.assertTrue(any(atom_map[r_index] in [6, 7, 8] for r_index in [5, 6, 7, 8]))

        r_1 = ARCSpecies(label='CH3OO', smiles='CO[O]', xyz="""C      -0.41690000    0.03757000    0.00590000
                                                               O       0.83973000    0.69383000   -0.05239000
                                                               O       1.79663000   -0.33527000   -0.02406000
                                                               H      -0.54204000   -0.62249000   -0.85805000
                                                               H      -1.20487000    0.79501000   -0.01439000
                                                               H      -0.50439000   -0.53527000    0.93431000""")
        r_2 = ARCSpecies(label='CH3CH2OH', smiles='CCO', xyz="""C      -0.97459464    0.29181710    0.10303882
                                                                C       0.39565894   -0.35143697    0.10221676
                                                                O       0.30253309   -1.63748710   -0.49196889
                                                                H      -1.68942501   -0.32359616    0.65926091
                                                                H      -0.93861751    1.28685508    0.55523033
                                                                H      -1.35943743    0.38135479   -0.91822428
                                                                H       0.76858330   -0.46187184    1.12485643
                                                                H       1.10301149    0.25256708   -0.47388355
                                                                H       1.19485981   -2.02360458   -0.47786539""")
        p_1 = ARCSpecies(label='CH3OOH', smiles='COO', xyz="""C      -0.76039072    0.01483858   -0.00903344
                                                              O       0.44475333    0.76952102    0.02291303
                                                              O       0.16024511    1.92327904    0.86381800
                                                              H      -1.56632337    0.61401630   -0.44251282
                                                              H      -1.02943316   -0.30449156    1.00193709
                                                              H      -0.60052507   -0.86954495   -0.63086438
                                                              H       0.30391344    2.59629139    0.17435159""")
        p_2 = ARCSpecies(label='CH3CH2O', smiles='CC[O]', xyz="""C      -0.74046271    0.02568566   -0.00568694
                                                                 C       0.79799272   -0.01511040    0.00517437
                                                                 O       1.17260343   -0.72227959   -1.04851579
                                                                 H      -1.13881231   -0.99286049    0.06963185
                                                                 H      -1.14162013    0.59700303    0.84092854
                                                                 H      -1.13266865    0.46233725   -0.93283228
                                                                 H       1.11374677    1.03794239    0.06905096
                                                                 H       1.06944350   -0.38306117    1.00698657""")
        # CH3OO + CH3CH2OH <=> CH3OOH + CH3CH2O  / peroxyl to alkoxyl
        rxn = ARCReaction(r_species=[r_1, r_2], p_species=[p_1, p_2])
        atom_map = mapping.map_h_abstraction(rxn=rxn, db=self.rmgdb)
        self.assertEqual(atom_map, [0, 1, 2, 4, 5, 3, 7, 8, 9, 10, 11, 12, 14, 13, 6])

        # CH3OO + CH3CH2OH <=> CH3OOH + CH3CH2O  / peroxyl to alkoxyl, modified atom and product order
        r_2 = ARCSpecies(label='CH3CH2OH', smiles='CCO', xyz="""C      -0.97459464    0.29181710    0.10303882
                                                                C       0.39565894   -0.35143697    0.10221676
                                                                H      -1.68942501   -0.32359616    0.65926091
                                                                H      -0.93861751    1.28685508    0.55523033
                                                                H      -1.35943743    0.38135479   -0.91822428
                                                                H       0.76858330   -0.46187184    1.12485643
                                                                H       1.10301149    0.25256708   -0.47388355
                                                                O       0.30253309   -1.63748710   -0.49196889
                                                                H       1.19485981   -2.02360458   -0.47786539""")
        p_1 = ARCSpecies(label='CH3OOH', smiles='COO', xyz="""C      -0.76039072    0.01483858   -0.00903344
                                                              H      -1.56632337    0.61401630   -0.44251282
                                                              H      -1.02943316   -0.30449156    1.00193709
                                                              O       0.16024511    1.92327904    0.86381800
                                                              H      -0.60052507   -0.86954495   -0.63086438
                                                              O       0.44475333    0.76952102    0.02291303
                                                              H       0.30391344    2.59629139    0.17435159""")
        p_2 = ARCSpecies(label='CH3CH2O', smiles='CC[O]', xyz="""C       0.79799272   -0.01511040    0.00517437
                                                                 H      -1.13881231   -0.99286049    0.06963185
                                                                 O       1.17260343   -0.72227959   -1.04851579
                                                                 H      -1.14162013    0.59700303    0.84092854
                                                                 H      -1.13266865    0.46233725   -0.93283228
                                                                 C      -0.74046271    0.02568566   -0.00568694
                                                                 H       1.11374677    1.03794239    0.06905096
                                                                 H       1.06944350   -0.38306117    1.00698657""")
        rxn = ARCReaction(r_species=[r_1, r_2], p_species=[p_2, p_1])
        atom_map = mapping.map_h_abstraction(rxn=rxn, db=self.rmgdb)
        self.assertEqual(atom_map, [8, 13, 11, 10, 12, 9, 5, 0, 1, 3, 4, 7, 6, 2, 14])

        # C3H6O + OH <=> C3H5O + H2O
        r_1 = ARCSpecies(label='C3H6O', smiles='CCC=O', xyz=self.c3h6o_xyz)
        r_2 = ARCSpecies(label='OH', smiles='[OH]', xyz=self.oh_xyz)
        p_1 = ARCSpecies(label='C3H5O', smiles='C[CH]C=O', xyz=self.c3h5o_xyz)
        p_2 = ARCSpecies(label='H2O', smiles='O', xyz=self.h2o_xyz)
        rxn = ARCReaction(r_species=[r_1, r_2], p_species=[p_1, p_2])
        atom_map = mapping.map_h_abstraction(rxn=rxn, db=self.rmgdb)
        self.assertEqual(atom_map[:4], [0, 1, 3, 4])
        self.assertIn(atom_map[4], [5, 7])
        self.assertIn(atom_map[5], [6, 7])
        self.assertIn(atom_map[6], [5, 6])
        self.assertIn(atom_map[7], [2, 11])
        self.assertIn(atom_map[8], [2, 11])
        self.assertEqual(atom_map[9:], [8, 9, 10])

        # C4H10O + OH <=> C4H9O + H2O
        r_1 = ARCSpecies(label='C4H10O', smiles='CC(C)CO', xyz=self.c4h10o_xyz)
        r_2 = ARCSpecies(label='OH', smiles='[OH]', xyz=self.oh_xyz)
        p_1 = ARCSpecies(label='C4H9O', smiles='[CH2]C(C)CO', xyz=self.c4h9o_xyz)
        p_2 = ARCSpecies(label='H2O', smiles='O', xyz=self.h2o_xyz)
        rxn = ARCReaction(r_species=[r_1, r_2], p_species=[p_1, p_2])
        atom_map = mapping.map_h_abstraction(rxn=rxn, db=self.rmgdb)
        self.assertEqual(atom_map[:5], [0, 3, 4, 5, 6])
        for index in [5, 6, 7]:
            self.assertIn(atom_map[index], [1, 2, 15, 16])
        self.assertEqual(atom_map[8:16], [7, 8, 10, 9, 11, 12, 13, 14])
        self.assertIn(atom_map[16], [15, 16])

        # C3H6O + C4H9O <=> C3H5O + C4H10O
        r_1 = ARCSpecies(label='C3H6O', smiles='CCC=O', xyz=self.c3h6o_xyz)
        r_2 = ARCSpecies(label='C4H9O', smiles='[CH2]C(C)CO', xyz=self.c4h9o_xyz)
        p_1 = ARCSpecies(label='C3H5O', smiles='C[CH]C=O', xyz=self.c3h5o_xyz)
        p_2 = ARCSpecies(label='C4H10O', smiles='CC(C)CO', xyz=self.c4h10o_xyz)
        rxn = ARCReaction(r_species=[r_1, r_2], p_species=[p_1, p_2])
        atom_map = mapping.map_h_abstraction(rxn=rxn, db=self.rmgdb)
        self.assertEqual(atom_map[0:4], [0, 1, 3, 4])
        self.assertIn(atom_map[4], [5, 7])
        self.assertIn(atom_map[5], [6, 7])
        self.assertIn(atom_map[6], [5, 6])
        self.assertIn(atom_map[7], [2, 14, 15, 16, 18, 19, 20])
        self.assertIn(atom_map[8], [2, 14, 15, 16, 18, 19, 20])
        self.assertTrue(any(entry == 2 for entry in [atom_map[7], atom_map[8]]))
        self.assertEqual(atom_map[9], 8)
        self.assertIn(atom_map[10], [9])
        self.assertIn(atom_map[11], [14, 15, 16])
        self.assertIn(atom_map[12], [14, 15, 16])
        self.assertEqual(atom_map[13:], [10, 11, 12, 13, 17, 18, 20, 19, 21, 22, 23])

    def test_map_ho2_elimination_from_peroxy_radical(self):
        """Test the map_ho2_elimination_from_peroxy_radical() function."""
        r_xyz = """N      -0.82151000   -0.98211000   -0.58727000
                   C      -0.60348000    0.16392000    0.30629000
                   C       0.85739000    0.41515000    0.58956000
                   C       1.91892000   -0.27446000    0.14220000
                   O      -1.16415000    1.38916000   -0.20784000
                   O      -2.39497344    1.57487672    0.46214548
                   H      -0.50088000   -0.69919000   -1.51181000
                   H      -1.83926000   -1.03148000   -0.69340000
                   H      -1.09049000   -0.04790000    1.26633000
                   H       1.04975000    1.25531000    1.25575000
                   H       2.92700000    0.00462000    0.43370000
                   H       1.81273000   -1.13911000   -0.50660000"""  # NC(C=C)O[O]
        p_1_xyz = """N       1.16378795    1.46842703   -0.82620909
                     C       0.75492192    0.42940001   -0.18269967
                     C      -0.66835457    0.05917401   -0.13490822
                     C      -1.06020680   -1.02517494    0.54162130
                     H       2.18280085    1.55132949   -0.73741996
                     H       1.46479392   -0.22062618    0.35707573
                     H      -1.36374229    0.69906451   -0.66578157
                     H      -2.11095970   -1.29660899    0.57562763
                     H      -0.36304116   -1.66498540    1.07269317"""  # N=CC=C
        p_2_xyz = """N      -1.60333711   -0.23049987   -0.35673484
                     C      -0.63074775    0.59837442    0.08043329
                     C       0.59441219    0.18489797    0.16411656
                     C       1.81978128   -0.23541908    0.24564488
                     H      -2.56057110    0.09083582   -0.42266843
                     H      -1.37296018   -1.18147301   -0.62077856
                     H      -0.92437032    1.60768040    0.35200716
                     H       2.49347824   -0.13648710   -0.59717108
                     H       2.18431385   -0.69791121    1.15515621"""  # NC=C=C
        ho2_xyz = """O      -0.18935000    0.42639000    0.00000000
                     O       1.07669000   -0.17591000    0.00000000
                     H      -0.88668000   -0.25075000    0.00000000"""  # O[O]
        rxn_1 = ARCReaction(r_species=[ARCSpecies(label='R', smiles='NC(C=C)O[O]', xyz=r_xyz)],
                            p_species=[ARCSpecies(label='P1', smiles='N=CC=C', xyz=p_1_xyz),
                                       ARCSpecies(label='HO2', smiles='O[O]', xyz=ho2_xyz)])
        atom_map = mapping.map_ho2_elimination_from_peroxy_radical(rxn_1)
        self.assertEqual(atom_map[:6], [0, 1, 2, 3, 10, 9])
        self.assertIn(atom_map[6], [4, 11])
        self.assertIn(atom_map[7], [4, 11])
        self.assertEqual(atom_map[8], 5)
        self.assertEqual(atom_map[9], 6)
        self.assertIn(atom_map[10], [7, 8])
        self.assertIn(atom_map[11], [7, 8])

    def test_map_intra_h_migration(self):
        """Test the map_intra_h_migration() function."""
        atom_map = mapping.map_intra_h_migration(self.arc_reaction_4)
        self.assertEqual(atom_map[0], 0)
        self.assertEqual(atom_map[1], 1)
        self.assertEqual(atom_map[2], 2)
        self.assertIn(atom_map[3], [3, 4, 5])
        self.assertIn(atom_map[4], [3, 4, 5])
        self.assertIn(atom_map[5], [6, 7])
        self.assertIn(atom_map[6], [6, 7])
        self.assertIn(atom_map[7], [3, 4, 5, 8])
        self.assertIn(atom_map[8], [3, 4, 5, 8])

    def test_map_isomerization_reaction(self):
        """Test the map_isomerization_reaction() function."""
        reactant_xyz = """C  -1.3087    0.0068    0.0318
                          C   0.1715   -0.0344    0.0210
                          N   0.9054   -0.9001    0.6395
                          O   2.1683   -0.5483    0.3437
                          N   2.1499    0.5449   -0.4631
                          N   0.9613    0.8655   -0.6660
                          H  -1.6558    0.9505    0.4530
                          H  -1.6934   -0.0680   -0.9854
                          H  -1.6986   -0.8169    0.6255"""
        reactant = ARCSpecies(label='reactant', smiles='C([C]1=[N]O[N]=[N]1)', xyz=reactant_xyz)
        product_xyz = """C  -1.0108   -0.0114   -0.0610
                         C   0.4780    0.0191    0.0139
                         N   1.2974   -0.9930    0.4693
                         O   0.6928   -1.9845    0.8337
                         N   1.7456    1.9701   -0.6976
                         N   1.1642    1.0763   -0.3716
                         H  -1.4020    0.9134   -0.4821
                         H  -1.3327   -0.8499   -0.6803
                         H  -1.4329   -0.1554    0.9349"""
        product = ARCSpecies(label='product', smiles='[N-]=[N+]=C(N=O)C', xyz=product_xyz)
        rxn_1 = ARCReaction(label='reactant <=> product', ts_label='TS0', r_species=[reactant], p_species=[product])
        atom_map = mapping.map_isomerization_reaction(rxn_1)
        self.assertEqual(atom_map[:6], [0, 1, 2, 3, 4, 5])
        self.assertIn(atom_map[6], [6, 8])
        self.assertIn(atom_map[7], [6, 7])
        self.assertIn(atom_map[8], [7, 8])

    def test_get_atom_indices_of_labeled_atoms_in_an_rmg_reaction(self):
        """Test the get_atom_indices_of_labeled_atoms_in_an_rmg_reaction() function."""
        determine_family(self.arc_reaction_1, db=self.rmgdb)
        rmg_reactions = mapping.get_rmg_reactions_from_arc_reaction(arc_reaction=self.arc_reaction_1, db=self.rmgdb)
        r_dict, p_dict = mapping.get_atom_indices_of_labeled_atoms_in_an_rmg_reaction(arc_reaction=self.arc_reaction_1,
                                                                                      rmg_reaction=rmg_reactions[0])
        self.assertEqual(r_dict['*1'], 0)
        self.assertIn(r_dict['*2'], [1, 2, 3, 4])
        self.assertEqual(r_dict['*3'], 5)
        self.assertEqual(p_dict['*1'], 0)
        self.assertIn(p_dict['*2'], [5, 6])
        self.assertEqual(p_dict['*3'], 4)
        self.assertTrue(check_r_n_p_symbols_between_rmg_and_arc_rxns(self.arc_reaction_1, rmg_reactions))

        determine_family(self.arc_reaction_2, db=self.rmgdb)
        rmg_reactions = mapping.get_rmg_reactions_from_arc_reaction(arc_reaction=self.arc_reaction_2, db=self.rmgdb)
        r_dict, p_dict = mapping.get_atom_indices_of_labeled_atoms_in_an_rmg_reaction(arc_reaction=self.arc_reaction_2,
                                                                                      rmg_reaction=rmg_reactions[0])
        self.assertIn(r_dict['*1'], [0, 2])
        self.assertIn(r_dict['*2'], [3, 4, 5, 8, 9, 10])
        self.assertEqual(r_dict['*3'], 11)
        self.assertEqual(p_dict['*1'], 0)
        self.assertIn(p_dict['*2'], [11, 12, 13])
        self.assertEqual(p_dict['*3'], 10)
        self.assertTrue(check_r_n_p_symbols_between_rmg_and_arc_rxns(self.arc_reaction_2, rmg_reactions))


        determine_family(self.arc_reaction_4, db=self.rmgdb)
        rmg_reactions = mapping.get_rmg_reactions_from_arc_reaction(arc_reaction=self.arc_reaction_4, db=self.rmgdb)
        r_dict, p_dict = mapping.get_atom_indices_of_labeled_atoms_in_an_rmg_reaction(arc_reaction=self.arc_reaction_4,
                                                                                      rmg_reaction=rmg_reactions[0])
        self.assertEqual(r_dict['*1'], 0)
        self.assertEqual(r_dict['*2'], 2)
        self.assertIn(r_dict['*3'], [7, 8])
        self.assertEqual(p_dict['*1'], 0)
        self.assertEqual(p_dict['*2'], 2)
        self.assertIn(p_dict['*3'], [3, 4, 5])
        self.assertTrue(check_r_n_p_symbols_between_rmg_and_arc_rxns(self.arc_reaction_4, rmg_reactions))

        determine_family(self.rxn_2a, db=self.rmgdb)
        for atom, symbol in zip(self.rxn_2a.r_species[0].mol.atoms, ['C', 'C', 'C', 'H', 'H', 'H', 'H', 'H', 'H', 'H']):
            self.assertEqual(atom.symbol, symbol)
        self.assertEqual(self.rxn_2a.r_species[0].mol.atoms[0].radical_electrons, 0)
        self.assertEqual(self.rxn_2a.r_species[0].mol.atoms[1].radical_electrons, 1)
        self.assertEqual(self.rxn_2a.r_species[0].mol.atoms[2].radical_electrons, 0)
        self.assertEqual(self.rxn_2a.p_species[0].mol.atoms[0].radical_electrons, 0)
        self.assertEqual(self.rxn_2a.p_species[0].mol.atoms[1].radical_electrons, 0)
        self.assertEqual(self.rxn_2a.p_species[0].mol.atoms[2].radical_electrons, 1)
        rmg_reactions = mapping.get_rmg_reactions_from_arc_reaction(arc_reaction=self.rxn_2a, db=self.rmgdb)
        r_dict, p_dict = mapping.get_atom_indices_of_labeled_atoms_in_an_rmg_reaction(arc_reaction=self.rxn_2a,
                                                                                      rmg_reaction=rmg_reactions[0])
        self.assertEqual(r_dict['*1'], 1)
        self.assertIn(r_dict['*2'], [0, 2])
        self.assertIn(r_dict['*3'], [4, 5, 6, 7, 8, 9])
        self.assertEqual(p_dict['*1'], 1)
        self.assertEqual(p_dict['*2'], 2)
        self.assertIn(p_dict['*3'], [3, 6])
        self.assertTrue(check_r_n_p_symbols_between_rmg_and_arc_rxns(self.rxn_2a, rmg_reactions))

        determine_family(self.rxn_2b, db=self.rmgdb)
        for atom, symbol in zip(self.rxn_2b.r_species[0].mol.atoms, ['C', 'C', 'H', 'H', 'H', 'H', 'C', 'H', 'H', 'H']):
            self.assertEqual(atom.symbol, symbol)
        rmg_reactions = mapping.get_rmg_reactions_from_arc_reaction(arc_reaction=self.rxn_2b, db=self.rmgdb)
        r_dict, p_dict = mapping.get_atom_indices_of_labeled_atoms_in_an_rmg_reaction(arc_reaction=self.rxn_2b,
                                                                                      rmg_reaction=rmg_reactions[0])
        self.assertEqual(r_dict['*1'], 1)
        self.assertIn(r_dict['*2'], [0, 6])
        self.assertIn(r_dict['*3'], [3, 4, 5, 7, 8, 9])
        self.assertEqual(p_dict['*1'], 1)
        self.assertEqual(p_dict['*2'], 2)
        self.assertIn(p_dict['*3'], [3, 6])
        self.assertTrue(check_r_n_p_symbols_between_rmg_and_arc_rxns(self.rxn_2b, rmg_reactions))

        # C3H6O + C4H9O <=> C3H5O + C4H10O
        r_1 = ARCSpecies(label='C3H6O', smiles='CCC=O')
        r_2 = ARCSpecies(label='C4H9O', smiles='[CH2]C(C)CO')
        p_1 = ARCSpecies(label='C3H5O', smiles='C[CH]C=O')
        p_2 = ARCSpecies(label='C4H10O', smiles='CC(C)CO')
        rxn_1 = ARCReaction(reactants=['C3H6O', 'C4H9O'], products=['C3H5O', 'C4H10O'],
                            r_species=[r_1, r_2], p_species=[p_1, p_2])
        determine_family(rxn_1, db=self.rmgdb)
        rmg_reactions = mapping.get_rmg_reactions_from_arc_reaction(arc_reaction=rxn_1, db=self.rmgdb)
        for rmg_reaction in rmg_reactions:
            r_dict, p_dict = mapping.get_atom_indices_of_labeled_atoms_in_an_rmg_reaction(arc_reaction=rxn_1,
                                                                                          rmg_reaction=rmg_reaction)
            for d in [r_dict, p_dict]:
                self.assertEqual(len(list(d.keys())), 3)
                keys = list(d.keys())
                for label in ['*1', '*2', '*3']:
                    self.assertIn(label, keys)
        self.assertTrue(check_r_n_p_symbols_between_rmg_and_arc_rxns(rxn_1, rmg_reactions))

        p_1 = ARCSpecies(label='C3H5O', smiles='CC=C[O]')  # Use a wrong resonance structure and repeat the above.
        rxn_2 = ARCReaction(reactants=['C3H6O', 'C4H9O'], products=['C3H5O', 'C4H10O'],
                            r_species=[r_1, r_2], p_species=[p_1, p_2])
        determine_family(rxn_2, db=self.rmgdb)
        rmg_reactions = mapping.get_rmg_reactions_from_arc_reaction(arc_reaction=rxn_2, db=self.rmgdb)
        for rmg_reaction in rmg_reactions:
            r_dict, p_dict = mapping.get_atom_indices_of_labeled_atoms_in_an_rmg_reaction(arc_reaction=rxn_2,
                                                                                          rmg_reaction=rmg_reaction)
            for d in [r_dict, p_dict]:
                self.assertEqual(len(list(d.keys())), 3)
                keys = list(d.keys())
                for label in ['*1', '*2', '*3']:
                    self.assertIn(label, keys)
        self.assertTrue(check_r_n_p_symbols_between_rmg_and_arc_rxns(rxn_2, rmg_reactions))

        # C3H6O + C4H9O <=> C3H5O + C4H10O
        r_1 = ARCSpecies(label='C3H6O', smiles='CCC=O', xyz=self.c3h6o_xyz)
        r_2 = ARCSpecies(label='C4H9O', smiles='[CH2]C(C)CO', xyz=self.c4h9o_xyz)
        p_1 = ARCSpecies(label='C3H5O', smiles='C[CH]C=O', xyz=self.c3h5o_xyz)
        p_2 = ARCSpecies(label='C4H10O', smiles='CC(C)CO', xyz=self.c4h10o_xyz)
        rxn_3 = ARCReaction(reactants=['C3H6O', 'C4H9O'], products=['C3H5O', 'C4H10O'],
                            r_species=[r_1, r_2], p_species=[p_1, p_2])
        determine_family(rxn_3, db=self.rmgdb)
        rmg_reactions = mapping.get_rmg_reactions_from_arc_reaction(arc_reaction=rxn_3, db=self.rmgdb)
        r_dict, p_dict = mapping.get_atom_indices_of_labeled_atoms_in_an_rmg_reaction(arc_reaction=rxn_3,
                                                                                      rmg_reaction=rmg_reactions[0])
        self.assertEqual(r_dict, {'*3': 10, '*1': 1, '*2': 7})
        self.assertEqual(p_dict, {'*1': 1, '*3': 9, '*2': 16})
        self.assertTrue(check_r_n_p_symbols_between_rmg_and_arc_rxns(rxn_3, rmg_reactions))

    def test_map_arc_rmg_species(self):
        """Test the map_arc_rmg_species() function."""
        r_map, p_map = mapping.map_arc_rmg_species(arc_reaction=ARCReaction(r_species=[ARCSpecies(label='CCjC', smiles='C[CH]C')],
                                                                            p_species=[ARCSpecies(label='CjCC', smiles='[CH2]CC')]),
                                                   rmg_reaction=Reaction(reactants=[Species(smiles='C[CH]C')],
                                                                         products=[Species(smiles='[CH2]CC')]),
                                                   concatenate=False)
        self.assertEqual(r_map, {0: 0})
        self.assertEqual(p_map, {0: 0})

        r_map, p_map = mapping.map_arc_rmg_species(arc_reaction=ARCReaction(r_species=[ARCSpecies(label='CCjC', smiles='C[CH]C')],
                                                                            p_species=[ARCSpecies(label='CjCC', smiles='[CH2]CC')]),
                                                   rmg_reaction=Reaction(reactants=[Species(smiles='C[CH]C')],
                                                                         products=[Species(smiles='[CH2]CC')]))
        self.assertEqual(r_map, {0: [0]})
        self.assertEqual(p_map, {0: [0]})

        r_map, p_map = mapping.map_arc_rmg_species(arc_reaction=self.arc_reaction_1, rmg_reaction=self.rmg_reaction_1)
        self.assertEqual(r_map, {0: [0], 1: [1]})
        self.assertEqual(p_map, {0: [0], 1: [1]})

        r_map, p_map = mapping.map_arc_rmg_species(arc_reaction=self.arc_reaction_1, rmg_reaction=self.rmg_reaction_2)
        self.assertEqual(r_map, {0: [1], 1: [0]})
        self.assertEqual(p_map, {0: [0], 1: [1]})

        r_map, p_map = mapping.map_arc_rmg_species(arc_reaction=self.arc_reaction_3, rmg_reaction=self.rmg_reaction_3)
        self.assertEqual(r_map, {0: [0, 1], 1: [0, 1]})
        self.assertEqual(p_map, {0: [0]})

        rmg_reaction_1 = Reaction(reactants=[Species(smiles='N'), Species(smiles='[H]')],
                                  products=[Species(smiles='[NH2]'), Species(smiles='[H][H]')])
        rmg_reaction_2 = Reaction(reactants=[Species(smiles='[H]'), Species(smiles='N')],
                                  products=[Species(smiles='[H][H]'), Species(smiles='[NH2]')])
        rmg_reaction_3 = Reaction(reactants=[Species(smiles='N'), Species(smiles='[H]')],
                                  products=[Species(smiles='[H][H]'), Species(smiles='[NH2]')])
        arc_reaction = ARCReaction(r_species=[ARCSpecies(label='NH3', smiles='N'), ARCSpecies(label='H', smiles='[H]')],
                                   p_species=[ARCSpecies(label='NH2', smiles='[NH2]'), ARCSpecies(label='H2', smiles='[H][H]')])

        r_map, p_map = mapping.map_arc_rmg_species(rmg_reaction=rmg_reaction_1, arc_reaction=arc_reaction)
        self.assertEqual(r_map, {0: [0], 1: [1]})
        self.assertEqual(p_map, {0: [0], 1: [1]})

        r_map, p_map = mapping.map_arc_rmg_species(rmg_reaction=rmg_reaction_2, arc_reaction=arc_reaction)
        self.assertEqual(r_map, {0: [1], 1: [0]})
        self.assertEqual(p_map, {0: [1], 1: [0]})

        r_map, p_map = mapping.map_arc_rmg_species(rmg_reaction=rmg_reaction_3, arc_reaction=arc_reaction)
        self.assertEqual(r_map, {0: [0], 1: [1]})
        self.assertEqual(p_map, {0: [1], 1: [0]})

        rmg_reaction = Reaction(reactants=[Species(smiles='[CH3]'), Species(smiles='[CH3]')],
                                products=[Species(smiles='CC')])
        arc_reaction = ARCReaction(r_species=[ARCSpecies(label='CH3', smiles='[CH3]'),
                                              ARCSpecies(label='CH3', smiles='[CH3]')],
                                   p_species=[ARCSpecies(label='C2H6', smiles='CC')])

        r_map, p_map = mapping.map_arc_rmg_species(rmg_reaction=rmg_reaction, arc_reaction=arc_reaction)
        self.assertEqual(r_map, {0: [0, 1], 1: [0, 1]})
        self.assertEqual(p_map, {0: [0]})

    def test_find_equivalent_atoms_in_reactants_and_products(self):
        """Test the find_equivalent_atoms_in_reactants_and_products() function"""
        equivalence_map_1 = mapping.find_equivalent_atoms_in_reactants(arc_reaction=self.rxn_2a)
        # Both C 0 and C 2 are equivalent, C 1 is unique, and H 4-9 are equivalent as well.
        self.assertEqual(equivalence_map_1, [[0, 2], [1], [4, 5, 6, 7, 8, 9]])
        equivalence_map_2 = mapping.find_equivalent_atoms_in_reactants(arc_reaction=self.rxn_2b)
        self.assertEqual(equivalence_map_2, [[0, 6], [1], [3, 4, 5, 7, 8, 9]])

    def test_map_two_species(self):
        """Test the map_two_species() function."""
        # H
        spc1 = ARCSpecies(label='H', smiles='[H]')
        spc2 = ARCSpecies(label='H', smiles='[H]')
        atom_map = mapping.map_two_species(spc1, spc2)
        self.assertEqual(atom_map, [0])

        # OH same order
        spc1 = ARCSpecies(label='OH', smiles='[OH]', xyz="""O 0 0 0\nH 0.8 0 0""")
        spc2 = ARCSpecies(label='OH', smiles='[OH]', xyz="""O 0 0.9 0\nH 0 0 0""")
        atom_map = mapping.map_two_species(spc1, spc2)
        self.assertEqual(atom_map, [0, 1])

        # OH different order
        spc1 = ARCSpecies(label='OH', smiles='[OH]', xyz="""O 0 0 0\nH 0.8 0 0""")
        spc2 = ARCSpecies(label='OH', smiles='[OH]', xyz="""H 0 0 0\nO 0 0.9 0""")
        atom_map = mapping.map_two_species(spc1, spc2)
        self.assertEqual(atom_map, [1, 0])

        # N2 - homonuclear diatomic
        spc1 = ARCSpecies(label='N2', smiles='N#N', xyz="""N 0.73 0.0 0.0\nN -0.73 0.0 0.0""")
        spc2 = ARCSpecies(label='N2', smiles='N#N', xyz="""N 0 0 0\nN 0 1.5 0""")
        atom_map = mapping.map_two_species(spc1, spc2, map_type='dict')
        self.assertEqual(atom_map, {0: 0, 1: 1})

        # HNCO - all different elements
        spc1 = ARCSpecies(label='HNCO', smiles='N=C=O', xyz="""N      -0.70061553    0.28289128   -0.18856549
                                                               C       0.42761869    0.11537693    0.07336374
                                                               O       1.55063087   -0.07323229    0.35677630
                                                               H      -1.27763403   -0.32503592    0.39725197""")
        spc2 = ARCSpecies(label='HNCO', smiles='N=C=O', xyz="""C       0.4276    0.1153    0.0734
                                                               N      -0.7003    0.2828   -0.1885
                                                               O       1.5506   -0.0732    0.3567
                                                               H      -1.2776   -0.3250    0.3972""")
        atom_map = mapping.map_two_species(spc1, spc2)
        self.assertEqual(atom_map, [1, 0, 2, 3])

        # CH4 different order
        spc1 = ARCSpecies(label='CH4', smiles='C', xyz=self.ch4_xyz)
        spc2 = ARCSpecies(label='CH4', smiles='C', xyz=self.ch4_xyz_diff_order)
        atom_map = mapping.map_two_species(spc1, spc2)
        self.assertEqual(atom_map, [2, 0, 1, 3, 4])

        # Different resonance structures
        spc1 = ARCSpecies(label='CCHCHO', smiles='CC=C[O]',
                          xyz="""C      -1.13395267   -0.11366348   -0.17361178
                                 C       0.13159896    0.19315013    0.53752911
                                 H       0.12186476    0.54790233    1.55875218
                                 C       1.43562359    0.02676226   -0.11697685
                                 O       1.55598455   -0.36783593   -1.26770149
                                 H      -1.68369943   -0.89075589    0.36574636
                                 H      -1.76224262    0.78103071   -0.21575167
                                 H      -0.97045270   -0.46195733   -1.19702783
                                 H       2.30527556    0.28536722    0.50904198""")
        spc2 = ARCSpecies(label='CCCHO', smiles='C[CH]C=O',
                          xyz="""C      -1.06143529   -0.35086071    0.33145469
                                 C       0.08232694    0.59498214    0.02076751
                                 C       1.31964362   -0.12382221   -0.45792840
                                 O       1.41455012   -1.33937415   -0.58963354
                                 H      -0.78135455   -1.06257549    1.11514049
                                 H      -1.34818048   -0.92583899   -0.55529428
                                 H      -1.93705665    0.20873674    0.67438486
                                 H      -0.21622798    1.30213069   -0.75968738
                                 H       2.17552448    0.53161689   -0.69470108""")
        atom_map = mapping.map_two_species(spc1, spc2)
        self.assertEqual(atom_map, [0, 1, 7, 2, 3, 6, 5, 4, 8])

        # R1H w/o its H and R1 from an H abstraction reaction for C=CC=N.
        spc1 = ARCSpecies(label='CCCN_a', smiles='C=CC=N',
                          xyz="""N       1.16378795    1.46842703   -0.82620909
                                 C       0.75492192    0.42940001   -0.18269967
                                 C      -0.66835457    0.05917401   -0.13490822
                                 C      -1.06020680   -1.02517494    0.54162130
                                 H       2.18280085    1.55132949   -0.73741996
                                 H       1.46479392   -0.22062618    0.35707573
                                 H      -1.36374229    0.69906451   -0.66578157
                                 H      -2.11095970   -1.29660899    0.57562763
                                 H      -0.36304116   -1.66498540    1.07269317""")
        spc2 = ARCSpecies(label='CCCN_b', smiles='C=CC=N',
                          xyz="""N      -0.82151000   -0.98211000   -0.58727000
                                 C      -0.60348000    0.16392000    0.30629000
                                 C       0.85739000    0.41515000    0.58956000
                                 C       1.39979110    1.37278509    1.35850405
                                 H      -1.83926000   -1.03148000   -0.69340000
                                 H      -1.09049000   -0.04790000    1.26633000
                                 H       1.53896925   -0.26718857    0.08305817
                                 H       2.47688530    1.45345357    1.47005614
                                 H       0.79186768    2.10712053    1.87909788""")
        atom_map = mapping.map_two_species(spc1, spc2)
        self.assertEqual(atom_map, [0, 1, 2, 3, 4, 5, 6, 7, 8])

        # R1H w/o its H and R1 from an H abstraction reaction for [CH2]N.
        spc1 = ARCSpecies(label='CH2NH2_a', smiles='[CH2]N',
                          xyz="""C      -0.75196103   -0.01443262    0.18205588
                                 N       0.63200970   -0.03460976   -0.25660404
                                 H      -0.81177190   -0.09860215    1.27086719
                                 H      -1.23175080    0.91916596   -0.12450387
                                 H       1.14009235    0.72760640    0.19014495
                                 H       1.07553732   -0.89544330    0.06113001""")
        spc2 = ARCSpecies(label='CH2NH2_b', smiles='[CH2]N',
                          xyz="""C       0.69194930    0.05438938    0.02065423
                                 H       1.30945080   -0.83093491    0.14456348
                                 H       1.16491421    1.03039618    0.08526955
                                 N      -0.72781945   -0.06628299   -0.30657582
                                 H      -1.28327572    0.73076677    0.00177732
                                 H      -1.15521915   -0.91833442    0.05431125""")
        atom_map = mapping.map_two_species(spc1, spc2)
        self.assertEqual(atom_map[0], 0)
        self.assertEqual(atom_map[1], 3)
        self.assertIn(atom_map[2], [1, 2])
        self.assertIn(atom_map[3], [1, 2])
        self.assertIn(atom_map[4], [4, 5])
        self.assertIn(atom_map[5], [4, 5])

        # Chiral center and two internal rotations.
        atom_map = mapping.map_two_species(self.spc1, self.spc2)
        self.assertEqual(atom_map[0], 0)  # part of the backbone
        for index in [1, 2]:
            self.assertIn(atom_map[index], [5, 6])  # H's on terminal CH2
        self.assertEqual(atom_map[3:7], [1, 2, 3, 4])  # part of the backbone
        self.assertEqual(atom_map[7], 7)  # H on tertiary C
        for index in [8, 9, 10]:
            self.assertIn(atom_map[index], [8, 9, 10])  # H's on CH3
        for index in [11, 12]:
            self.assertIn(atom_map[index], [11, 12])  # H's on internal CH2
        self.assertEqual(atom_map[13], 13)  # H on O atom

        # Multiple chiral centers
        atom_map = mapping.map_two_species(self.chiral_spc_1, self.chiral_spc_1_b)
        self.assertEqual(atom_map, [0, 2, 8, 5, 6, 7, 9, 10, 3, 4, 11, 1, 12])

        # Different resonance structures.
        atom_map = mapping.map_two_species(self.cccoj, self.ccjco)
        self.assertEqual(atom_map[:4], [0, 1, 3, 4])
        for index in [4, 5, 6]:
            self.assertIn(atom_map[index], [5, 6, 7])
        self.assertEqual(atom_map[7], 2)
        self.assertEqual(atom_map[8], 8)

    def test_get_arc_species(self):
        """Test the get_arc_species function."""
        self.assertIsInstance(mapping.get_arc_species(ARCSpecies(label='S', smiles='C')), ARCSpecies)
        self.assertIsInstance(mapping.get_arc_species(Species(smiles='C')), ARCSpecies)
        self.assertIsInstance(mapping.get_arc_species(Molecule(smiles='C')), ARCSpecies)

    def test_create_qc_mol(self):
        """Test the create_qc_mol() function."""
        qcmol1 = mapping.create_qc_mol(species=ARCSpecies(label='S1', smiles='C'))
        self.assertIsInstance(qcmol1, QCMolecule)
        self.assertEqual(qcmol1.molecular_charge, 0)
        self.assertEqual(qcmol1.molecular_multiplicity, 1)
        for symbol, expected_symbol in zip(qcmol1.symbols, ['C', 'H', 'H', 'H', 'H']):
            self.assertEqual(symbol, expected_symbol)

        qcmol2 = mapping.create_qc_mol(species=[ARCSpecies(label='S1', smiles='C'),
                                                ARCSpecies(label='S2', smiles='N[CH2]')],
                                       charge=0,
                                       multiplicity=2,
                                       )
        self.assertIsInstance(qcmol2, QCMolecule)
        self.assertEqual(qcmol2.molecular_charge, 0)
        self.assertEqual(qcmol2.molecular_multiplicity, 2)
        for symbol, expected_symbol in zip(qcmol2.symbols, ['C', 'H', 'H', 'H', 'H', 'N', 'C', 'H', 'H', 'H', 'H']):
            self.assertEqual(symbol, expected_symbol)

    def test_check_species_before_mapping(self):
        """Test the check_species_before_mapping function."""
        self.assertFalse(mapping.check_species_before_mapping(spc_1=ARCSpecies(label='CH4', smiles='C'),
                                                              spc_2=ARCSpecies(label='CH3', smiles='[CH3]'),
                                                              verbose=True))
        self.assertFalse(mapping.check_species_before_mapping(spc_1=ARCSpecies(label='CO2', smiles='O=C=O'),
                                                              spc_2=ARCSpecies(label='SCO', smiles='S=C=O'),
                                                              verbose=True))
        self.assertFalse(mapping.check_species_before_mapping(spc_1=ARCSpecies(label='CH4', smiles='C'),
                                                              spc_2=ARCSpecies(label='CH3F', smiles='CF'),
                                                              verbose=True))
        self.assertFalse(mapping.check_species_before_mapping(spc_1=ARCSpecies(label='COC', smiles='COC'),
                                                              spc_2=ARCSpecies(label='CCO', smiles='CCO'),
                                                              verbose=True))
        self.assertFalse(mapping.check_species_before_mapping(spc_1=ARCSpecies(label='linear_C6H12', smiles='C=CCCCC'),
                                                              spc_2=ARCSpecies(label='cyclic_C6H12', smiles='C1CCCCC1'),
                                                              verbose=True))
        self.assertTrue(mapping.check_species_before_mapping(spc_1=ARCSpecies(label='H2O', smiles='O'),
                                                             spc_2=ARCSpecies(label='H2O', smiles='O'),
                                                             verbose=True))
        self.assertTrue(mapping.check_species_before_mapping(spc_1=ARCSpecies(label='nC4H10', smiles='CCCC'),
                                                             spc_2=ARCSpecies(label='iC4H10', smiles='CC(C)C'),
                                                             verbose=True))
        self.assertTrue(mapping.check_species_before_mapping(spc_1=self.spc1, spc_2=self.spc2, verbose=True))

    def test_get_bonds_dict(self):
        """Test the get_bonds_dict function."""
        bond_dict = mapping.get_bonds_dict(spc=ARCSpecies(label='CH4', smiles='C'))
        self.assertEqual(bond_dict, {'C-H': 4})
        bond_dict = mapping.get_bonds_dict(spc=self.ccjco)
        self.assertEqual(bond_dict, {'C-C': 2, 'C-H': 5, 'C-O': 1})
        bond_dict = mapping.get_bonds_dict(spc=ARCSpecies(label='nC4H10', smiles='CCCC'))
        self.assertEqual(bond_dict, {'C-C': 3, 'C-H': 10})
        bond_dict = mapping.get_bonds_dict(spc=ARCSpecies(label='iC4H10', smiles='CC(C)C'))
        self.assertEqual(bond_dict, {'C-C': 3, 'C-H': 10})

    def test_fingerprint(self):
        """Test the fingerprint function."""
        fingerprint = mapping.fingerprint(ARCSpecies(label='CH4', smiles='C'))
        self.assertEqual(fingerprint, {0: {'self': 'C', 'H': [1, 2, 3, 4]}})

        so2_s = ARCSpecies(label='SO2', smiles='O=S=O', multiplicity=1,
                           xyz={'coords': ((-1.3554230894998571, -0.4084942756329785, 0.0),
                                           (-0.04605352293144468, 0.6082507106551855, 0.0),
                                           (1.4014766124312934, -0.19975643502220325, 0.0)),
                                'isotopes': (16, 32, 16), 'symbols': ('O', 'S', 'O')})
        fingerprint = mapping.fingerprint(so2_s)
        self.assertEqual(fingerprint, {0: {'self': 'O', 'S': [1]},
                                       1: {'self': 'S', 'O': [0, 2]},
                                       2: {'self': 'O', 'S': [1]}})

        so2_t = ARCSpecies(label='SO2', smiles='[O][S]=O', multiplicity=3,
                           xyz={'coords': ((0.02724478716956233, 0.6093829407458188, 0.0),
                                           (-1.3946381818031768, -0.24294788636871906, 0.0),
                                           (1.3673933946336125, -0.36643505437710233, 0.0)),
                                'isotopes': (32, 16, 16), 'symbols': ('S', 'O', 'O')})
        fingerprint = mapping.fingerprint(so2_t)
        self.assertIn(fingerprint[0]['O'], [[1, 2], [2, 1]])  # non-deterministic
        for i in [1, 2]:
            self.assertEqual(fingerprint[i], {'self': 'O', 'S': [0]})

        fingerprint = mapping.fingerprint(self.ccjco)
        self.assertEqual(fingerprint, {0: {'self': 'C', 'C': [1], 'H': [5, 6, 7]},
                                       1: {'self': 'C', 'chirality': 'Z', 'C': [0, 3], 'H': [2]},
                                       3: {'self': 'C', 'chirality': 'Z', 'C': [1], 'O': [4], 'H': [8]},
                                       4: {'self': 'O', 'C': [3]}})

        fingerprint = mapping.fingerprint(self.spc1)
        self.assertEqual(fingerprint, self.fingerprint_1)

        fingerprint = mapping.fingerprint(self.spc2)
        self.assertEqual(fingerprint, self.fingerprint_2)

        fingerprint = mapping.fingerprint(self.chiral_spc_1)
        self.assertEqual(fingerprint, {0: {'self': 'C', 'chirality': 'Z', 'C': [2], 'O': [4], 'H': [1]},
                                       2: {'self': 'C', 'chirality': 'Z', 'C': [0, 6], 'H': [3]},
                                       4: {'self': 'O', 'C': [0], 'H': [5]},
                                       6: {'self': 'C', 'chirality': 'S', 'C': [2], 'N': [8], 'S': [11], 'H': [7]},
                                       8: {'self': 'N', 'C': [6], 'H': [9, 10]},
                                       11: {'self': 'S', 'C': [6], 'H': [12]}})

        fingerprint = mapping.fingerprint(self.chiral_spc_2)
        self.assertEqual(fingerprint, {0: {'self': 'C', 'chirality': 'E', 'C': [1], 'O': [10], 'H': [12]},
                                       1: {'self': 'C', 'chirality': 'E', 'C': [0, 3], 'H': [2]},
                                       3: {'self': 'C', 'chirality': 'R', 'C': [1], 'N': [7], 'S': [5], 'H': [4]},
                                       5: {'self': 'S', 'C': [3], 'H': [6]},
                                       7: {'self': 'N', 'C': [3], 'H': [8, 9]},
                                       10: {'self': 'O', 'C': [0], 'H': [11]}})

    def test_identify_superimposable_candidates(self):
        """Test the identify_superimposable_candidates function."""
        candidates = mapping.identify_superimposable_candidates(fingerprint_1=self.fingerprint_1,
                                                                fingerprint_2=self.fingerprint_2)
        self.assertEqual(candidates, [{0: 0, 3: 1, 4: 2, 5: 3, 6: 4}])

    def test_are_adj_elements_in_agreement(self):
        """Test the are_adj_elements_in_agreement function."""
        self.assertFalse(mapping.are_adj_elements_in_agreement({'self': 'C', 'C': [3]},
                                                               {'self': 'C', 'C': [0, 3], 'H': [2]}))
        self.assertFalse(mapping.are_adj_elements_in_agreement({'self': 'C', 'C': [3]},
                                                               {'self': 'C', 'O': [3]}))
        self.assertFalse(mapping.are_adj_elements_in_agreement({'self': 'C', 'C': [3]},
                                                               {'self': 'O', 'C': [3]}))
        self.assertTrue(mapping.are_adj_elements_in_agreement({'self': 'C', 'C': [3]},
                                                              {'self': 'C', 'C': [2]}))
        self.assertTrue(mapping.are_adj_elements_in_agreement({'self': 'C', 'C': [1], 'O': [4], 'H': [8]},
                                                              {'self': 'C', 'C': [1], 'O': [4], 'H': [7]}))
        self.assertTrue(mapping.are_adj_elements_in_agreement({'self': 'C', 'C': [3]},
                                                              {'self': 'C', 'C': [3]}))
        self.assertTrue(mapping.are_adj_elements_in_agreement({'self': 'C', 'C': [1], 'O': [4], 'H': [8]},
                                                              {'self': 'C', 'C': [1], 'O': [4], 'H': [8]}))

    def test_iterative_dfs(self):
        """Test the iterative_dfs function."""
        result = mapping.iterative_dfs(fingerprint_1={0: {'self': 'C', 'C': [1]},
                                                      1: {'self': 'C', 'H': [4, 3]}},
                                       fingerprint_2={0: {'self': 'C', 'C': [1]},
                                                      1: {'self': 'C', 'H': [4, 3]}},
                                       key_1=0,
                                       key_2=0,
                                       )
        self.assertEqual(result, {0: 0, 1: 1})
        result = mapping.iterative_dfs(fingerprint_1=self.fingerprint_1,
                                       fingerprint_2=self.fingerprint_2,
                                       key_1=0,
                                       key_2=0,
                                       )
        self.assertEqual(result, {0: 0, 3: 1, 4: 2, 5: 3, 6: 4})

    def test_prune_identical_dicts(self):
        """Test the prune_identical_dicts function."""
        new_dicts_list = mapping.prune_identical_dicts([{0: 0}])
        self.assertEqual(new_dicts_list, [{0: 0}])
        new_dicts_list = mapping.prune_identical_dicts([{0: 0}, {0: 0}, {0: 0}])
        self.assertEqual(new_dicts_list, [{0: 0}])
        new_dicts_list = mapping.prune_identical_dicts([{0: 0}, {0: 0}, {0: 0}, {0: 1}])
        self.assertEqual(new_dicts_list, [{0: 0}, {0: 1}])
        new_dicts_list = mapping.prune_identical_dicts([{0: 0, 3: 1, 4: 2, 5: 3, 6: 4},
                                                        {0: 0, 3: 1, 4: 2, 5: 3, 6: 4},
                                                        {0: 0, 3: 1, 4: 2, 5: 3, 6: 4},
                                                        {0: 0, 3: 1, 4: 2, 5: 3, 6: 4},
                                                        {0: 0, 3: 1, 4: 2, 5: 3, 6: 4}])
        self.assertEqual(new_dicts_list, [{0: 0, 3: 1, 4: 2, 5: 3, 6: 4}])

    def test_remove_gaps_from_values(self):
        """Test the remove_gaps_from_values function."""
        self.assertEqual(mapping.remove_gaps_from_values({5: 18, 7: 502, 21: 0, 0: 55, 2: 1}),
                         {5: 2, 7: 4, 21: 0, 0: 3, 2: 1})

    def test_fix_dihedrals_by_backbone_mapping(self):
        """Test the fix_dihedrals_by_backbone_mapping function."""
        self.spc1.determine_rotors()
        self.spc2.determine_rotors()
        atom_map = {0: 0, 1: 6, 2: 5, 3: 1, 4: 2, 5: 3, 6: 4, 7: 7, 8: 8, 9: 10, 10: 9, 11: 11, 12: 12, 13: 13}
        original_dihedrals_1 = [calculate_dihedral_angle(coords=self.spc1.get_xyz(),
                                                         torsion=rotor_dict['torsion'])
                                for rotor_dict in self.spc1.rotors_dict.values()]
        original_dihedrals_2 = [calculate_dihedral_angle(coords=self.spc2.get_xyz(),
                                                         torsion=[atom_map[t] for t in rotor_dict['torsion']])
                                for rotor_dict in self.spc1.rotors_dict.values()]
        spc1, spc2 = mapping.fix_dihedrals_by_backbone_mapping(spc_1=self.spc1, spc_2=self.spc2, backbone_map=atom_map)
        new_dihedrals_1 = [calculate_dihedral_angle(coords=spc1.get_xyz(),
                                                    torsion=rotor_dict['torsion'])
                           for rotor_dict in spc1.rotors_dict.values()]
        new_dihedrals_2 = [calculate_dihedral_angle(coords=spc2.get_xyz(),
                                                    torsion=[atom_map[t] for t in rotor_dict['torsion']])
                           for rotor_dict in spc1.rotors_dict.values()]
        self.assertAlmostEqual(original_dihedrals_1[2], 67.81049913527622)
        self.assertAlmostEqual(original_dihedrals_2[2], 174.65228274664804)
        self.assertAlmostEqual(new_dihedrals_1[2], 121.23139159126627)
        self.assertAlmostEqual(new_dihedrals_2[2], 121.23139016907017)

    def test_get_backbone_dihedral_deviation_score(self):
        """Test the get_backbone_dihedral_deviation_score function."""
        self.spc1.determine_rotors()
        self.spc2.determine_rotors()
        fingerprint_1, fingerprint_2 = mapping.fingerprint(self.spc1), mapping.fingerprint(self.spc2)
        backbone_map = mapping.identify_superimposable_candidates(fingerprint_1, fingerprint_2)[0]
        score = mapping.get_backbone_dihedral_deviation_score(spc_1=self.spc1, spc_2=self.spc2, backbone_map=backbone_map)
        self.assertAlmostEqual(score, 106.8417836)

    def test_get_backbone_dihedral_angles(self):
        """Test the get_backbone_dihedral_angles function."""
        self.spc1.determine_rotors()
        self.spc2.determine_rotors()
        torsions = mapping.get_backbone_dihedral_angles(self.spc1, self.spc2, backbone_map={0: 0, 3: 1, 5: 3, 6: 4, 4: 2})
        self.assertEqual(torsions, [{'torsion 1': [0, 3, 5, 6],
                                     'torsion 2': [0, 1, 3, 4],
                                     'angle 1': 67.81049913527622,
                                     'angle 2': 174.65228274664804}])

    def test_map_lists(self):
        """Test the map_lists function."""
        self.assertEqual(mapping.map_lists([], []), {})
        self.assertEqual(mapping.map_lists([0], [0]), {0: 0})
        self.assertEqual(mapping.map_lists([120.5, 80.7, 345.9], [90.2, 355.0, 111.1]), {0: 2, 1: 0, 2: 1})
        self.assertEqual(mapping.map_lists([179.9, 4.18e-06], [180.8, 359.7]), {0: 0, 1: 1})
        with self.assertRaises(ValueError):
            mapping.map_lists([5.0], [3.2, 7.9])

    def test_map_hydrogens(self):
        """Test the map_hydrogens function."""
        # Todo: Add tests with many types of torsions and non(pseudo)-torsions, multiple bonds, cyclics.
        # CH4 different order
        spc1 = ARCSpecies(label='CH4', smiles='C', xyz=self.ch4_xyz)
        spc2 = ARCSpecies(label='CH4', smiles='C', xyz=self.ch4_xyz_diff_order)
        atom_map = mapping.map_hydrogens(spc1, spc2, {0: 2})
        self.assertEqual(atom_map, {0: 2, 1: 0, 2: 1, 3: 3, 4: 4})

        # One inner torsion, several terminal torsions.
        self.spc1.determine_rotors()
        self.spc2.determine_rotors()
        fingerprint_1, fingerprint_2 = mapping.fingerprint(self.spc1), mapping.fingerprint(self.spc2)
        backbone_map = mapping.identify_superimposable_candidates(fingerprint_1, fingerprint_2)[0]
        self.assertEqual(backbone_map, {0: 0, 3: 1, 5: 3, 6: 4, 4: 2})
        atom_map = mapping.map_hydrogens(self.spc1, self.spc2, backbone_map)
        self.assertEqual(atom_map,
                         {0: 0, 1: 6, 2: 5, 3: 1, 4: 2, 5: 3, 6: 4, 7: 7, 8: 8, 9: 10, 10: 9, 11: 11, 12: 12, 13: 13})


if __name__ == '__main__':
    unittest.main(testRunner=unittest.TextTestRunner(verbosity=2))
