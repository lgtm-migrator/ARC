"""
A module for atom-mapping a species or a set of species.

Species atom-map logic:
1. Determine adjacent elements for each heavy atom
2. Identify and loop superposition possibilities
3. Recursively modify dihedrals until the structures overlap to some tolerance
4. Determine RMSD to backbone, if good then determine RMSD to H's
5. When mapping H's on terminal heavy atoms, check whether rotating this rotor will reduce the overall RMSD
   if there's more than one H on that terminal atom
"""

from collections import deque
from itertools import product
from typing import TYPE_CHECKING, Dict, List, Optional, Tuple, Union

from qcelemental.exceptions import ValidationError
from qcelemental.models.molecule import Molecule as QCMolecule

from rmgpy.molecule import Molecule
from rmgpy.species import Species

import arc.rmgdb as rmgdb
from arc.common import convert_list_index_0_to_1, logger
from arc.species import ARCSpecies
from arc.species.converter import compare_confs, sort_xyz_using_indices, translate_xyz, xyz_from_data, xyz_to_str
from arc.species.vectors import calculate_dihedral_angle


if TYPE_CHECKING:
    from rmgpy.data.kinetics.family import TemplateReaction
    from rmgpy.data.rmg import RMGDatabase
    from rmgpy.reaction import Reaction
    from arc.reaction import ARCReaction


def map_reaction(rxn: 'ARCReaction',
                 db: Optional['RMGDatabase'] = None,
                 ) -> Optional[List[int]]:
    """
    Map a reaction.

    Args:
        rxn (ARCReaction): An ARCReaction object instance.
        db (RMGDatabase, optional): The RMG database instance.

    Returns:
        Optional[List[int]]:
            Entry indices are running atom indices of the reactants,
            corresponding entry values are running atom indices of the products.
    """
    if rxn.family is None:
        rmgdb.determine_family(reaction=rxn, db=db)
    if rxn.family is None:
        return map_general_rxn(rxn)

    fam_func_dict = {'H_Abstraction': map_h_abstraction,
                     'HO2_Elimination_from_PeroxyRadical': map_ho2_elimination_from_peroxy_radical,
                     'intra_H_migration': map_intra_h_migration,
                     }

    if rxn.family.label not in fam_func_dict.keys():
        logger.info(f'Using a generic mapping algorithm for {rxn} of family {rxn.family.label}')

    map_func = fam_func_dict.get(rxn.family.label, map_general_rxn)

    return map_func(rxn, db)


def map_general_rxn(rxn: 'ARCReaction',
                    db: Optional['RMGDatabase'] = None,
                    ) -> Optional[List[int]]:
    """
    Map a general reaction (one that was not categorized into a reaction family by RMG).
    The general method isn't great, a family-specific method should be implemented where possible.

    Args:
        rxn (ARCReaction): An ARCReaction object instance.
        db (RMGDatabase, optional): The RMG database instance.

    Returns:
        Optional[List[int]]:
            Entry indices are running atom indices of the reactants,
            corresponding entry values are running atom indices of the products.
    """
    if rxn.is_isomerization():
        return map_two_species(rxn.r_species[0], rxn.p_species[0], map_type='list')

    qcmol_1 = create_qc_mol(species=[spc.copy() for spc in rxn.r_species],
                            charge=rxn.charge,
                            multiplicity=rxn.multiplicity,
                            )
    qcmol_2 = create_qc_mol(species=[spc.copy() for spc in rxn.p_species],
                            charge=rxn.charge,
                            multiplicity=rxn.multiplicity,
                            )
    if qcmol_1 is None or qcmol_2 is None:
        return None
    data = qcmol_2.align(ref_mol=qcmol_1, verbose=0)[1]
    atom_map = data['mill'].atommap.tolist()
    return atom_map


# Family-specific mapping functions:


def map_h_abstraction(rxn: 'ARCReaction',
                      db: Optional['RMGDatabase'] = None,
                      ) -> Optional[List[int]]:
    """
    Map a hydrogen abstraction reaction.
    Strategy: Map species R(*1)-H(*2) to species R(*1)j and map species R(*3)j to species R(*3)-H(*2).
    Use scissors to map the backbone.

    Args:
        rxn (ARCReaction): An ARCReaction object instance that belongs to the RMG H_Abstraction reaction family.
        db (RMGDatabase, optional): The RMG database instance.

    Returns:
        Optional[List[int]]:
            Entry indices are running atom indices of the reactants,
            corresponding entry values are running atom indices of the products.
    """
    if not check_family_for_mapping_function(rxn=rxn, db=db, family='H_Abstraction'):
        return None

    rmg_reactions = _get_rmg_reactions_from_arc_reaction(arc_reaction=rxn)
    r_label_dict, p_label_dict = get_atom_indices_of_labeled_atoms_in_an_rmg_reaction(arc_reaction=rxn,
                                                                                      rmg_reaction=rmg_reactions[0])
    r_h_index = r_label_dict['*2']
    p_h_index = p_label_dict['*2']
    len_r1, len_p1 = rxn.r_species[0].number_of_atoms, rxn.p_species[0].number_of_atoms
    r1_h2 = 0 if r_h_index < len_r1 else 1  # Identify R(*1)-H(*2), it's either reactant 0 or reactant 1.
    r3 = 1 - r1_h2  # Identify R(*3) in the reactants.
    r3_h2 = 0 if p_h_index < len_p1 else 1  # Identify R(*3)-H(*2), it's either product 0 or product 1.
    r1 = 1 - r3_h2  # Identify R(*1) in the products.

    spc_r1_h2 = ARCSpecies(label='R1-H2',
                           mol=rxn.r_species[r1_h2].mol.copy(deep=True),
                           xyz=rxn.r_species[r1_h2].get_xyz(),
                           bdes=[(r_label_dict['*1'] + 1 - r1_h2 * len_r1,
                                  r_label_dict['*2'] + 1 - r1_h2 * len_r1)],  # Mark the R(*1)-H(*2) bond for scission.
                           )
    spc_r1_h2.final_xyz = spc_r1_h2.get_xyz()  # Scissors require the .final_xyz attribute to be populated.
    spc_r1_h2_cuts = spc_r1_h2.scissors()
    spc_r1_h2_cut = [spc for spc in spc_r1_h2_cuts if spc.label != 'H'][0] \
        if any(spc.label != 'H' for spc in spc_r1_h2_cuts) else spc_r1_h2_cuts[0]  # Treat H2 as well :)
    spc_r3_h2 = ARCSpecies(label='R3-H2',
                           mol=rxn.p_species[r3_h2].mol.copy(deep=True),
                           xyz=rxn.p_species[r3_h2].get_xyz(),
                           bdes=[(p_label_dict['*3'] + 1 - r3_h2 * len_p1,
                                  p_label_dict['*2'] + 1 - r3_h2 * len_p1)],  # Mark the R(*3)-H(*2) bond for scission.
                           )
    spc_r3_h2.final_xyz = spc_r3_h2.get_xyz()  # Scissors require the .final_xyz attribute to be populated.
    spc_r3_h2_cuts = spc_r3_h2.scissors()
    spc_r3_h2_cut = [spc for spc in spc_r3_h2_cuts if spc.label != 'H'][0] \
        if any(spc.label != 'H' for spc in spc_r3_h2_cuts) else spc_r3_h2_cuts[0]  # Treat H2 as well :)
    map_1 = map_two_species(spc_r1_h2_cut, rxn.p_species[r1])
    map_2 = map_two_species(rxn.r_species[r3], spc_r3_h2_cut)

    result = {r_h_index: p_h_index}
    for r_increment, p_increment, map_i in zip([r1_h2 * len_r1, (1 - r1_h2) * len_r1],
                                              [(1 - r3_h2) * len_p1, r3_h2 * len_p1],
                                              [map_1, map_2]):
        for i, entry in enumerate(map_i):
            r_index = i + r_increment + int(i + r_increment >= r_h_index)
            p_index = entry + p_increment
            result[r_index] = p_index
    return [val for key, val in sorted(result.items(), key=lambda item: item[0])]


def map_ho2_elimination_from_peroxy_radical(rxn: 'ARCReaction',
                                            db: Optional['RMGDatabase'] = None,
                                            ) -> Optional[List[int]]:
    """
    Map an HO2 elimination from peroxy radical reaction.
    Strategy: Remove the O(*3), O(*4), and H(*5) atoms from the reactant and map to the R(*1)=R(*2) product.
    Note that two consecutive scissions must be performed.

    Args:
        rxn (ARCReaction): An ARCReaction object instance that belongs to the RMG H_Abstraction reaction family.
        db (RMGDatabase, optional): The RMG database instance.

    Returns:
        Optional[List[int]]:
            Entry indices are running atom indices of the reactants,
            corresponding entry values are running atom indices of the products.
    """
    if not check_family_for_mapping_function(rxn=rxn, db=db, family='HO2_Elimination_from_PeroxyRadical'):
        return None

    rmg_reactions = _get_rmg_reactions_from_arc_reaction(arc_reaction=rxn)
    r_label_dict, p_label_dict = get_atom_indices_of_labeled_atoms_in_an_rmg_reaction(arc_reaction=rxn,
                                                                                      rmg_reaction=rmg_reactions[0])

    # Todo:
    # reverse = False
    # if len(rxn.p_species) == 1 and len(rxn.r_species) == 2:
    #     reverse = True
    #     r_label_dict, p_label_dict = p_label_dict, r_label_dict

    if len(rxn.r_species) == 1 and len(rxn.p_species) == 2:
        r_o3_index = r_label_dict['*3']
        r_o4_index = r_label_dict['*4']
        r_h5_index = r_label_dict['*5']
        len_p1 = rxn.p_species[0].number_of_atoms
        r1dr2 = 0 if p_label_dict['*1'] < len_p1 else 1  # Identify R(*1)=R(*2), it's either product 0 or product 1.

        mol_r_mod = rxn.r_species[0].mol.copy(deep=True)
        xyz_r_mod = rxn.r_species[0].get_xyz()
        vertex_indices = sorted([r_o3_index, r_o4_index, r_h5_index], reverse=True)
        for vertex_index in vertex_indices:
            mol_r_mod.vertices.remove(mol_r_mod.vertices[vertex_index])
        xyz_r_mod['symbols'] = tuple(symbol for i, symbol in enumerate(xyz_r_mod['symbols']) if i not in vertex_indices)
        xyz_r_mod['isotopes'] = tuple(isotope for i, isotope in enumerate(xyz_r_mod['isotopes']) if i not in vertex_indices)
        xyz_r_mod['coords'] = tuple(coord for i, coord in enumerate(xyz_r_mod['coords']) if i not in vertex_indices)
        spc_r_mod = ARCSpecies(label='R', mol=mol_r_mod, xyz=xyz_r_mod)
        spc_r_mod.final_xyz = xyz_r_mod  # .set_dihedral() requires the .final_xyz attribute.

        # Different dihedral angles in the reactant and product will make mapping H atoms hard.
        # Fix dihedrals between 4 heavy atom sequences.
        spc_r_mod.determine_rotors()
        map_1 = map_two_species(spc_r_mod, rxn.p_species[r1dr2])
        for rotor in spc_r_mod.rotors_dict.values():
            torsion = rotor['torsion']
            if not spc_r_mod.mol.atoms[torsion[0]].is_hydrogen() and not spc_r_mod.mol.atoms[torsion[1]].is_hydrogen():
                spc_r_mod.set_dihedral(scan=convert_list_index_0_to_1(torsion),
                                       deg_abs=calculate_dihedral_angle(coords=rxn.p_species[r1dr2].get_xyz(),
                                                                        torsion=[map_1[t] for t in torsion]),
                                       chk_rotor_list=False)
                spc_r_mod.final_xyz = spc_r_mod.initial_xyz
        map_2 = map_two_species(spc_r_mod, rxn.p_species[r1dr2])
        new_map, added_ho2_atoms = list(), list()
        star_map = {r_o3_index: '*3', r_o4_index: '*4', r_h5_index: '*5'}
        for i, entry in enumerate(map_2):
            for j in [i, i + 1, i + 2]:
                # Check three consecutive indices, we don't know whether the HO2 atoms are mapped consecutively or not.
                if j in star_map.keys() and j not in added_ho2_atoms:
                    new_map.append(p_label_dict[star_map[j]])
                    added_ho2_atoms.append(j)
                else:
                    break
            new_map.append(entry)
        return new_map


def map_intra_h_migration(rxn: 'ARCReaction',
                          db: Optional['RMGDatabase'] = None,
                          ) -> Optional[List[int]]:
    """
    Map an intra hydrogen migration reaction.
    Strategy: Remove the *3 H atom from both the reactant and product to have the same backbone.
    Map the backbone and add the (known) *3 H atom.

    Args:
        rxn (ARCReaction): An ARCReaction object instance that belongs to the RMG H_Abstraction reaction family.
        db (RMGDatabase, optional): The RMG database instance.

    Returns:
        Optional[List[int]]:
            Entry indices are running atom indices of the reactants,
            corresponding entry values are running atom indices of the products.
    """
    if not check_family_for_mapping_function(rxn=rxn, db=db, family='intra_H_migration'):
        return None

    rmg_reactions = _get_rmg_reactions_from_arc_reaction(arc_reaction=rxn)
    r_label_dict, p_label_dict = get_atom_indices_of_labeled_atoms_in_an_rmg_reaction(arc_reaction=rxn,
                                                                                      rmg_reaction=rmg_reactions[0])

    r_h_index = r_label_dict['*3']
    p_h_index = p_label_dict['*3']

    spc_r = ARCSpecies(label='R',
                       mol=rxn.r_species[0].mol.copy(deep=True),
                       xyz=rxn.r_species[0].get_xyz(),
                       bdes=[(r_label_dict['*2'] + 1, r_label_dict['*3'] + 1)],  # Mark the R(*2)-H(*3) bond for scission.
                       )
    spc_r.final_xyz = spc_r.get_xyz()  # Scissors require the .final_xyz attribute to be populated.
    spc_r_dot = [spc for spc in spc_r.scissors() if spc.label != 'H'][0]
    spc_p = ARCSpecies(label='P',
                       mol=rxn.p_species[0].mol.copy(deep=True),
                       xyz=rxn.p_species[0].get_xyz(),
                       bdes=[(p_label_dict['*1'] + 1, p_label_dict['*3'] + 1)],  # Mark the R(*1)-H(*3) bond for scission.
                       )
    spc_p.final_xyz = spc_p.get_xyz()  # Scissors require the .final_xyz attribute to be populated.
    spc_p_dot = [spc for spc in spc_p.scissors() if spc.label != 'H'][0]
    map_ = map_two_species(spc_r_dot, spc_p_dot)

    new_map = list()
    for i, entry in enumerate(map_):
        if i == r_h_index:
            new_map.append(p_h_index)
        new_map.append(entry if entry < p_h_index else entry + 1)
    return new_map


# Mapping functions:


def check_family_for_mapping_function(rxn: 'ARCReaction',
                                      family: str,
                                      db: Optional['RMGDatabase'] = None,
                                      ) -> bool:
    """
    Check that the actual reaction family and the desired reaction family are the same.

    Args:
        rxn (ARCReaction): An ARCReaction object instance.
        family (str): The desired reaction family to check for.
        db (RMGDatabase, optional): The RMG database instance.

    Returns:
        bool: Whether the reaction family and the desired ``family`` are consistent.
    """
    if rxn.family is None:
        rmgdb.determine_family(reaction=rxn, db=db)
    if rxn.family is None or rxn.family.label != family:
        return False
    return True


def get_atom_indices_of_labeled_atoms_in_an_rmg_reaction(arc_reaction: 'ARCReaction',
                                                         rmg_reaction: 'TemplateReaction',
                                                         ) -> Tuple[Optional[Dict[str, int]], Optional[Dict[str, int]]]:
    """
    Get the RMG reaction atom labels and the corresponding 0-indexed atom indices
    for all labeled atoms in a TemplateReaction.

    Args:
        arc_reaction (ARCReaction): An ARCReaction object instance.
        rmg_reaction (TemplateReaction): A respective RMG family TemplateReaction object instance.

    Returns:
        Tuple[Optional[Dict[str, int]], Optional[Dict[str, int]]]:
            The tuple entries relate to reactants and products.
            Keys are labels (e.g., '*1'), values are corresponding 0-indices atoms.
    """
    if not hasattr(rmg_reaction, 'labeled_atoms') or not rmg_reaction.labeled_atoms:
        return None, None

    for mol in rmg_reaction.reactants + rmg_reaction.products:
        mol.generate_resonance_structures(save_order=True)

    r_map, p_map = map_arc_rmg_species(arc_reaction=arc_reaction, rmg_reaction=rmg_reaction)

    reactant_index_dict, product_index_dict = dict(), dict()
    reactant_atoms, product_atoms = list(), list()
    rmg_reactant_order = [val[0] for key, val in sorted(r_map.items(), key=lambda item: item[0])]
    rmg_product_order = [val[0] for key, val in sorted(p_map.items(), key=lambda item: item[0])]
    for i in rmg_reactant_order:
        reactant_atoms.extend([atom for atom in rmg_reaction.reactants[i].atoms])
    for i in rmg_product_order:
        product_atoms.extend([atom for atom in rmg_reaction.products[i].atoms])

    for labeled_atom_dict, atom_list, index_dict in zip([rmg_reaction.labeled_atoms['reactants'],
                                                         rmg_reaction.labeled_atoms['products']],
                                                        [reactant_atoms, product_atoms],
                                                        [reactant_index_dict, product_index_dict]):
        for label, atom_1 in labeled_atom_dict.items():
            for i, atom_2 in enumerate(atom_list):
                if atom_1.id == atom_2.id:
                    index_dict[label] = i
                    break
    return reactant_index_dict, product_index_dict


def map_arc_rmg_species(arc_reaction: 'ARCReaction',
                        rmg_reaction: Union['Reaction', 'TemplateReaction'],
                        concatenate: bool = True,
                        ) -> Tuple[Dict[int, Union[List[int], int]], Dict[int, Union[List[int], int]]]:
    """
    Map the species pairs in an ARC reaction to those in a respective RMG reaction
    which is defined in the same direction.

    Args:
        arc_reaction (ARCReaction): An ARCReaction object instance.
        rmg_reaction (Union[Reaction, TemplateReaction]): A respective RMG family TemplateReaction object instance.
        concatenate (bool, optional): Whether to return isomorphic species as a single list (``True``, default),
                                      or to return isomorphic species separately (``False``).

    Returns:
        Tuple[Dict[int, Union[List[int], int]], Dict[int, Union[List[int], int]]]:
            The first tuple entry refers to reactants, the second to products.
            Keys are specie indices in the ARC reaction,
            values are respective indices in the RMG reaction.
            If ``concatenate`` is ``True``, values are lists of integers. Otherwise, values are integers.
    """
    if rmg_reaction.is_isomerization():
        if concatenate:
            return {0: [0]}, {0: [0]}
        else:
            return {0: 0}, {0: 0}
    r_map, p_map = dict(), dict()
    arc_reactants, arc_products = arc_reaction.get_reactants_and_products(arc=True)
    for spc_map, rmg_species, arc_species in [(r_map, rmg_reaction.reactants, arc_reactants),
                                              (p_map, rmg_reaction.products, arc_products)]:
        for i, arc_spc in enumerate(arc_species):
            for j, rmg_obj in enumerate(rmg_species):
                if isinstance(rmg_obj, Molecule):
                    rmg_spc = Species(molecule=[rmg_obj])
                elif isinstance(rmg_obj, Species):
                    rmg_spc = rmg_obj
                else:
                    raise ValueError(f'Expected an RMG object instance of Molecule() or Species(),'
                                     f'got {rmg_obj} which is a {type(rmg_obj)}.')
                rmg_spc.generate_resonance_structures(save_order=True)
                if rmg_spc.is_isomorphic(arc_spc.mol, save_order=True):
                    if i in spc_map.keys() and concatenate:  # ** Todo: test
                        spc_map[i].append(j)
                    elif concatenate:
                        spc_map[i] = [j]
                    else:
                        spc_map[i] = j
                        break
    return r_map, p_map


def find_equivalent_atoms_in_reactants(arc_reaction: 'ARCReaction') -> Optional[List[List[int]]]:
    """
    Find atom indices that are equivalent in the reactants of an ARCReaction
    in the sense that they represent degenerate reaction sites that are indifferentiable in 2D.
    Bridges between RMG reaction templates and ARC's 3D TS structures.
    Running indices in the returned structure relate to reactant_0 + reactant_1 + ...

    Args:
        arc_reaction ('ARCReaction'): The ARCReaction object instance.

    Returns:
        Optional[List[List[int]]]: Entries are lists of 0-indices, each such list represents equivalent atoms.
    """
    rmg_reactions = _get_rmg_reactions_from_arc_reaction(arc_reaction)
    dicts = [get_atom_indices_of_labeled_atoms_in_an_rmg_reaction(rmg_reaction=rmg_reaction,
                                                                  arc_reaction=arc_reaction)[0]
             for rmg_reaction in rmg_reactions]
    equivalence_map = dict()
    for index_dict in dicts:
        for key, value in index_dict.items():
            if key in equivalence_map:
                equivalence_map[key].append(value)
            else:
                equivalence_map[key] = [value]
    equivalent_indices = list(list(set(equivalent_list)) for equivalent_list in equivalence_map.values())
    return equivalent_indices


def _get_rmg_reactions_from_arc_reaction(arc_reaction: 'ARCReaction',
                                         db: Optional['RMGDatabase'] = None,
                                         ) -> Optional[List['TemplateReaction']]:
    """
    A helper function for getting RMG reactions from an ARC reaction.

    Args:
        arc_reaction (ARCReaction): The ARCReaction object instance.
        db (RMGDatabase, optional): The RMG database instance.

    Returns:
        Optional[List[TemplateReaction]]:
            The respective RMG TemplateReaction object instances (considering resonance structures).
    """
    if arc_reaction.family is None:
        rmgdb.determine_family(reaction=arc_reaction, db=db)
    if arc_reaction.family is None:
        return None
    rmg_reactions = arc_reaction.family.generate_reactions(reactants=[spc.mol for spc in arc_reaction.r_species],
                                                           products=[spc.mol for spc in arc_reaction.p_species],
                                                           prod_resonance=True,
                                                           delete_labels=False,
                                                           relabel_atoms=False,
                                                           )
    for rmg_reaction in rmg_reactions:
        r_map, p_map = map_arc_rmg_species(arc_reaction=arc_reaction, rmg_reaction=rmg_reaction, concatenate=False)
        ordered_rmg_reactants = [rmg_reaction.reactants[r_map[i]] for i in range(len(rmg_reaction.reactants))]
        ordered_rmg_products = [rmg_reaction.products[p_map[i]] for i in range(len(rmg_reaction.products))]
        mapped_rmg_reactants, mapped_rmg_products = list(), list()
        for ordered_rmg_mols, arc_species, mapped_mols in zip([ordered_rmg_reactants, ordered_rmg_products],
                                                              [arc_reaction.r_species, arc_reaction.p_species],
                                                              [mapped_rmg_reactants, mapped_rmg_products],
                                                              ):
            for rmg_mol, arc_spc in zip(ordered_rmg_mols, arc_species):
                mol = arc_spc.copy().mol
                atom_map = map_two_species(mol, rmg_mol, map_type='dict')
                new_atoms_list = list()
                for i in range(len(rmg_mol.atoms)):
                    rmg_mol.atoms[atom_map[i]].id = mol.atoms[i].id
                    new_atoms_list.append(rmg_mol.atoms[atom_map[i]])
                rmg_mol.atoms = new_atoms_list
                mapped_mols.append(rmg_mol)
        rmg_reaction.reactants, rmg_reaction.products = mapped_rmg_reactants, mapped_rmg_products
    return rmg_reactions


def map_two_species(spc_1: Union[ARCSpecies, Species, Molecule],
                    spc_2: Union[ARCSpecies, Species, Molecule],
                    map_type: str = 'list',
                    verbose: bool = False,
                    backend: str = 'rmsd',
                    ) -> Optional[Union[List[int], Dict[int, int]]]:
    """
    Map the atoms in spc1 to the atoms in spc2.
    All indices are 0-indexed.
    If a dict type atom map is returned, it cold conveniently be used to map ``spc_2`` -> ``spc_1`` by doing::

        ordered_spc1.atoms = [spc_2.atoms[atom_map[i]] for i in range(len(spc_2.atoms))]

    Args:
        spc_1 (Union[ARCSpecies, Species, Molecule]): Species 1.
        spc_2 (Union[ARCSpecies, Species, Molecule]): Species 2.
        map_type (str, optional): Whether to return a 'list' or a 'dict' map type.
        verbose (bool, optional): Whether to use logging.
        backend (str, optional): Whether to use 'QCElemental' or ARC's RMSD method as the backend.

    Returns:
        Optional[Union[List[int], Dict[int, int]]]:
            The atom map. By default, a list is returned.
            If the map is of ``list`` type, entry indices are atom indices of ``spc_1``, entry values are atom indices of ``spc_2``.
            If the map is of ``dict`` type, keys are atom indices of ``spc_1``, values are atom indices of ``spc_2``.
    """
    if backend not in ['QCElemental', 'RMSD']:
        raise ValueError(f'The backend method could be either "QCElemental" or "RMSD", got {backend}.')
    if backend == 'RMSD':
        spc_1, spc_2 = get_arc_species(spc_1), get_arc_species(spc_2)
        if not check_species_before_mapping(spc_1, spc_2, verbose):
            return None
        adj_element_dict_1, adj_element_dict_2 = determine_adjacent_elements(spc_1), determine_adjacent_elements(spc_2)
        candidates = identify_superimposable_candidates(adj_element_dict_1, adj_element_dict_2)
        if not len(candidates):
            backend = 'QCElemental'
        else:
            rmsds, fixed_spcs = list(), list()
            for candidate in candidates:
                fixed_spc_1, fixed_spc_2 = fix_dihedrals_by_backbone_mapping(spc_1, spc_2, backbone_map=candidate)
                fixed_spcs.append((fixed_spc_1, fixed_spc_2))
                backbone_1, backbone_2 = set(list(candidate.keys())), set(list(candidate.values()))
                xyz1, xyz2 = fixed_spc_1.get_xyz(), fixed_spc_2.get_xyz()
                for xyz, spc, backbone in zip([xyz1, xyz2], [fixed_spc_1, fixed_spc_2], [backbone_1, backbone_2]):
                    xyz = xyz_from_data(coords=[xyz1['coords'][i] for i in range(spc.number_of_atoms) if i in [backbone]],
                                        symbols=[xyz1['symbols'][i] for i in range(spc.number_of_atoms) if i in [backbone]],
                                        isotopes=[xyz1['isotopes'][i] for i in range(spc.number_of_atoms) if i in [backbone]])
                xyz2 = sort_xyz_using_indices(xyz2, indices=[v for k, v in sorted(candidate.items(), key=lambda item: item[0])])
                rmsds.append(compare_confs(xyz1=xyz1, xyz2=xyz2, rmsd_score=True))
            chosen_candidate_index = rmsds.index(min(rmsds))
            fixed_spc_1, fixed_spc_2 = fixed_spcs[chosen_candidate_index]
            atom_map = map_hydrogens(fixed_spc_1, fixed_spc_2, candidate)
            if map_type == 'list':
                atom_map = [v for k, v in sorted(atom_map.items(), key=lambda item: item[0])]
    if backend == 'QCElemental':
        qcmol_1 = create_qc_mol(species=spc_1.copy())
        qcmol_2 = create_qc_mol(species=spc_2.copy())
        if qcmol_1 is None or qcmol_2 is None:
            return None
        if len(qcmol_1.symbols) != len(qcmol_2.symbols):
            raise ValueError(f'The number of atoms in spc1 ({spc_1.number_of_atoms}) must be equal '
                             f'to the number of atoms in spc1 ({spc_2.number_of_atoms}).')
        data = qcmol_2.align(ref_mol=qcmol_1, verbose=0, uno_cutoff=0.01)
        atom_map = data[1]['mill'].atommap.tolist()
        if map_type == 'dict':
            atom_map = {key: val for key, val in enumerate(atom_map)}
    return atom_map


def get_arc_species(spc: Union[ARCSpecies, Species, Molecule]) -> ARCSpecies:
    """
    Convert an object to an ARCSpecies object.

    Args:
        spc (Union[ARCSpecies, Species, Molecule]): An input object.

    Returns:
        ARCSpecies: THe corresponding ARCSpecies object.
    """
    if isinstance(spc, ARCSpecies):
        return spc
    if isinstance(spc, Species):
        return ARCSpecies(label='S', mol=spc.molecule[0])
    if isinstance(spc, Molecule):
        return ARCSpecies(label='S', mol=spc)
    raise ValueError(f'Species entries may only be ARCSpecies, RMG Species, or RMG Molecule, '
                     f'got {spc} which is a {type(spc)}.')


def create_qc_mol(species: Union[ARCSpecies, Species, Molecule, List[Union[ARCSpecies, Species, Molecule]]],
                  charge: Optional[int] = None,
                  multiplicity: Optional[int] = None,
                  ) -> Optional[QCMolecule]:
    """
    Create a single QCMolecule object instance from a ARCSpecies object instances.

    Args:
        species (List[Union[ARCSpecies, Species, Molecule]]): Entries are ARCSpecies / RMG Species / RMG Molecule
                                                              object instances.
        charge (int, optional): The overall charge of the surface.
        multiplicity (int, optional): The overall electron multiplicity of the surface.

    Returns:
        Optional[QCMolecule]: The respective QCMolecule object instance.
    """
    species = [species] if not isinstance(species, list) else species
    species_list = list()
    for spc in species:
        species_list.append(get_arc_species(spc))
    if len(species_list) == 1:
        if charge is None:
            charge = species_list[0].charge
        if multiplicity is None:
            multiplicity = species_list[0].multiplicity
    if charge is None or multiplicity is None:
        raise ValueError(f'An overall charge and multiplicity must be specified for multiple species, '
                         f'got: {charge} and {multiplicity}, respectively')
    radius = max([spc.radius for spc in species_list]) if len(species_list) > 1 else 0
    qcmol = None
    data = '\n--\n'.join([xyz_to_str(translate_xyz(spc.get_xyz(), translation=(i * radius, 0, 0)))
                          for i, spc in enumerate(species_list)]) \
        if len(species_list) > 1 else xyz_to_str(species_list[0].get_xyz())
    try:
        qcmol = QCMolecule.from_data(
            data=data,
            molecular_charge=charge,
            molecular_multiplicity=multiplicity,
            fragment_charges=[spc.charge for spc in species_list],
            fragment_multiplicities=[spc.multiplicity for spc in species_list],
            orient=False,
        )
    except ValidationError as err:
        logger.warning(f'Could not get atom map for {[spc.label for spc in species_list]}, got:\n{err}')
    return qcmol


def check_species_before_mapping(spc_1: ARCSpecies,
                                 spc_2: ARCSpecies,
                                 verbose: bool = False,
                                 ) -> bool:
    """
    Perform general checks before mapping two species.

    Args:
        spc_1 (ARCSpecies): Species 1.
        spc_2 (ARCSpecies): Species 2.
        verbose (bool, optional): Whether to use logging.

    Returns:
        bool: ``True`` if all checks passed, ``False`` otherwise.
    """
    # Check number of atoms > 0.
    if spc_1.number_of_atoms == 0 or spc_2.number_of_atoms == 0:
        if verbose:
            logger.warning(f'The number of atoms must be larger than 0, '
                           f'got {spc_1.number_of_atoms} and {spc_2.number_of_atoms}.')
        return False
    # Check the same number of atoms.
    if spc_1.number_of_atoms != spc_2.number_of_atoms:
        if verbose:
            logger.warning(f'The number of atoms must be identical between the two species, '
                           f'got {spc_1.number_of_atoms} and {spc_2.number_of_atoms}.')
        return False
    # Check the same number of each element.
    element_dict_1, element_dict_2 = dict(), dict()
    for atom in spc_1.mol.vertices:
        element_dict_1[atom.element.symbol] = element_dict_1.get(atom.element.symbol, 0) + 1
    for atom in spc_2.mol.vertices:
        element_dict_2[atom.element.symbol] = element_dict_2.get(atom.element.symbol, 0) + 1
    for key, val in element_dict_1.items():
        if val != element_dict_2[key]:
            if verbose:
                logger.warning(f'The chemical formula of the two species is not identical, got the following elements:\n'
                               f'{element_dict_1}\n{element_dict_2}')
            return False
    # Check the same number of bonds between similar elements (ignore bond order).
    bonds_dict_1, bonds_dict_2 = get_bonds_dict(spc_1), get_bonds_dict(spc_2)
    for key, val in bonds_dict_1.items():
        if val != bonds_dict_2[key]:
            if verbose:
                logger.warning(f'The chemical bonds in the two species are not identical, got the following bonds:\n'
                               f'{bonds_dict_1}\n{bonds_dict_2}')
            return False
    # Check whether both species are linear or both are non-linear.
    if spc_1.mol.is_linear() != spc_2.mol.is_linear():
        if verbose:
            logger.warning(f'Both species should be either linear or non-linear, got:\n'
                           f'linear = {spc_1.mol.is_linear()} and linear = {spc_2.mol.is_linear()}.')
        return False
    return True


def get_bonds_dict(spc: ARCSpecies) -> Dict[str, int]:
    """
    Get a dictionary of bonds by elements in the species ignoring bond orders.

    Args:
        spc (ARCSpecies): The species to examine.

    Returns:
        Dict[str, int]: Keys are 'A-B' strings of element symbols sorted alphabetically (e.g., 'C-H' or 'H-O'),
                        values are the number of such bonds (ignoring bond order).
    """
    bond_dict = dict()
    bonds = spc.mol.get_all_edges()
    for bond in bonds:
        elements = sorted([bond.atom1.element.symbol, bond.atom2.element.symbol])
        key = f'{elements[0]}-{elements[1]}'
        bond_dict[key] = bond_dict.get(key, 0) + 1
    return bond_dict


def determine_adjacent_elements(spc: ARCSpecies) -> Dict[int, Dict[str, Union[str, List[int]]]]:
    """
    Determine the type and number of adjacent elements for each heavy atom in ``spc``.

    Args:
        spc (ARCSpecies): The input species.

    Returns:
        Dict[int, Dict[str, List[int]]]: Keys are indices of heavy atoms, values are dicts. keys are element symbols,
                                         values are indices of adjacent atoms corresponding to this element.
    """
    adjacent_element_dict = dict()
    for i, atom_1 in enumerate(spc.mol.atoms):
        if atom_1.is_hydrogen():
            continue
        adjacent_elements = {'self': atom_1.element.symbol}
        for atom_2 in atom_1.edges.keys():
            if atom_2.element.symbol not in adjacent_elements.keys():
                adjacent_elements[atom_2.element.symbol] = list()
            adjacent_elements[atom_2.element.symbol].append(spc.mol.atoms.index(atom_2))
        adjacent_element_dict[i] = adjacent_elements
    return adjacent_element_dict


def identify_superimposable_candidates(adj_element_dict_1: Dict[int, Dict[str, Union[str, List[int]]]],
                                       adj_element_dict_2: Dict[int, Dict[str, Union[str, List[int]]]],
                                       ) -> List[Dict[int, int]]:
    """
    Identify candidate ordering of heavy atoms (only) that could potentially be superiposed.

    Args:
        adj_element_dict_1 (Dict[int, Dict[str, Union[str, List[int]]]]): Adjacent element dict for species 1.
        adj_element_dict_2 (Dict[int, Dict[str, Union[str, List[int]]]]): Adjacent element dict for species 2.

    Returns:
        List[Dict[int, int]]: Entries are superimposable candidate dicts. Keys are atom indices of heavy atoms
                              of species 1, values are potentially mapped atom indices of species 2.
    """
    candidates = list()
    for key_1, val_1 in adj_element_dict_1.items():
        for key_2, val_2 in adj_element_dict_2.items():
            # Try all combinations of heavy atoms.
            result = iterative_dfs(adj_element_dict_1, adj_element_dict_2, key_1, key_2)
            if result:
                candidates.append(result)
    return prune_identical_dicts(candidates)


def are_adj_elements_in_agreement(adj_elements_1: Dict[str, Union[str, List[int]]],
                                  adj_elements_2: Dict[str, Union[str, List[int]]],
                                  ) -> bool:
    """
    Check whether two dictionaries representing adjacent elements are in agreement
    w.r.t. the type and number of adjacent elements.
    Also checks the identity of the parent ("self") element.

    Args:
          adj_elements_1 (Dict[str, List[int]]): Adjacent elements dictionary 1.
          adj_elements_2 (Dict[str, List[int]]): Adjacent elements dictionary 2.

    Returns:
        bool: ``True`` if the two dicts represent identical adjacent elements, ``False`` otherwise.
    """
    if len(list(adj_elements_1.keys())) != len(list(adj_elements_2.keys())):
        return False
    if adj_elements_1['self'] != adj_elements_2['self']:
        return False
    for key, val in adj_elements_1.items():
        if key != 'self' and (key not in adj_elements_2 or len(val) != len(adj_elements_2[key])):
            return False
    return True


def iterative_dfs(adj_elements_1: Dict[int, Dict[str, List[int]]],
                  adj_elements_2: Dict[int, Dict[str, List[int]]],
                  key_1: int,
                  key_2: int,
                  ) -> Dict[int, int]:
    """
    A depth first search (DFS) graph traversal algorithm to determine possible superimposable ordering of heavy atoms.
    This is an iterative and not a recursive algorithm since Python doesn't have a great support for recursion
    since it lacks Tail Recursion Elimination and because there is a limit of recursion stack depth (by default is 1000).

    Args:
        adj_elements_1 (Dict[int, Dict[str, List[int]]]): Adjacent elements dictionary 1 (graph 1).
        adj_elements_2 (Dict[int, Dict[str, List[int]]]): Adjacent elements dictionary 2 (graph 2).
        key_1 (int): The starting index for graph 1.
        key_2 (int): The starting index for graph 2.

    Returns:
        Dict[int, int]: ``None`` if this is not a valid superimposable candidate. Keys are atom indices of
                        heavy atoms of species 1, values are potentially mapped atom indices of species 2.
    """
    visited_1, visited_2 = list(), list()
    stack_1, stack_2 = deque(), deque()
    stack_1.append(key_1)
    stack_2.append(key_2)
    result = dict()
    while stack_1 and stack_2:
        key_1 = stack_1.pop()
        key_2 = stack_2.pop()
        if key_1 in visited_1 or key_2 in visited_2:
            continue
        visited_1.append(key_1)
        visited_2.append(key_2)
        if not are_adj_elements_in_agreement(adj_elements_1[key_1], adj_elements_2[key_2]):
            continue
        result[key_1] = key_2
        for symbol in adj_elements_1[key_1].keys():
            if symbol not in ['self', 'H']:
                for combination_tuple in product(adj_elements_1[key_1][symbol], adj_elements_2[key_2][symbol]):
                    if combination_tuple[0] not in visited_1 and combination_tuple[1] not in visited_2:
                        stack_1.append(combination_tuple[0])
                        stack_2.append(combination_tuple[1])
    return result


def prune_identical_dicts(dicts_list: List[dict]) -> List[dict]:
    """
    Return a list of unique dictionaries.

    Args:
        dicts_list (List[dict]): A list of dicts to prune.

    Returns:
        List[dict]: A list of unique dicts.
    """
    new_dicts_list = list()
    for new_dict in dicts_list:
        unique = True
        for existing_dict in new_dicts_list:
            if unique:
                for new_key, new_val in new_dict.items():
                    if new_key not in existing_dict.keys() or new_val == existing_dict[new_key]:
                        unique = False
                        break
        if unique:
            new_dicts_list.append(new_dict)
    return new_dicts_list


def fix_dihedrals_by_backbone_mapping(spc_1: ARCSpecies,
                                      spc_2: ARCSpecies,
                                      backbone_map: Dict[int, int],
                                      ) -> Tuple[ARCSpecies, ARCSpecies]:
    """
    Fix the dihedral angles of two mapped species to align them.

    Args:
        spc_1 (ARCSpecies): Species 1.
        spc_2 (ARCSpecies): Species 2.
        backbone_map (Dict[int, int]): The backbone map.

    Returns:
        Tuple[ARCSpecies, ARCSpecies]: The corresponding species with aligned dihedral angles.
    """
    if not spc_1.rotors_dict:
        spc_1.determine_rotors()
    if not spc_2.rotors_dict:
        spc_2.determine_rotors()
    spc_1_copy, spc_2_copy = spc_1.copy(), spc_2.copy()
    torsions = get_backbone_dihedral_angles(spc_1, spc_2, backbone_map)
    deviations = [get_backbone_dihedral_deviation_score(spc_1, spc_2, backbone_map, torsions=torsions)]
    # Loop while the deviation improves by more than 1 degree:
    while len(torsions) and (len(deviations) < 2 or deviations[-2] - deviations[-1] > 1):
        for torsion_dict in torsions:
            angle = 0.5 * sum([torsion_dict['angle 1'], torsion_dict['angle 2']])
            spc_1_copy.set_dihedral(scan=convert_list_index_0_to_1(torsion_dict['torsion 1']),
                                    deg_abs=angle, count=False, chk_rotor_list=False)
            spc_2_copy.set_dihedral(scan=convert_list_index_0_to_1(torsion_dict['torsion 2']),
                                    deg_abs=angle, count=False, chk_rotor_list=False)
            spc_1_copy.final_xyz, spc_2_copy.final_xyz = spc_1_copy.initial_xyz, spc_2_copy.initial_xyz
        torsions = get_backbone_dihedral_angles(spc_1_copy, spc_2_copy, backbone_map)
        deviations.append(get_backbone_dihedral_deviation_score(spc_1_copy, spc_2_copy, backbone_map, torsions=torsions))
    return spc_1_copy, spc_2_copy


def get_backbone_dihedral_deviation_score(spc_1: ARCSpecies,
                                          spc_2: ARCSpecies,
                                          backbone_map: Dict[int, int],
                                          torsions: Optional[List[Dict[str, Union[float, List[int]]]]] = None
                                          ) -> float:
    """
    Determine a deviation score for dihedral angles of torsions.
    We don't consider here "terminal" torsions, just pure backbone torsions.

    Args:
        spc_1 (ARCSpecies): Species 1.
        spc_2 (ARCSpecies): Species 2.
        backbone_map (Dict[int, int]): The backbone map.
        torsions (Optional[List[Dict[str, Union[float, List[int]]]]], optional): The backbone dihedral angles.
    """
    torsions = torsions or get_backbone_dihedral_angles(spc_1, spc_2, backbone_map)
    return sum([abs(torsion_dict['angle 1'] - torsion_dict['angle 2']) for torsion_dict in torsions])


def get_backbone_dihedral_angles(spc_1: ARCSpecies,
                                 spc_2: ARCSpecies,
                                 backbone_map: Dict[int, int],
                                 ) -> List[Dict[str, Union[float, List[int]]]]:
    """
    Determine the dihedral angles of the backbone torsions of two backbone mapped species.
    The output has the following format::

        torsions = [{'torsion 1': [0, 1, 2, 3],  # The first torsion in terms of species 1 indices.
                     'torsion 2': [5, 7, 2, 4],  # The first torsion in terms of species 2 indices.
                     'angle 1': 60.0,  # The corresponding dihedral angle to 'torsion 1'.
                     'angle 2': 125.1,  # The corresponding dihedral angle to 'torsion 2'.
                    },
                    {}, ...  # The second torsion, and so on.
                   ]

    Args:
        spc_1 (ARCSpecies): Species 1.
        spc_2 (ARCSpecies): Species 2.
        backbone_map (Dict[int, int]): The backbone map.

    Returns:
        List[Dict[str, Union[float, List[int]]]]: The corresponding species with aligned dihedral angles.
    """
    torsions = list()
    for rotor_dict_1 in spc_1.rotors_dict.values():
        torsion_1 = rotor_dict_1['torsion']
        if not spc_1.mol.atoms[torsion_1[0]].is_hydrogen() \
                and not spc_1.mol.atoms[torsion_1[3]].is_hydrogen():
            # This is not a "terminal" torsion
            for rotor_dict_2 in spc_2.rotors_dict.values():
                torsion_2 = [backbone_map[t_1] for t_1 in torsion_1]
                if all(pivot_2 in [torsion_2[1], torsion_2[2]]
                       for pivot_2 in [rotor_dict_2['torsion'][1], rotor_dict_2['torsion'][2]]):
                    torsions.append({'torsion 1': torsion_1,
                                     'torsion 2': torsion_2,
                                     'angle 1': calculate_dihedral_angle(coords=spc_1.get_xyz(), torsion=torsion_1),
                                     'angle 2': calculate_dihedral_angle(coords=spc_2.get_xyz(), torsion=torsion_2)})
    return torsions


def map_hydrogens(spc_1: ARCSpecies,
                  spc_2: ARCSpecies,
                  backbone_map: Dict[int, int],
                  ) -> Dict[int, int]:
    """
    Atom map hydrogen atoms between two species with a known mapped heavy atom backbone.
    If only a single hydrogen atom is bonded to a given heavy atom, it is straight-forwardly mapped.
    If more than one hydrogen atom is bonded to a given heavy atom forming a "terminal" internal rotor,
    an internal rotation will be attempted to find the closest match, e.g., in cases such as::

        C -- H1     |         H1
          \         |       /
           H2       |     C -- H2

    To avoid mapping H2 to H1 due to small RMSD, but H1 to H2 although the RMSD is huge.

    Args:
        spc_1 (ARCSpecies): Species 1.
        spc_2 (ARCSpecies): Species 2.
        backbone_map (Dict[int, int]): The backbone map.

    Returns:
        Dict[int, int]: The atom map. Keys are 0-indices in ``spc_1``, values are corresponding 0-indices in ``spc_2``.
    """
    # convert to zmats by putting the backbone first
    # then convert back to xyz - expect to have overlapping xyz's
    # check H RMSD in the two species by actual deviation in XYZ as if they are on the same coordinate space
    # rotate to minimize RMSD and switcing H on this rotor mapping if needed
    # append each positive map to the map dict and return it







































