"""
An adapter for executing xTB (Semiempirical Extended Tight-Binding Program Package) jobs

https://github.com/grimme-lab/xtb
https://xtb-docs.readthedocs.io/en/latest/contents.html
run types: https://xtb-docs.readthedocs.io/en/latest/commandline.html?highlight=--grad#runtypes
"""

import datetime
import os
from typing import TYPE_CHECKING, List, Optional, Tuple, Union

from mako.template import Template

from arc.common import get_logger, torsions_to_scans
from arc.imports import incore_commands, settings
from arc.job.adapter import JobAdapter
from arc.job.adapters.common import _initialize_adapter
from arc.job.factory import register_job_adapter
from arc.job.local import execute_command
from arc.level import Level
from arc.species.converter import xyz_to_turbomol_format
from arc.species.vectors import calculate_dihedral_angle

if TYPE_CHECKING:
    from arc.reaction import ARCReaction
    from arc.species import ARCSpecies

logger = get_logger()

default_job_settings, global_ess_settings, input_filenames, output_filenames, rotor_scan_resolution, servers, \
    submit_filenames = settings['default_job_settings'], settings['global_ess_settings'], settings['input_filenames'], \
                       settings['output_filenames'], settings['rotor_scan_resolution'], settings['servers'], \
                       settings['submit_filenames']


input_template = """${coords}
${block}
"""


class xTBAdapter(JobAdapter):
    """
    A class for executing xTB jobs.

    Args:
        project (str): The project's name. Used for setting the remote path.
        project_directory (str): The path to the local project directory.
        job_type (list, str): The job's type, validated against ``JobTypeEnum``. If it's a list, pipe.py will be called.
        args (dict, optional): Methods (including troubleshooting) to be used in input files.
                               Keys are either 'keyword', 'block', or 'trsh', values are dictionaries with values
                               to be used either as keywords or as blocks in the respective software input file.
                               If 'trsh' is specified, an action might be taken instead of appending a keyword or a
                               block to the input file (e.g., change server or change scan resolution).
        bath_gas (str, optional): A bath gas. Currently only used in OneDMin to calculate L-J parameters.
        checkfile (str, optional): The path to a previous Gaussian checkfile to be used in the current job.
        conformer (int, optional): Conformer number if optimizing conformers.
        constraints (list, optional): A list of constraints to use during an optimization or scan.
        cpu_cores (int, optional): The total number of cpu cores requested for a job.
        dihedral_increment (float, optional): The degrees increment to use when scanning dihedrals of TS guesses.
        dihedrals (List[float], optional): The dihedral angels corresponding to self.torsions.
        directed_scan_type (str, optional): The type of the directed scan.
        ess_settings (dict, optional): A dictionary of available ESS and a corresponding server list.
        ess_trsh_methods (List[str], optional): A list of troubleshooting methods already tried out.
        execution_type (str, optional): The execution type, 'incore', 'queue', or 'pipe'.
        fine (bool, optional): Whether to use fine geometry optimization parameters. Default: ``False``.
        initial_time (datetime.datetime or str, optional): The time at which this job was initiated.
        irc_direction (str, optional): The direction of the IRC job (`forward` or `reverse`).
        job_id (int, optional): The job's ID determined by the server.
        job_memory_gb (int, optional): The total job allocated memory in GB (14 by default).
        job_name (str, optional): The job's name (e.g., 'opt_a103').
        job_num (int, optional): Used as the entry number in the database, as well as in ``job_name``.
        job_server_name (str, optional): Job's name on the server (e.g., 'a103').
        job_status (list, optional): The job's server and ESS statuses.
        level (Level, optional): The level of theory to use.
        max_job_time (float, optional): The maximal allowed job time on the server in hours (can be fractional).
        reactions (List[ARCReaction], optional): Entries are ARCReaction instances, used for TS search methods.
        rotor_index (int, optional): The 0-indexed rotor number (key) in the species.rotors_dict dictionary.
        server (str): The server to run on.
        server_nodes (list, optional): The nodes this job was previously submitted to.
        species (List[ARCSpecies], optional): Entries are ARCSpecies instances.
                                              Either ``reactions`` or ``species`` must be given.
        testing (bool, optional): Whether the object is generated for testing purposes, ``True`` if it is.
        times_rerun (int, optional): Number of times this job was re-run with the same arguments (no trsh methods).
        torsions (List[List[int]], optional): The 0-indexed atom indices of the torsion(s).
        tsg (int, optional): TSGuess number if optimizing TS guesses.
        xyz (dict, optional): The 3D coordinates to use. If not give, species.get_xyz() will be used.
    """

    def __init__(self,
                 project: str,
                 project_directory: str,
                 job_type: Union[List[str], str],
                 args: Optional[dict] = None,
                 bath_gas: Optional[str] = None,
                 checkfile: Optional[str] = None,
                 conformer: Optional[int] = None,
                 constraints: Optional[List[Tuple[List[int], float]]] = None,
                 cpu_cores: Optional[str] = None,
                 dihedral_increment: Optional[float] = None,
                 dihedrals: Optional[List[float]] = None,
                 directed_scan_type: Optional[str] = None,
                 ess_settings: Optional[dict] = None,
                 ess_trsh_methods: Optional[List[str]] = None,
                 execution_type: Optional[str] = None,
                 fine: bool = False,
                 initial_time: Optional[Union['datetime.datetime', str]] = None,
                 irc_direction: Optional[str] = None,
                 job_id: Optional[int] = None,
                 job_memory_gb: float = 14.0,
                 job_name: Optional[str] = None,
                 job_num: Optional[int] = None,
                 job_server_name: Optional[str] = None,
                 job_status: Optional[List[Union[dict, str]]] = None,
                 level: Optional[Level] = None,
                 max_job_time: Optional[float] = None,
                 reactions: Optional[List['ARCReaction']] = None,
                 rotor_index: Optional[int] = None,
                 server: Optional[str] = None,
                 server_nodes: Optional[list] = None,
                 species: Optional[List['ARCSpecies']] = None,
                 testing: bool = False,
                 times_rerun: int = 0,
                 torsions: Optional[List[List[int]]] = None,
                 tsg: Optional[int] = None,
                 xyz: Optional[dict] = None,
                 ):

        self.incore_capacity = 100
        self.job_adapter = 'xtb'
        self.execution_type = execution_type or 'incore'
        self.command = 'xtb'
        self.url = 'https://github.com/grimme-lab/xtb'

        if species is None:
            raise ValueError('Cannot execute xTB without an ARCSpecies object.')

        _initialize_adapter(obj=self,
                            is_ts=False,
                            project=project,
                            project_directory=project_directory,
                            job_type=job_type,
                            args=args,
                            bath_gas=bath_gas,
                            checkfile=checkfile,
                            conformer=conformer,
                            constraints=constraints,
                            cpu_cores=cpu_cores,
                            dihedral_increment=dihedral_increment,
                            dihedrals=dihedrals,
                            directed_scan_type=directed_scan_type,
                            ess_settings=ess_settings,
                            ess_trsh_methods=ess_trsh_methods,
                            fine=fine,
                            initial_time=initial_time,
                            irc_direction=irc_direction,
                            job_id=job_id,
                            job_memory_gb=job_memory_gb,
                            job_name=job_name,
                            job_num=job_num,
                            job_server_name=job_server_name,
                            job_status=job_status,
                            level=level,
                            max_job_time=max_job_time,
                            reactions=reactions,
                            rotor_index=rotor_index,
                            server=server,
                            server_nodes=server_nodes,
                            species=species,
                            testing=testing,
                            times_rerun=times_rerun,
                            torsions=torsions,
                            tsg=tsg,
                            xyz=xyz,
                            )

    def write_input_file(self) -> None:
        """
        Write the input file to execute the job on the server.
        """
        block, job_type = '', ''
        coords = xyz_to_turbomol_format(self.xyz,
                                        charge=self.charge,
                                        unpaired=self.species[0].number_of_radicals or self.multiplicity - 1)

        if self.job_type in ['opt', 'conformers', 'scan']:
            job_type = ' --opt'
            job_type = self.add_accuracy(job_type)
            if self.constraints and job_type != 'scan':
                block += f'$constrain\n'
                block += self.add_constraints_to_block()
                block += f'$opt\n   maxcycle=5\n$end'

        elif self.job_type in ['freq']:
            job_type = ' --hess'

        elif self.job_type in ['optfreq']:
            job_type = ' --ohess'

        elif self.job_type in ['fukui']:
            job_type = ' --vfukui'

        elif self.job_type == 'sp':
            pass

        if self.level is not None and self.level.solvent is not None:
            job_type += f' --alpb {self.level.solvent.lower()}'

        if self.job_type in ['scan']:
            scans = list()
            if self.rotor_index is not None:
                if self.species[0].rotors_dict \
                        and self.species[0].rotors_dict[self.rotor_index]['directed_scan_type'] == 'ess':
                    scans = self.species[0].rotors_dict[self.rotor_index]['scan']
                    scans = [scans] if not isinstance(scans[0], list) else scans
            elif self.torsions is not None and len(self.torsions):
                scans = torsions_to_scans(self.torsions)

            force_constant = 0.05
            block += f'$constrain\n   force constant={force_constant}\n'
            block += self.add_constraints_to_block()
            block += '$scan\n'
            for scan in scans:
                scan_string = '   dihedral: '
                dihedral_angle = int(calculate_dihedral_angle(coords=self.xyz, torsion=scan, index=1))
                scan_string += ', '.join([str(atom_index) for atom_index in scan])
                scan_string += f', {dihedral_angle}; {dihedral_angle}, {dihedral_angle + 360.0}, {int(360 / self.scan_res)}\n'
                block += scan_string
            block += '$end'

        with open(os.path.join(self.local_path, input_filenames[self.job_adapter]), 'w') as f:
            f.write(f'{self.command} input.in{job_type} > {output_filenames[self.job_adapter]}\n')

        with open(os.path.join(self.local_path, 'input.in'), 'w') as f:
            f.write(Template(input_template).render(coords=coords, block=block))

    def add_constraints_to_block(self) -> str:
        """
        Add the constraints section to an xTB input file.

        Returns:
            str: The updated ``block``.
        """
        block = ''
        if self.constraints is not None:
            for constraint in self.constraints:
                if len(constraint[0]) == 2:
                    block += f'   distance: {constraint[0][0]}, {constraint[0][1]}, {constraint[1]}\n'
                if len(constraint[0]) == 3:
                    block += f'   angle: {constraint[0][0]}, {constraint[0][1]}, {constraint[0][2]}, {constraint[1]}\n'
                if len(constraint[0]) == 4:
                    block += f'   dihedral: {constraint[0][0]}, {constraint[0][1]}, {constraint[0][2]}, {constraint[0][3]}, {constraint[1]}\n'
        else:
            block += '\n'
        return block

    def add_accuracy(self, job_type: str) -> str:
        """
        Add an accuracy level to the job specifications.

        Args:
            job_type (str): The original ``job_type``.

        Returns:
            str: The updated ``job_type``.
        """
        if self.fine:
            job_type += ' vtight'
        elif self.args and 'keyword' in self.args.keys() and 'accuracy' in self.args['keyword'].keys():
            # Accuracy threshold level ('crude', 'sloppy', 'loose', 'lax', 'normal', 'tight', 'vtight', 'extreme').
            job_type += f' {self.args["keyword"]["accuracy"]}'
        return job_type

    def set_files(self) -> None:
        """
        Set files to be uploaded and downloaded. Writes the files if needed.
        Modifies the self.files_to_upload and self.files_to_download attributes.

        self.files_to_download is a list of remote paths.

        self.files_to_upload is a list of dictionaries, each with the following keys:
        ``'name'``, ``'source'``, ``'make_x'``, ``'local'``, and ``'remote'``.
        If ``'source'`` = ``'path'``, then the value in ``'local'`` is treated as a file path.
        Else if ``'source'`` = ``'input_files'``, then the value in ``'local'`` will be taken
        from the respective entry in inputs.py
        If ``'make_x'`` is ``True``, the file will be made executable.
        """
        # 1. ** Upload **
        # 1.1. submit file
        if self.execution_type != 'incore':
            # we need a submit file for single or array jobs (either submitted to local or via SSH)
            self.write_submit_script()
            self.files_to_upload.append(self.get_file_property_dictionary(
                file_name=submit_filenames[servers[self.server]['cluster_soft']]))
        # 1.2. input file
        self.write_input_file()
        self.files_to_upload.append(self.get_file_property_dictionary(file_name=input_filenames[self.job_adapter]))
        self.files_to_upload.append(self.get_file_property_dictionary(file_name='input.in'))
        # 1.3. HDF5 file
        if self.iterate_by and os.path.isfile(os.path.join(self.local_path, 'data.hdf5')):
            self.files_to_upload.append(self.get_file_property_dictionary(file_name='data.hdf5'))
        # 1.4 job.sh
        job_sh_dict = self.set_job_shell_file_to_upload()  # Set optional job.sh files if relevant.
        if job_sh_dict is not None:
            self.files_to_upload.append(job_sh_dict)
        # 2. ** Download **
        # 2.1. HDF5 file
        if self.iterate_by and os.path.isfile(os.path.join(self.local_path, 'data.hdf5')):
            self.files_to_download.append(self.get_file_property_dictionary(file_name='data.hdf5'))
        else:
            # 2.2. log file
            self.files_to_download.append(self.get_file_property_dictionary(
                file_name=output_filenames[self.job_adapter]))
            # 2.3. scan log file
            self.files_to_download.append(self.get_file_property_dictionary(file_name='xtbscan.log'))
            # 2.4. normal mode displacement log file
            self.files_to_download.append(self.get_file_property_dictionary(file_name='g98.out'))
            # 2.5. hessian log file
            self.files_to_download.append(self.get_file_property_dictionary(file_name='hessian'))

    def set_additional_file_paths(self) -> None:
        """
        Set additional file paths specific for the adapter.
        Called from set_file_paths() and extends it.
        """
        pass

    def set_input_file_memory(self) -> None:
        """
        Set the input_file_memory attribute.
        """
        pass

    def execute_incore(self):
        """
        Execute a job incore.
        """
        self._log_job_execution()
        execute_command([f'cd {self.local_path}'] + incore_commands[self.job_adapter], executable='/bin/bash')

    def execute_queue(self):
        """
        Execute a job to the server's queue.
        """
        self.legacy_queue_execution()


register_job_adapter('xtb', xTBAdapter)