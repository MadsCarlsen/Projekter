# %%
import numpy as np

#%%
class BasisFunction:
    """ Class that functions as a contracted GTO basis function """
    def __init__(self, i, j, k, exponent_list, contract_list, atom_position):
        # Exponents and contraction coefficients for the contracted GTO's:
        self.exp_list = exponent_list
        self.contract_list = contract_list

        # Coefficients for x, y and z respectively:
        self.i = i
        self.j = j
        self.k = k

        self.atom_position = atom_position

        # Calculate the front factors of the GTO's and save them for later use:
        self.front_factor = (2/np.pi)**(3/4) * np.sqrt(8**(i+j+k) * np.math.factorial(i) \
                * np.math.factorial(j) * np.math.factorial(k) / (np.math.factorial(2*i) * np.math.factorial(2*j) * np.math.factorial(2*k)))
        self.front_alphas = [alpha**(3/4) * alpha**(1/2 * (i+j+k)) for alpha in exponent_list]

    def eval(self, x, y, z):
        """ Evaluates the basis function at a given x,y,z """
        # Transform to the atom coordinate system!
        x = x - self.atom_position[0]
        y = y - self.atom_position[1]
        z = z - self.atom_position[2]

        if type(x) == np.ndarray:
            res = np.zeros_like(x)
        else:
            res = 0

        # Sum up the individual primitive Gaussians!
        for alpha, front_alpha, contract in zip(self.exp_list, self.front_alphas, self.contract_list):
            res += front_alpha * contract * np.exp(-alpha * (x**2 + y**2 + z**2))

        factor = self.front_factor * x**self.i * y**self.j * z**self.k
        return res * factor


class OutputInterface:
    """ Class that loads data of the output files from GAMESS """

    def __init__(self, file_name):
        self.file_name = file_name
        self.position = {}  # Position of atoms in molecule
        self.elements = {}  # Element corresponding to number

        # Contains the information from the ATOMIC BASIS SET section
        # Organized as: basis[atom_nr] = [type of orbital, list of exponents, list of contrction coeff.]
        self.basis = {}

        # Contains the information from the MOLECULAR ORBITALS section
        # Organized as: saved_orbital[orbital_nr] = [energy, list with atom nr., list with shell_type, list with MO_coeffs.]
        self.saved_orbitals = {}  # Data of all orbitals loaded

        self.basis_funcs = []  # Basis functions ready to call

        # Main data load loop:
        with open(file_name, 'r') as file:
            line = file.readline()
            while line:
                # Here we detect when we reach the basis set section!
                if 'ATOMIC BASIS SET' in line:
                    line = file.readline()
                    basis_lines = []
                    while True:
                        line = file.readline()
                        if 'TOTAL NUMBER OF BASIS SET SHELLS' in line:
                            break
                        basis_lines.append(line)
                    self.extract_basis(basis_lines)  # Send the basis data to this function for handling

                # Get the number of occupied orbitals! (Same as number of HOMO)
                if 'NUMBER OF OCCUPIED ORBITALS (ALPHA)' in line:
                    self.HOMO = int(line.split()[-1])

                # Here we obtain the geometry of the molecule!
                if 'EQUILIBRIUM GEOMETRY LOCATED' in line:
                    angs_pr_bohr = 0.529177210903  # Å / Bohr_radius
                    for _ in range(4):
                        line = file.readline()
                    atom_nr = 1
                    while line != '\n':
                        self.position[atom_nr] = [float(num)/angs_pr_bohr for num in line.split()[2:]]  # Convert to atomic units!
                        self.elements[atom_nr] = line.split()[1]
                        line = file.readline()
                        atom_nr += 1
                    break  # We don't want to read more of the file!

                line = file.readline()

        # Now construct the basis functions to be used when calculating values of orbitals!
        self.make_basis_functions()

        # Load the HOMO as standard
        self.load_orbital(self.HOMO)

    def extract_basis(self, lines):
        """ Function that extracts the basis set data from the lines of the output file, and
         puts it together in the form described in __init__. This function is a bit ugly..."""
        atoms = []
        contract_list = []
        exponent_list = []
        shell_type = []
        lines = [line for line in lines if line != '\n']  # Removes all empty lines
        atom_nr = 0
        current_shell = 0
        current_shell_type = 'S'

        # Remove text in top:
        while True:
            if len(lines[0].split()) != 1:
                lines.pop(0)
            else:
                break

        # Loop over the shells
        for line in lines:
            split = line.split()

            # New atom
            if len(split) == 1:
                current_shell = 0
                atom_nr += 1
                atoms.append(line)
                if atom_nr != 1:  # Save the data for the last atom looped over
                    if current_shell_type == 'L':  # Make sure we append the P if an L shell was last in an atom
                        contract_list.append(contract_coeff[:])
                        exponent_list.append(exponent[:])
                        contract_list.append(contract_p[:])
                        exponent_list.append(exponent[:])
                    else: 
                        contract_list.append(contract_coeff[:])
                        exponent_list.append(exponent[:])
                    # Save and reset lists for next atom
                    self.basis[atom_nr-1] = [shell_type[:], exponent_list[:], contract_list[:]]
                    contract_list = []
                    exponent_list = []
                    shell_type = []
                    current_shell_type = 'S'  # Assumes we always start with an S shell
                continue

            # New shell
            if current_shell != int(split[0]):
                if current_shell != 0:
                    if current_shell_type != 'L':  # Make sure to save the P also if current shell is L type
                        contract_list.append(contract_coeff[:])
                        exponent_list.append(exponent[:])
                    else:
                        exponent_list.append(exponent[:])
                        exponent_list.append(exponent[:])
                        contract_list.append(contract_coeff[:])
                        contract_list.append(contract_p[:])
                if split[1] != 'L':
                    shell_type.append(split[1])
                else:
                    shell_type.append('S')
                    shell_type.append('P')

                current_shell = int(split[0])
                current_shell_type = split[1]
                contract_coeff = []  # Reset
                exponent = []
                contract_p = []

            # For every loop save the contraction coeffs. and exponents
            exponent.append(float(split[3]))
            contract_coeff.append(float(split[4]))
            if split[1] == 'L':
                contract_p.append(float(split[5]))

        # Out of loop - remember to append the last atom!
        if current_shell_type != 'L':
            contract_list.append(contract_coeff[:])
            exponent_list.append(exponent[:])
        else:
            exponent_list.append(exponent[:])
            exponent_list.append(exponent[:])
            contract_list.append(contract_coeff[:])
            contract_list.append(contract_p[:])
        self.basis[atom_nr] = [shell_type[:], exponent_list[:], contract_list[:]]

    def load_orbital(self, orbital_nr):
        """ Loads a given orbital from the output file """

        # Check if the orbital is already loaded
        if orbital_nr in self.saved_orbitals.keys():
            print('This orbital is already loaded!')
            return None

        # Main loop over the lines in the output file
        with open(self.file_name, 'r') as file:
            line = file.readline()
            while line:
                if 'MOLECULAR ORBITALS' in line:  # Located the part where the orbitals live
                    for _ in range(3):
                        line = file.readline()
                    raw_data = []
                    # Now loop over the data of the orbitals
                    while True:
                        if str(orbital_nr) in line.split():  # If we are at the right orbital
                            # Get the chosen orbital data
                            while line != '\n':
                                raw_data.append(line)
                                line = file.readline()

                            # Now save the data in the format in __init__
                            nr_list = []
                            shell_type = []
                            MO_coeff = []
                            line0 = np.array([int(num) for num in raw_data[0].split()])  # I am not a pro with argwhere...
                            target_index = np.argwhere(line0 == orbital_nr)[0, 0] + 4  # Index of target column
                            energy = float(raw_data[1].split()[target_index-4])  # Get the energy of the orbital

                            # Fix the types
                            for line in raw_data[3:]:
                                split = line.split()
                                nr_list.append(int(split[2]))
                                shell_type.append(split[3])
                                MO_coeff.append(float(split[target_index]))

                            self.saved_orbitals[orbital_nr] = [energy, nr_list, shell_type, MO_coeff]
                            break
                        else:
                            # This runs when we did not find the right orbital - skip lines until next set begins
                            while line != '\n':
                                line = file.readline()
                            line = file.readline()

                        # If we detect a line with ---- the MO data is over.
                        if '---------------' in line:
                            print('Did not find target orbital!')
                            break

                    break  # Don't read the rest of the file!
                line = file.readline()

    def get_combinations(self, shell_type):
        """ Here the combinations for the different shell types are defined. Note that the order must be as in the
        output file. Add more if higher shell types excist in the basis! """
        if shell_type == 'P':
            return ['X', 'Y', 'Z']
        elif shell_type == 'D':
            return ['XX', 'YY', 'ZZ', 'XY', 'XZ', 'YZ']
        else:
            print("I don't know this shell type yet!")
            # Raise some error somehow...
            return None

    def make_basis_functions(self):
        """ This functions creates the contracted basis functions and save them for later use. In this way we avoid
         putting them together for every point in the calculation. """
        number_of_atoms = max(self.basis.keys())
        # Loop over the different atoms
        for atom_i in range(1, number_of_atoms + 1):
            atom_pos = self.position[atom_i]
            basis_shells, exponent_list, contract_list = self.basis[atom_i]

            # Loop over the shells in the atom and build the basis functions
            for shell_i, shell in enumerate(basis_shells):
                if shell == 'S':
                    self.basis_funcs.append(BasisFunction(0, 0, 0, exponent_list[shell_i], contract_list[shell_i], atom_pos))
                else:
                    for combi in self.get_combinations(shell):
                        # Find the values of i, j, k for the specific combination
                        # - a bit stupid way, but then it looks like the output file...
                        i, j, k = (0, 0, 0)
                        for val in combi:
                            if val == 'X':
                                i += 1
                            elif val == 'Y':
                                j += 1
                            else:
                                k += 1
                        self.basis_funcs.append(BasisFunction(i, j, k, exponent_list[shell_i], contract_list[shell_i], atom_pos))

    def output_GTOs(self, orbital_nr=None):
        """
        Outputs a list with all GTO parameters, total front factor, alpha, (i,j,k), displacement (x0,y0,z0)
        """
        if orbital_nr is None:
            orbital_nr = self.HOMO

        GTO_list = []
        MO_coeffs = self.saved_orbitals[orbital_nr][3]
        for n, func in enumerate(self.basis_funcs):
            for m, alpha in enumerate(func.exp_list):
                i, j, k = (func.i, func.j, func.k)
                x0, y0, z0 = func.atom_position
                
                N = (2*alpha/np.pi)**(3/4) * np.sqrt((8*alpha)**(i + j + k) * np.math.factorial(i) \
                    * np.math.factorial(j) * np.math.factorial(k) / (np.math.factorial(2*i) \
                    * np.math.factorial(2*j) * np.math.factorial(2*k)))
                
                front_factor = MO_coeffs[n] * func.contract_list[m] * N
                
                GTO_list.append([front_factor, alpha, i, j, k, x0, y0, z0])
        return GTO_list

    def export_GTOs(self, path=None, orbital_nr=None):
        """
        Outputs the data from output_GTOs to a txt file saved at the given path. If none is given the working dic. is used.
        """
        if orbital_nr is None:
            orbital_nr = self.HOMO

        # Get the name for the output file
        name = self.file_name
        if '/' in name:  # Remove the path
            name = name.split('/')[-1]
        if '.out' in name:  # Remove the file extension
            name = name[:-4]

        output = self.output_GTOs(orbital_nr)
        np.savetxt(f'{name}_GTOs.txt', output)

    def eval_orbital(self, x, y, z, orbital_nr=None):
        """ Evaluates a given orbital in the point (x,y,z). Should be able to accept ndarrays as input."""
        # Load the coeffs. and multiply them on the contracted basis sets
        if orbital_nr is None: 
            orbital_nr = self.HOMO
        
        MO_coeffs = self.saved_orbitals[orbital_nr][3]
        return np.sum([MO_c * basis_func.eval(x, y, z) for MO_c, basis_func in zip(MO_coeffs, self.basis_funcs)], axis=0)

    def eval_orbital_spherical(self, r, theta, phi, orbital_nr=None):
        """ Evaluates a given orbital given spherical coordinates (r, theta, phi) """
        if orbital_nr is None: 
            orbital_nr = self.HOMO
        
        x = r * np.sin(theta) * np.cos(phi)
        y = r * np.sin(theta) * np.sin(phi)
        z = r * np.cos(theta)
        return self.eval_orbital(x, y, z, orbital_nr)

