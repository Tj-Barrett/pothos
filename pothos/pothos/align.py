'''
Crystal Finder

"Crystal Growth in Polyethylene by Molecular Dynamics: The Crystal Edge and Lame
llar Thickness" Macromolecules 51, 13, 2018
https://doi.org/10.1021/acs.macromol.8b00857

See for more info:

Ryan Davidson
https://scrunts23.medium.com/dbscan-algorithm-from-scratch-in-python-475b82e0571c

Erik Lindernoren
https://github.com/eriklindernoren/ML-From-Scratch

Jake VanderPlas
https://gist.github.com/jakevdp/5216193

Tuukka Verho
https://doi.org/10.1021/acs.macromol.8b00857
'''

from pothos.vdbscan import VDBSCAN

class align(object):
    def __init__(self, filename):
        ####################################
        # Validate File and create filenames
        ####################################
        if isinstance(filename, str):
            self.filename = filename
        else:
            print("Filename is not a string")
            exit()

        # remove .dump
        self.prefix = filename[0:len(filename)-5]
        # names for all outputs
        self.cntname = self.prefix+'_cnt_end_xyz.dump'
        self.totaloutname = self.prefix+'_PointCloud.dump'
        self.vectorname = self.prefix+'_Cry.directions'

        #----------------------------------
        # Defaults
        #----------------------------------
        self.atom_types=False
        self.ncpus = 1
        # orientation
        self.align_type = 'verho'
        self.length_coeff=7
        # Verho
        self.verho_cutoff=0.9
        # Legendre
        self.legendre_cutoff=0.9
        # Clustering
        self.min_length=8
        self.cluster_method = 'P2'
        self.neighlist_method='BallTree'
        self.eps=1.1
        self.lam=0.9
        self.leaf_size=30
        # Post Process
        self.min_pts=8
        self.second_filt=True
        self.crystalmin=100
        self.window='hann'
        # Other
        self.coloring = 'chain_avg'
        self.verbose = False
        self.clustered = False
        self.stats = []

        self.custom_str = None
        self.just_atoms = True
        self.timesteps = None
        self.series = True

    def _variable_check(self):
        ####################################
        # Check number of monomers and polymers
        ####################################
        #
        # Atom types to select
        #
        if isinstance(self.atom_types, (int,list) ) and self.atom_types is not False:
            if isinstance(self.atom_types, int ):
                self.atom_types=[self.atom_types]
            else:
                self.atom_types=self.atom_types
        else:
            self.atom_types=False

        #
        # Parallel
        #

        if isinstance(self.ncpus, int):
            self.ncpus = self.ncpus
        else:
            print("ncpus is not an integer")
            exit()

        #
        # Is data a series of events/steps
        #

        if isinstance(self.series, bool):
            if not self.series:
                self.n_steps = 1
        else:
            print("Data series variable is not a boolean")
            exit()

        #
        # custom string
        #

        if isinstance(self.custom_str, str):
            self.custom_str = '-'+self.custom_str
        else:
            self.custom_str = ''

        #
        # Just these atoms
        #

        if isinstance(self.just_atoms, bool):
            if self.just_atoms:
                self.just_atoms = True
            elif not self.just_atoms:
                self.just_atoms = False
            else:
                print("just_atoms is not a boolean")
                exit()

        #
        # Timesteps
        #

        if isinstance(self.timesteps, (int,list) ) and self.timesteps is not False:
            if isinstance(self.timesteps, int ):
                self.timesteps=[self.timesteps]
            else:
                self.timesteps=self.timesteps
        else:
            self.timesteps=False

        ####################################
        # Orientation Parameter determination
        ####################################
        if self.align_type == 'verho':
            ####################################
            # Verho Equation Information
            ####################################
            #
            # Verho Coefficient
            #
            if self.length_coeff < 0 or self.length_coeff == 0:
                print(
                    "Length coefficient cannot be negative or equal to zero")
                exit()
            elif not isinstance(self.length_coeff, int):
                print("Length coefficient is not an integer")
                exit()

            #
            # Verho Cutoff Lambda
            #
            if not 0.0 < self.verho_cutoff < 1.0:
                print("Verho Cutoff must be between 0 and 1")
                exit()

        elif self.align_type == 'P2' or self.align_type == 'P4' or self.align_type == 'P6':
            ####################################
            # legendre orientation Parameters
            ####################################
            if self.length_coeff < 0 or self.length_coeff == 0:
                print(
                    "Length coefficient cannot be negative or equal to zero")
                exit()
            elif not isinstance(self.length_coeff, int):
                print("Length coefficient is not an integer")
                exit()

            if self.legendre_cutoff < -0.5 or self.legendre_cutoff > 1.0:
                print(
                    "Legendre cutoff coefficient cannot be less than -0.5, or above 1.0 ")
                exit()

        else:
            warning_str  = "Orientation parameter not supported. "
            warning_str += "Current methods : \n Verho :\'verho\'\n "
            warning_str += "Legendre P2 : \'P2\' \n "
            warning_str += "Legendre P4 : \'P4\' \n "
            warning_str += "Legendre P6 : \'P6\' \n "

            print(warning_str)
            exit()

        ####################################
        # Nearest neighbors and Clustering
        ####################################
        #
        # Clustering Method
        #
        if self.cluster_method == 'P2' \
            or self.cluster_method == 'cosine':

            self.cluster_method = self.cluster_method
        else:
            print('Unsupported clustering method type')
            exit()

        if self.neighlist_method == 'BallTree' \
            or self.neighlist_method == 'KDTree' \
            or self.neighlist_method == 'Brute':

            self.neighlist_method = self.neighlist_method
        else:
            print('Unsupported neighborlist method type')
            exit()

        #
        # Epsilon Value for Distance
        #
        if self.eps < 0.0 or self.eps == 0.0:
            print("Epsilon cannot be negative or equal to zero")
            exit()

        #
        # Lambda Value for Alignment
        #
        if not 0.0 < self.lam < 1.0:
            print("Lambda Cutoff must be between 0 and 1")
            exit()

        #
        # Leaf Size for BallTree
        #
        if self.leaf_size < 0 or self.leaf_size == 0:
            print("Leaf size cannot be negative or equal to zero")
            exit()
        elif not isinstance(self.leaf_size, int):
            print("Leaf size is not an integer")
            exit()

        #
        # Minimum sampling points
        #
        if self.min_pts < 1:
            print("Minimum samples must be greater than zero")
            exit()

        ####################################
        # Post Processing
        ####################################
        #
        # Minimum Length
        #
        if self.min_length < 1:
            print("Minimum samples must be greater than zero")
            exit()
        elif not isinstance(self.min_length, int):
            print("Minimum length filter is not an integer")
            exit()

        #
        # Secondary Filter Boolean
        #
        if self.second_filt > 1:
            print("Second filter application is not a boolean")
            exit()
        elif not isinstance(self.second_filt, int):
            print("Second filter application is not a boolean ")
            exit()

        #
        # Crystal Size Minimum
        #
        if not self.crystalmin > 0:
            print(
                "Crystal filter cutoff cannot be negative or equal to zero")
            exit()
        elif not isinstance(self.crystalmin, int):
            print("Crystal filter cutoff is not an integer")
            exit()

        #
        # Windowing Method
        #
        if  self.window == 'tukey' or self.window == 'nuttall' \
            or self.window == 'parzen' or self.window == 'hann' \
            or self.window == 'none':
            self.window_warning_print = False
        else:
            print('Unsupported window type')
            exit()

        #
        # Verbose
        #
        if not isinstance(self.verbose, bool):
            print("Verbose flag needs to be a boolean")
            exit()

        #
        # Colorcoding
        #
        if self.coloring == 'chain_avg' or self.coloring == 'bead' \
            or self.coloring == 'size' or self.coloring == 'cluster_avg' \
            or self.coloring == 'herman':
            self.coloring = [self.coloring]
        else:
            print('Unsupported coloring type')
            exit()

    '''
    #########################################
    # Internal Facing
    #########################################
    '''
    def avg_vectors(self, vectorlist, vectorarray):
        '''
        Finding each passing section of chain and average

        Inputs :
        vectorlist - list of booleans indicating true/false of crystallinity of bead
        vectorarray - array of vector information for each bead

        Returns :
        locations - x,y,z location data of atoms
        chords - chord vector of each atom, ie (-k,k) vector
        chain_id - chain index number

        Global :
        self.chain_averaged = average chord vector for atoms, zeroed (0,0,0) for amorphous
        '''
        from numpy import sqrt, array, sum, zeros
        try:
            from scipy.signal import hann, tukey, parzen, nuttall
        except:
            from scipy.signal.windows import hann, tukey, parzen, nuttall
        finally:
            print('Scipy seems to have changed things, try an older version (1.16.1 should be ok')
            exit()

        #
        # Segment Averaging - averages across crystalline chain segments
        #
        def seg_avg(cc_x,cc_y,cc_z):
            sum_x = 0
            sum_y = 0
            sum_z = 0

            #
            # Windowing - weights bead vector values towards center of chain
            #
            if self.window == 'hann':
                wn = hann(len(cc_x))
            elif self.window == 'nuttall':
                wn = nuttall(len(cc_x))
            elif self.window == 'parzen':
                wn = parzen(len(cc_x))
            elif self.window == 'tukey':
                wn = tukey(len(cc_x))
            elif self.window == 'none':
                wn = [1]*len(cc_x)

            if self.window == 'nuttall' and len(cc_x) < 5:
                wn = [1]*len(cc_x)
                if not self.window_warning_print:
                    print('\n Warning: Nuttall requires a minimum window size of 4. Check results against another window to confirm behavior.\n')
                    self.window_warning_print = True

            # weight vectors in each direction
            for k in range(len(cc_x)):
                sum_x = sum_x + cc_x[k]*wn[k]
                sum_y = sum_y + cc_y[k]*wn[k]
                sum_z = sum_z + cc_z[k]*wn[k]

            # divide by sum of weighting funtion
            if sum(wn) > 0:
                chain_x = sum_x/sum(wn)
                chain_y = sum_y/sum(wn)
                chain_z = sum_z/sum(wn)
            else:
                # handle double_scalars issues
                chain_x = sum_x/1.
                chain_y = sum_y/1.
                chain_z = sum_z/1.

            return chain_x, chain_y, chain_z

        lenvl = len(vectorlist)
        chain_id = zeros((lenvl,1))
        chain_vector = zeros((lenvl,3))
        chain_idx  = []

        n = 0

        cc_x = []
        cc_y = []
        cc_z = []

        for i in range(self.polymers):
            for j in range(self.monomers):
                idx = self.monomers*i+j

                # first point
                if vectorlist[idx] and idx == 0:
                    cc_x.append(vectorarray[idx][0])
                    cc_y.append(vectorarray[idx][1])
                    cc_z.append(vectorarray[idx][2])
                    chain_idx.append(idx)
                    # last point in chain
                elif vectorlist[idx] and j == self.monomers-1:
                    cc_x.append(vectorarray[idx][0])
                    cc_y.append(vectorarray[idx][1])
                    cc_z.append(vectorarray[idx][2])
                    chain_idx.append(idx)

                    chain_x, chain_y, chain_z = seg_avg(cc_x,cc_y,cc_z)

                    for k in chain_idx:
                        chain_vector[k,0] = chain_x
                        chain_vector[k,1] = chain_y
                        chain_vector[k,2] = chain_z
                        chain_id[k] = n

                    cc_x = []
                    cc_y = []
                    cc_z = []
                    chain_idx = []
                    n = n + 1

                elif vectorlist[idx]:
                    cc_x.append(vectorarray[idx][0])
                    cc_y.append(vectorarray[idx][1])
                    cc_z.append(vectorarray[idx][2])
                    chain_idx.append(idx)

                    chain_x, chain_y, chain_z = seg_avg(cc_x,cc_y,cc_z)

                    for k in chain_idx:
                        chain_vector[k][0] = chain_x
                        chain_vector[k][1] = chain_y
                        chain_vector[k][2] = chain_z
                        chain_id[k] = n

                    cc_x = []
                    cc_y = []
                    cc_z = []
                    chain_idx = []
                    n = n + 1

                # general passing
                elif vectorlist[idx] and self.id[idx] == self.id[idx+1]:
                    cc_x.append(vectorarray[idx][0])
                    cc_y.append(vectorarray[idx][1])
                    cc_z.append(vectorarray[idx][2])
                    chain_idx.append(idx)

                # not passing
                else:
                    if len(cc_x) > 0:
                        chain_x, chain_y, chain_z = seg_avg(cc_x,cc_y,cc_z)

                        for k in chain_idx:
                            chain_vector[k][0] = chain_x
                            chain_vector[k][1] = chain_y
                            chain_vector[k][2] = chain_z
                            chain_id[k] = n

                        cc_x = []
                        cc_y = []
                        cc_z = []
                        chain_idx = []
                        n = n + 1

        chain_vector = array(chain_vector)
        chain_id = array(chain_id)

        locations = []
        chords = []
        ca = []
        for i in range(len(vectorlist)):
            if vectorlist[i]:
                locations.append([self.x[i], self.y[i], self.z[i]])
                chords.append(
                    [chain_vector[i][0], chain_vector[i][1], chain_vector[i][2]])
                ca.append(
                    [chain_vector[i][0], chain_vector[i][1], chain_vector[i][2]])
            else:
                ca.append([0, 0, 0])

        locations = array(locations)
        chords = array(chords)

        self.chain_averaged = array(ca)

        return locations, chords, chain_id

    def data_parse(self, filename, series, i = 0):
        '''
        Reading data from file

        Inputs :
        filename - string of file to be read

        Returns :
        polymers - int of polymers in system
        monomers - int of monomers in system

        global variables
        self.atom - atom numbers
        self.id - atom ids
        self.moltype - atom type
        self.x - atom x location
        self.y - atom y location
        self.z - atom z location
        ***
        self.xu - atom x unwrapped location
        self.yu - atom y unwrapped location
        self.zu - atom z unwrapped location
        *** or
        self.ix - atom x periodic index
        self.iy - atom y periodic index
        self.iz - atom z periodic index
        '''
        from numpy import array, sqrt, dot, zeros
        from rich.progress import track
        import mmap
        ####################################
        # Read Data
        ####################################
        if series:
            if i == 0:
                #https://stackoverflow.com/questions/845058/how-to-get-the-line-count-of-a-large-file-cheaply-in-python
                with open(filename, 'r') as f:
                    buf = mmap.mmap(f.fileno(), 0, prot=mmap.PROT_READ)
                    lines = 0
                    readline = buf.readline
                    while readline():
                        lines += 1
                self.n_steps = int(lines/(self.n_atoms+9))

            if i == 0:
                f = open(filename, 'r')
                data = f.readlines()[8]
                f.close()

                data = data.split()

                if data[8] == 'xu':
                    self.data_condition = 'unwrapped'
                elif data[8] == 'ix':
                    self.data_condition = 'index'
                else:
                    print(
                            f"Check data file, 7th column is not unwrapped or periodic index.")
                    exit()

                f = open(filename, 'r')
                data = f.readlines()[i*self.n_atoms+9:(i+1)*self.n_atoms+9]
                f.close()

                f = open(self.filename, 'r')
                box = f.readlines()[5:8]
                f.close()
            else:
                f = open(filename, 'r')
                data = f.readlines()[i*(self.n_atoms+9)+9:(i+1)*(self.n_atoms+9)]
                f.close()

                f = open(self.filename, 'r')
                box = f.readlines()[i*(self.n_atoms+9)+5:i*(self.n_atoms+9)+8]
                f.close()

            if len(box[0].split()) == 2:
                # regular box
                xlo, xhi = box[0].split()
                ylo, yhi = box[1].split()
                zlo, zhi = box[2].split()

            elif len(box[0].split()) == 3:
                # triclinic box
                xlo, xhi, _xt= box[0].split()
                ylo, yhi, _yt= box[1].split()
                zlo, zhi, _zt= box[2].split()

            xlen = float(xhi)-float(xlo)
            ylen = float(yhi)-float(ylo)
            zlen = float(zhi)-float(zlo)

            # atom data
            lendata = len(data)
            atom = zeros((lendata))
            id   = zeros((lendata))
            type = zeros((lendata))
            x    = zeros((lendata))
            y    = zeros((lendata))
            z    = zeros((lendata))
            xu   = zeros((lendata))
            yu   = zeros((lendata))
            zu   = zeros((lendata))
            ix   = zeros((lendata))
            iy   = zeros((lendata))
            iz   = zeros((lendata))

            for i in track(range(lendata),  description= "{:<25}".format('Reading Data ...')):
                idx = data[i].split()

                atom[i] = int(idx[0])
                id[i] = int(idx[1])
                type[i] = int(idx[2])
                x[i] = float(idx[3])
                y[i] = float(idx[4])
                z[i] = float(idx[5])
                if self.data_condition == 'unwrapped':
                    xu[i] = float(idx[6])
                    yu[i] = float(idx[7])
                    zu[i] = float(idx[8])
                elif self.data_condition == 'index':
                    xu[i] = float(idx[6])*xlen+float(idx[3])
                    yu[i] = float(idx[7])*ylen+float(idx[4])
                    zu[i] = float(idx[8])*zlen+float(idx[5])

                    ix[i] = float(idx[6])
                    iy[i] = float(idx[7])
                    iz[i] = float(idx[8])


            atom = array(atom)
            id = array(id)
            moltype = array(type)
            x = array(x)
            y = array(y)
            z = array(z)
            xu = array(xu)
            yu = array(yu)
            zu = array(zu)
            ix = array(ix)
            iy = array(iy)
            iz = array(iz)

            if self.atom_types is not False and sum(atom) == 0:
                print(
                        f"No atoms found. Check atom types : {self.atom_types}")
                exit()

            # Ordering
            id, atom, moltype, x, y, z, xu, yu, zu, ix, iy, iz = map(list, zip(*sorted(zip(id, atom, moltype, x, y, z, xu, yu, zu, ix, iy, iz), reverse=False)))


            if isinstance(self.atom_types, (int,list) ) and self.atom_types is not False:

                _atom    = []
                _id      = []
                _moltype = []
                _x       = []
                _y       = []
                _z       = []
                _xu      = []
                _yu      = []
                _zu      = []
                _ix      = []
                _iy      = []
                _iz      = []

                x_atom    = []
                x_id      = []
                x_moltype = []
                x_x       = []
                x_y       = []
                x_z       = []
                x_xu      = []
                x_yu      = []
                x_zu      = []
                x_ix      = []
                x_iy      = []
                x_iz      = []

                for i in track(range(lendata),  description= "{:<25}".format('Cleaning Data ...')):
                    if int(moltype[i]) in self.atom_types:
                        _atom.append(atom[i])
                        _id.append(id[i])
                        _moltype.append(moltype[i])
                        _x.append(x[i])
                        _y.append(y[i])
                        _z.append(z[i])
                        _xu.append(xu[i])
                        _yu.append(yu[i])
                        _zu.append(zu[i])
                        _ix.append(iz[i])
                        _iy.append(iy[i])
                        _iz.append(iz[i])

                    else:
                        x_atom.append(atom[i])
                        x_id.append(id[i])
                        x_moltype.append(moltype[i])
                        x_x.append(x[i])
                        x_y.append(y[i])
                        x_z.append(z[i])
                        x_xu.append(xu[i])
                        x_yu.append(yu[i])
                        x_zu.append(zu[i])
                        x_ix.append(iz[i])
                        x_iy.append(iy[i])
                        x_iz.append(iz[i])


                # place in global variables
                self.atom = _atom
                self.id = _id
                self.moltype = _moltype
                self.x = _x
                self.y = _y
                self.z = _z
                self.xu = _xu
                self.yu = _yu
                self.zu = _zu
                self.ix = _ix
                self.iy = _iy
                self.iz = _iz

                self.x_atom = x_atom
                self.x_id = x_id
                self.x_moltype = x_moltype
                self.x_x = x_x
                self.x_y = x_y
                self.x_z = x_z
                self.x_xu = x_xu
                self.x_yu = x_yu
                self.x_zu = x_zu
                self.x_ix = x_ix
                self.x_iy = x_iy
                self.x_iz = x_iz

                polymers = int(max(_id)+1 - min(_id))
                monomers = int(len(_atom)/polymers)

            else:
                # place in global variables
                self.atom = atom
                self.id = id
                self.moltype = moltype
                self.x = x
                self.y = y
                self.z = z
                self.xu = xu
                self.yu = yu
                self.zu = zu
                self.ix = ix
                self.iy = iy
                self.iz = iz

                polymers = int(max(id)+1 - min(id))
                monomers = int(len(atom)/polymers)

            return polymers, monomers

        else:
            try:
                f = open(filename, 'r')
                data = f.readlines()[8]
                f.close()

                data = data.split()

                if data[8] == 'xu':
                    self.data_condition = 'unwrapped'
                elif data[8] == 'ix':
                    self.data_condition = 'index'
                else:
                    print(
                            f"Check data file, 7th column is not unwrapped or periodic index.")
                    exit()

                f = open(filename, 'r')
                data = f.readlines()[i*self.n_atoms+9:(i+1)*self.n_atoms+9]
                f.close()

                f = open(self.filename, 'r')
                box = f.readlines()[i*(self.n_atoms+9)+5:i*(self.n_atoms+9)+8]
                f.close()

            except:
                print("Data is an empty array")
                exit()

            if len(box[0].split()) == 2:
                # regular box
                xlo, xhi = box[0].split()
                ylo, yhi = box[1].split()
                zlo, zhi = box[2].split()

            elif len(box[0].split()) == 3:
                # triclinic box
                xlo, xhi, _xt= box[0].split()
                ylo, yhi, _yt= box[1].split()
                zlo, zhi, _zt= box[2].split()

            xlen = float(xhi)-float(xlo)
            ylen = float(yhi)-float(ylo)
            zlen = float(zhi)-float(zlo)

            # atom data
            lendata = len(data)
            atom = zeros((lendata))
            id   = zeros((lendata))
            type = zeros((lendata))
            x    = zeros((lendata))
            y    = zeros((lendata))
            z    = zeros((lendata))
            xu   = zeros((lendata))
            yu   = zeros((lendata))
            zu   = zeros((lendata))
            ix   = zeros((lendata))
            iy   = zeros((lendata))
            iz   = zeros((lendata))

            for i in track(range(lendata),  description= "{:<25}".format('Reading Data ...')):
                idx = data[i].split()

                atom[i] = int(idx[0])
                id[i] = int(idx[1])
                type[i] = int(idx[2])
                x[i] = float(idx[3])
                y[i] = float(idx[4])
                z[i] = float(idx[5])
                if self.data_condition == 'unwrapped':
                    xu[i] = float(idx[6])
                    yu[i] = float(idx[7])
                    zu[i] = float(idx[8])
                elif self.data_condition == 'index':
                    xu[i] = float(idx[6])*xlen+float(idx[3])
                    yu[i] = float(idx[7])*ylen+float(idx[4])
                    zu[i] = float(idx[8])*zlen+float(idx[5])

                    ix[i] = float(idx[6])
                    iy[i] = float(idx[7])
                    iz[i] = float(idx[8])


            atom = array(atom)
            id = array(id)
            moltype = array(type)
            x = array(x)
            y = array(y)
            z = array(z)
            xu = array(xu)
            yu = array(yu)
            zu = array(zu)
            ix = array(ix)
            iy = array(iy)
            iz = array(iz)

            if self.atom_types is not False and sum(atom) == 0:
                print(
                        f"No atoms found. Check atom types : {self.atom_types}")
                exit()

            # Ordering
            id, atom, moltype, x, y, z, xu, yu, zu, ix, iy, iz = map(list, zip(*sorted(zip(id, atom, moltype, x, y, z, xu, yu, zu, ix, iy, iz), reverse=False)))


            if isinstance(self.atom_types, (int,list) ) and self.atom_types is not False:

                _atom    = []
                _id      = []
                _moltype = []
                _x       = []
                _y       = []
                _z       = []
                _xu      = []
                _yu      = []
                _zu      = []
                _ix      = []
                _iy      = []
                _iz      = []

                x_atom    = []
                x_id      = []
                x_moltype = []
                x_x       = []
                x_y       = []
                x_z       = []
                x_xu      = []
                x_yu      = []
                x_zu      = []
                x_ix      = []
                x_iy      = []
                x_iz      = []

                for i in track(range(lendata),  description= "{:<25}".format('Cleaning Data ...')):
                    if int(moltype[i]) in self.atom_types:
                        _atom.append(atom[i])
                        _id.append(id[i])
                        _moltype.append(moltype[i])
                        _x.append(x[i])
                        _y.append(y[i])
                        _z.append(z[i])
                        _xu.append(xu[i])
                        _yu.append(yu[i])
                        _zu.append(zu[i])
                        _ix.append(iz[i])
                        _iy.append(iy[i])
                        _iz.append(iz[i])

                    else:
                        x_atom.append(atom[i])
                        x_id.append(id[i])
                        x_moltype.append(moltype[i])
                        x_x.append(x[i])
                        x_y.append(y[i])
                        x_z.append(z[i])
                        x_xu.append(xu[i])
                        x_yu.append(yu[i])
                        x_zu.append(zu[i])
                        x_ix.append(iz[i])
                        x_iy.append(iy[i])
                        x_iz.append(iz[i])


                # place in global variables
                self.atom = _atom
                self.id = _id
                self.moltype = _moltype
                self.x = _x
                self.y = _y
                self.z = _z
                self.xu = _xu
                self.yu = _yu
                self.zu = _zu
                self.ix = _ix
                self.iy = _iy
                self.iz = _iz

                self.x_atom = x_atom
                self.x_id = x_id
                self.x_moltype = x_moltype
                self.x_x = x_x
                self.x_y = x_y
                self.x_z = x_z
                self.x_xu = x_xu
                self.x_yu = x_yu
                self.x_zu = x_zu
                self.x_ix = x_ix
                self.x_iy = x_iy
                self.x_iz = x_iz

                polymers = int(max(_id)+1 - min(_id))
                monomers = int(len(_atom)/polymers)

            else:
                # place in global variables
                self.atom = atom
                self.id = id
                self.moltype = moltype
                self.x = x
                self.y = y
                self.z = z
                self.xu = xu
                self.yu = yu
                self.zu = zu
                self.ix = ix
                self.iy = iy
                self.iz = iz

                polymers = int(max(id)+1 - min(id))
                monomers = int(len(atom)/polymers)

            return polymers, monomers

    def length_filter(self, vectorlist):
        '''
        Length filter on chain segments. Counts the number of consecutive beads
            passing (up to ends). If its above  the minimum count, it passes.

        Inputs:
        vectorlist - list of booleans for true/false of crystallinity per bead

        Returns:
        vectorlist - list of booleans for true/false of crystallinity per bead.
        '''
        #########################################
        # Filtering chains to make sure they have a minimum length
        #########################################
        from numpy import array, zeros
        from joblib import Parallel, delayed
        from rich.progress import track
        # Enforce Min length of N RUs
        lenvl = len(vectorlist)
        newvl = [False]*lenvl
        n = 0

        for i in track(range(self.polymers),  description= "{:<25}".format('Enforcing Length ...')):
            for j in range(self.monomers):
                idx = self.monomers*i+j

                if vectorlist[idx] and j < self.monomers-1:
                    # check if true
                    n = n+1
                elif vectorlist[idx] and j == self.monomers-1:
                    if n > self.min_length:
                        # Set true if passes
                        for k in range(0, n+1):
                            newvl[idx-k] = True
                    # if False
                    n = 0
                else:
                    if n > self.min_length:
                        # Set true if passes
                        for k in range(0, n+1):
                            newvl[idx-k] = True
                    # if False
                    n = 0

        del vectorlist
        vectorlist = array(newvl)

        return vectorlist

    def length_stats(self, vectorlist):
        '''
        Length stats chain segments. Counts the number of consecutive beads
            passing (up to ends). If its above  the minimum count, it passes.

        Inputs:
        vectorlist - list of booleans for true/false of crystallinity per bead

        Returns:
        vectorlist - list of booleans for true/false of crystallinity per bead.
        '''
        #########################################
        # Filtering chains to make sure they have a minimum length
        #########################################
        from numpy import array, zeros
        from joblib import Parallel, delayed
        from rich.progress import track
        # Enforce Min length of N RUs
        n = 0

        segments = []

        for i in track(range(self.polymers),  description= "{:<25}".format('Length Stats ...')):
            for j in range(self.monomers):
                idx = self.monomers*i+j

                if vectorlist[idx] and j < self.monomers-1:
                    # check if true
                    n = n+1
                elif vectorlist[idx] and j == self.monomers-1:
                    if n > self.min_length:
                        # Set true if passes
                        segments.append(n)
                    # if False
                    n = 0
                else:
                    if n > self.min_length:
                        # Set true if passes
                        segments.append(n)
                    # if False
                    n = 0

        return array(segments)

    def type_filter(self, vectorlist, labels):
        '''
        Type filter counts the number of beads per crystal label to make sure
            there is a minimum number of beads contributing to each crystal

        Input:
        vectorlist - list of booleans for true/false of bead crystallinity
        labels - crystal identification number for each bead

        Returns:
        moltype - crystal identification number now applied to the moltype for
            visualization
        '''

        from numpy import array
        #########################################
        # Filter out based on type
        #########################################
        lenvl = len(vectorlist)
        moltype = [0]*lenvl
        n = 0
        for i in range(lenvl):

            if vectorlist[i]:
                moltype[i] = int(labels[n])
                n = n+1

        checktype = [0]*lenvl

        for i in range(lenvl):

            if vectorlist[i] and moltype[i] > 0:
                checktype[i] = moltype[i]
            else:
                vectorlist[i] = False
                checktype[i] = -1

        moltype = array(checktype)

        return moltype

    def refine(self, vectorlist, vectorarray):
        '''
        Refining verho equation results. Sometimes if (-k,+k) is too large, it
            will classify aligned beads as amorphous. This fixes ~some~ of them

        Input :
        vectorlist - list of booleans for true/false of crystallinity for beads
        vectorarray - array of chord vectors for the beads

        Returns :
        vectorlist - list of booleans for true/false of crystallinity for beads
        vectorarray - array of chord vectors for the beads with updated chain
            vectors
        '''

        #########################################
        # Refine Ends
        #########################################
        from numpy import array, zeros, sqrt, dot
        # Enforce Min length of N RUs
        lenvl = len(vectorlist)
        newvl = [False]*lenvl
        newva = zeros((lenvl,3))
        n = 0

        for i in range(self.polymers):
            for j in range(self.monomers):
                idx = self.monomers*i+j

                if vectorlist[idx]:
                    # check if true
                    newvl[idx] = True
                    ux = vectorarray[idx][0]
                    uy = vectorarray[idx][1]
                    uz = vectorarray[idx][2]
                    newva[idx,0] = ux
                    newva[idx,1] = uy
                    newva[idx,2] = uz
                else:
                    # change to refine length
                    if newvl[idx-1] and newvl[idx+1]:
                        # to iterate from -k to k, needs to be -k to k+1
                        # vector
                        vx = self.xu[i+1] - self.xu[i-1]
                        vy = self.yu[i+1] - self.yu[i-1]
                        vz = self.zu[i+1] - self.zu[i-1]
                        # unit vector
                        ux = vx/sqrt(vx**2+vy**2+vz**2)
                        uy = vy/sqrt(vx**2+vy**2+vz**2)
                        uz = vz/sqrt(vx**2+vy**2+vz**2)

                        if ux < 0:
                            ux = -1*ux
                        if uy < 0:
                            uy = -1*uy
                        if uz < 0:
                            uz = -1*uz

                        newvl[idx] = True
                        newva[idx,0] = ux
                        newva[idx,1] = uy
                        newva[idx,2] = uz

                    elif newvl[idx-1]:
                        # to iterate from -k to k, needs to be -k to k+1
                        # vector
                        vx = self.xu[idx+1] - self.xu[idx-1]
                        vy = self.yu[idx+1] - self.yu[idx-1]
                        vz = self.zu[idx+1] - self.zu[idx-1]
                        # unit vector
                        ux = vx/sqrt(vx**2+vy**2+vz**2)
                        uy = vy/sqrt(vx**2+vy**2+vz**2)
                        uz = vz/sqrt(vx**2+vy**2+vz**2)

                        # avg d
                        lam = dot(vectorarray[idx-1], [ux,uy,uz])

                        if ux < 0:
                            ux = -1*ux
                        if uy < 0:
                            uy = -1*uy
                        if uz < 0:
                            uz = -1*uz

                        if lam > self.verho_cutoff:
                            newvl[idx] = True
                            newva[idx,0] = ux
                            newva[idx,1] = uy
                            newva[idx,2] = uz
                        else:
                            newvl[idx] = False
                            newva[idx,0] = ux
                            newva[idx,1] = uy
                            newva[idx,2] = uz
                    elif newvl[idx+1]:
                        # to iterate from -k to k, needs to be -k to k+1
                        # vector
                        vx = self.xu[idx+1] - self.xu[idx-1]
                        vy = self.yu[idx+1] - self.yu[idx-1]
                        vz = self.zu[idx+1] - self.zu[idx-1]
                        # unit vector
                        ux = vx/sqrt(vx**2+vy**2+vz**2)
                        uy = vy/sqrt(vx**2+vy**2+vz**2)
                        uz = vz/sqrt(vx**2+vy**2+vz**2)

                        # avg d
                        lam = dot(vectorarray[idx+1], [ux,uy,uz])

                        if ux < 0:
                            ux = -1*ux
                        if uy < 0:
                            uy = -1*uy
                        if uz < 0:
                            uz = -1*uz

                        if lam > self.verho_cutoff:
                            newvl[idx] = True
                            newva[idx,0] = ux
                            newva[idx,1] = uy
                            newva[idx,2] = uz
                        else:
                            newvl[idx] = False
                            newva[idx,0] = ux
                            newva[idx,1] = uy
                            newva[idx,2] = uz

                    # if False
                    n = 0

        return vectorlist, vectorarray

    def renumber(self,moltype, vectorlist, past_labels=None):
        '''
        Renumbering crystal list to start counting at 2 consecutively

        Input :
        moltype - type of crystal
        vectorlist - list of booleans controlling true/false of crystal per bead
        past_labels - in time series, if past labels are available use those

        Returns:
        moltype - type of crystal renumbered
        uniques - int of unique crystal types
        new_labels - list from -1 to max crystal types
        '''

        from numpy import unique, array, arange, zeros, max
        #########################################
        # Renumber vectorlist
        #########################################
        lenmol = len(moltype)
        checktype = [0]*lenmol

        # if past_labels is None:
        uniques = list(unique(moltype))
        re_vector = zeros((len(uniques[2:]),2))
        n = 0
        # start crystal numbers at 2
        for i in range(lenmol):
            if vectorlist[i] and moltype[i] > 1:
                checktype[i] = uniques.index(moltype[i])+1
            else:
                vectorlist[i] = False
                checktype[i] = -1

        # for each crystal, if previous crystal, use that label
        # moltype = array(checktype)
        # if past_labels not None:
        #     for un in uniques:
        #         for i in range(lenmol):
        #             if moltype[i]
        #              checktype[i] = un

        moltype = array(checktype)
        len_uniques = len(uniques)
        new_labels = arange(-1,len_uniques)

        return moltype, len_uniques, new_labels, moltype

    def quantity_filter(self, vectorlist, vectorarray, moltype, labels):
        '''
        Like type filter, checks the total beads contributing to a crystal

        Inputs :
        vectorlist - boolean true/false of a beads crystal status
        vectorarray - array of x,y,z data for each bead
        moltype - list of crystal type for each bead
        labels - labels of all crystals

        Returns :
        vectorlist - boolean true/false of a beads crystal status
        moltype - list of crystal type for each bead
        avgvector - array of vector components for

        Global:
        self.klk - array of vectors that contributes to the coloring for ovito
        '''

        #########################################
        # Filter out based on quantity of type
        #########################################
        from joblib import Parallel, delayed
        from rich.progress import track
        from numpy import array, sqrt, min, max, zeros, arange, arccos, cos
        import numba

        #
        # Average vector determination for each crystal. Applies to centroid
        #
        # @numba.jit(nopython=True,cache=True)
        # def aveVector(i, lst, vect, type):
        #     sum_x = 0
        #     sum_y = 0
        #     sum_z = 0
        #
        #     for j in range(len(lst)):
        #         if (lst[j] == type):
        #             sum_x = sum_x + vect[j][0]
        #             sum_y = sum_y + vect[j][1]
        #             sum_z = sum_z + vect[j][2]
        #     if (sum_x and sum_y and sum_z) == 0:
        #         out_x = 0
        #         out_y = 0
        #         out_z = 0
        #     else:
        #         out_x = sum_x/sqrt(sum_x**2+sum_y**2+sum_z**2)
        #         out_y = sum_y/sqrt(sum_x**2+sum_y**2+sum_z**2)
        #         out_z = sum_z/sqrt(sum_x**2+sum_y**2+sum_z**2)
        #
        #     return [type, out_x, out_y, out_z]
        #

        label_list = arange(min(labels), max(labels), 1)

        #########################################
        # Counting Beads
        #########################################
        # https://www.geeksforgeeks.org/python-count-occurrences-element-list/

        @numba.jit(nopython=True,cache=True)
        def countX(x,lst):
            count = 0
            for ele in lst:
                if (ele == x):
                    count = count+1

            return [x, count]

        pertype = []
        for i in track(label_list,  description= "{:<25}".format('Counting Beads ...')):
            pertype.append(countX(i,moltype))

        pertype = array(sorted(pertype, key=lambda x: x[0]))

        #########################################
        # Vector Listing
        #########################################

        range_vectorlist = range(len(vectorlist))

        @numba.jit(cache=True)
        def vectorlist_sort(i, pertype, type, crystalmin):
            for _type, _count in pertype:
                if type == _type and _count < crystalmin:
                    return -1
            else:
                return type

        int_moltype = Parallel(n_jobs=self.ncpus, require='sharedmem',batch_size='auto')(
            delayed(vectorlist_sort)(i, pertype, moltype[i], self.crystalmin) for i in track(range_vectorlist,  description= "{:<25}".format('Molecule Sorting ...')))

        # int_moltype = []
        # for i in track(range_vectorlist,  description= "{:<25}".format('Molecule Sorting ...')):
        #     int_moltype.append(vectorlist_sort(i, pertype, moltype[i], self.crystalmin))

        # int_moltype = sorted(int_moltype, key=lambda x: x[0])
        moltype = int_moltype
        del int_moltype

        moltype = array(moltype)

        # Vectorlist Sorting
        for i in moltype:
            if int(i) > 1:
                vectorlist[i] = True
            else:
                vectorlist[i] = False
        #########################################
        # Vector Array Averaging
        #########################################
        labs = arange(min(labels), max(labels)+1)
        len_labs = len(labs)
        # self.int_avgvector = [[0,0,0,0]]*len_labs

        @numba.jit(nopython=True,cache=True)
        def aveVector(i, lst, vect, type):
            sum_x = 0
            sum_y = 0
            sum_z = 0

            for j in range(len(lst)):
                if (lst[j] == type):
                    sum_x = sum_x + vect[j][0]
                    sum_y = sum_y + vect[j][1]
                    sum_z = sum_z + vect[j][2]
            if (sum_x and sum_y and sum_z) == 0:
                out_x = 0
                out_y = 0
                out_z = 0
            else:
                out_x = sum_x/sqrt(sum_x**2+sum_y**2+sum_z**2)
                out_y = sum_y/sqrt(sum_x**2+sum_y**2+sum_z**2)
                out_z = sum_z/sqrt(sum_x**2+sum_y**2+sum_z**2)

            return [type, out_x, out_y, out_z]
            # self.int_avgvector[i] = [type, out_x, out_y, out_z]

        rll = range(len_labs)
        int_avgvector = Parallel(n_jobs=self.ncpus, require='sharedmem')(
            delayed(aveVector)(i, moltype, vectorarray, labs[i]) for i in track(rll,  description= "{:<25}".format('Averaging Vectors ...')))

        self.int_avgvector = int_avgvector

        avgvector = array(self.int_avgvector, dtype='object')
        del self.int_avgvector

        len_avgvector = len(avgvector)

        refine_avgvector = []
        refine_avgvector.append([0.0,0.0,0.0,0.0])
        refine_avgvector.append([1.0,0.0,0.0,0.0])

        n = 2
        for i in range(len_avgvector):
            # anything containing normalized vectors should be above zero
            if sum(avgvector[i,1:]) > 0:
                a = n
                b1 = avgvector[i,1]
                b2 = avgvector[i,2]
                b3 = avgvector[i,3]
                n = n+1
                refine_avgvector.append([a,b1,b2,b3])

        avgvector = array(refine_avgvector)

        for _sc in self.coloring:
            if _sc == 'bead':
                # instantaneous bead
                self.klk = self.instantaneous

            elif _sc == 'chain_avg':
                # Segment Averaged Vectors
                self.klk = self.chain_averaged

            elif _sc == 'cluster_avg':
                #Cluster averaged vectors
                # self.klk = zeros((len(vectorarray),3))
                @numba.jit(nopython=True,cache=True)
                def cluster_avg_loop(i):
                    for j in range(len(vectorarray)):
                        if int(moltype[j]) == avgvector[i,0]:
                            # self.klk[j,0] =  avgvector[i,1]
                            # self.klk[j,1] =  avgvector[i,2]
                            # self.klk[j,2] =  avgvector[i,3]
                            return [avgvector[i,1],avgvector[i,2],avgvector[i,3]]

                klk = Parallel(n_jobs=self.ncpus, require='sharedmem')(
                    delayed(cluster_avg_loop)(i) for i in track(range(len(avgvector)),
                    description= "{:<25}".format('Coloring Beads ...')))

            elif _sc == 'size':
                # self.klk = zeros((len(vectorarray),3))
                @numba.jit(nopython=True,cache=True)
                def size_loop(i, pertype, moltype):
                    for _type, _count in pertype:
                        if int(moltype[i]) == _type:
                            # # cluster number
                            # self.klk[i,0] =  _type
                            # # cluster size
                            # self.klk[i,1] =  _count
                            # self.klk[i,2] =  0
                            return [_type,_count,0]

                klk = Parallel(n_jobs=self.ncpus, require='sharedmem')(
                    delayed(cluster_avg_loop)(i, pertype, moltype) for i in track(range(len(avgvector)),
                    description= "{:<25}".format('Coloring Beads ...')))

            elif _sc == 'herman':
                # self.klk = zeros((len(vectorarray),3))
                @numba.jit(nopython=True,cache=True)
                def hermans_loop(i, avgvector, moltype):
                    for j in range(len(vectorarray)):
                        if int(moltype[j]) == avgvector[i,0]:
                            # vector of system
                            idx_v = [avgvector[i,1],avgvector[i,2],avgvector[i,3]]
                            # magnitude of system
                            mag_idx = sqrt(idx_v[0]**2.+idx_v[1]**2.+idx_v[2]**2.)
                            # magnitude of target
                            mag_z = 1.
                            # phi x
                            phi_x = arccos( (idx_v[0]*1.+idx_v[1]*0.+idx_v[2]*0.)/(mag_z*mag_idx) )
                            h_x = (3.0*(cos(phi_x))**2.0 - 1.0) /2.0
                            # phi y
                            phi_y = arccos( (idx_v[0]*0.+idx_v[1]*1.+idx_v[2]*0.)/(mag_z*mag_idx) )
                            h_y = (3.0*(cos(phi_y))**2.0 - 1.0) /2.0
                            # phi z
                            phi_z = arccos( (idx_v[0]*0.+idx_v[1]*0.+idx_v[2]*1.)/(mag_z*mag_idx) )
                            h_z = (3.0*(cos(phi_z))**2.0 - 1.0) /2.0

                            # self.klk[j,0] =  h_x
                            # self.klk[j,1] =  h_y
                            # self.klk[j,2] =  h_z
                            return h_x, h_y, h_z

                klk = Parallel(n_jobs=self.ncpus, require='sharedmem')(
                    delayed(hermans_loop)(i, avgvector, moltype) for i in track(range(len(avgvector)),
                    description= "{:<25}".format('Coloring Beads ...')))

        self.klk = klk
        return vectorlist, moltype, avgvector

    def verho(self, polymers, monomers):
        '''
        Verho alignment of segments of poylmer chain

        Averages over 2k+1 beads, where k is the length_coeff

        Inputs :
        polymers - int for number of polymers in system
        monomers - int for number of monomers in system
        length_coeff - int for length of window

        Returns :
        vectorlist - boolean true/false of a beads crystal status
        vectorarray - array of x,y,z data for each bead
        '''
        from numpy import array, sqrt, arange
        from rich.progress import track
        from joblib import Parallel, delayed

        vectorlist = [False]*polymers*monomers
        vectorarray = [[0,0,0]]*polymers*monomers
        lenvl = len(vectorlist)
        len_coeff = self.length_coeff

        @numba.jit(nopython=True,cache=True)
        def verho_dot(indexes, idx):
            dx = 0
            dy = 0
            dz = 0

            n = 0

            for k in indexes:
                # vector
                vx = self.xu[k+1] - self.xu[k-1]
                vy = self.yu[k+1] - self.yu[k-1]
                vz = self.zu[k+1] - self.zu[k-1]
                # unit vectors
                ux = vx/sqrt(vx**2+vy**2+vz**2)
                uy = vy/sqrt(vx**2+vy**2+vz**2)
                uz = vz/sqrt(vx**2+vy**2+vz**2)
                # add to d
                dx = dx+ux
                dy = dy+uy
                dz = dz+uz

                n = n + 1
            return dx, dy, dz, n

        @numba.jit(nopython=True,cache=True)
        def polymer_loop(i):
            for j in range(monomers):
                ind = j+i*monomers
                if j < (len_coeff+1):
                    # [0 ... n]
                    # [-1 0 ... n]
                    # ...
                    # [-n ... n]
                    idx = j-1
                    index = arange(-idx, len_coeff+1)+j+i*monomers
                elif j > (monomers-len_coeff)-2:
                    # [-n ... end]
                    idx = monomers-j-1
                    index = arange(-len_coeff, idx)+j+i*monomers
                else:
                    index = arange(-len_coeff, len_coeff+1)+j+i*monomers

                dx, dy, dz, n = verho_dot(index, j)
                # avg d, at its longest n is 2k+1
                lam = 1.0/(n)*sqrt(dx**2+dy**2+dz**2)

                # final unit vector
                ux = dx/sqrt(dx**2+dy**2+dz**2)
                uy = dy/sqrt(dx**2+dy**2+dz**2)
                uz = dz/sqrt(dx**2+dy**2+dz**2)

                if ux < 0:
                    ux = -1*ux
                if uy < 0:
                    uy = -1*uy
                if uz < 0:
                    uz = -1*uz

                if lam > self.verho_cutoff:
                    vectorlist[j+i*monomers] = True
                vectorarray[j+i*monomers] = [ux, uy, uz]

        Parallel(n_jobs=self.ncpus, require='sharedmem')(
            delayed(polymer_loop)(i) for i in track(range(polymers),  description= "{:<25}".format('Processing Verho ...')))


        vectorlist = array(vectorlist)
        vectorarray = array(vectorarray)

        self.instantaneous = vectorarray

        return vectorlist, vectorarray

    def legendre(self, legendre_type, polymers, monomers):
        '''
        P2 / P4 / P6 Order Parameter

        Averages over 2n+1 beads

        Inputs :
        legendre_type = string P2 / P4 / P6
        polymers - int for number of polymers in system
        monomers - int for number of monomers in system

        Returns :
        vectorlist - boolean true/false of a beads crystal status
        vectorarray - array of x,y,z data for each bead
        '''
        from joblib import Parallel, delayed
        from numpy import array, sqrt, arange
        from rich.progress import track
        import numba

        vectorlist = [False]*(polymers*monomers)
        vectorarray = [[0,0,0]]*(polymers*monomers)

        lenvl = len(vectorlist)
        len_coeff = self.length_coeff
        S = 0.

        @numba.jit(nopython=True,cache=True)
        def p2_dot(indexes, _x, _y, _z):
            S = 0.
            n = 0

            dx = 0
            dy = 0
            dz = 0

            for k in range(indexes):
                # vector
                vx2 = _x[k+1] - _x[k]
                vy2 = _y[k+1] - _y[k]
                vz2 = _z[k+1] - _z[k]

                vx1 = _x[k] - _x[k-1]
                vy1 = _y[k] - _y[k-1]
                vz1 = _z[k] - _z[k-1]

                # chord vector
                dx += _x[k+1] - _x[k-1]
                dy += _y[k+1] - _y[k-1]
                dz += _z[k+1] - _z[k-1]

                # unit vectors
                ux2 = vx2/sqrt(vx2**2+vy2**2+vz2**2)
                uy2 = vy2/sqrt(vx2**2+vy2**2+vz2**2)
                uz2 = vz2/sqrt(vx2**2+vy2**2+vz2**2)

                ux1 = vx1/sqrt(vx1**2+vy1**2+vz1**2)
                uy1 = vy1/sqrt(vx1**2+vy1**2+vz1**2)
                uz1 = vz1/sqrt(vx1**2+vy1**2+vz1**2)
                # add to S
                S += 0.5*(3.* (ux2*ux1 + uy2*uy1 + uz2*uz1)**2.  - 1. )

                n = n + 1
            return S, dx, dy, dz, n

        @numba.jit(nopython=True,cache=True)
        def p4_dot(indexes, _x, _y, _z):
            S = 0.
            n = 0

            dx = 0
            dy = 0
            dz = 0

            for k in range(indexes):
                # vector
                vx2 = _x[k+1] - _x[k]
                vy2 = _y[k+1] - _y[k]
                vz2 = _z[k+1] - _z[k]

                vx1 = _x[k] - _x[k-1]
                vy1 = _y[k] - _y[k-1]
                vz1 = _z[k] - _z[k-1]

                # chord vector
                dx += _x[k+1] - _x[k-1]
                dy += _y[k+1] - _y[k-1]
                dz += _z[k+1] - _z[k-1]

                # unit vectors
                ux2 = vx2/sqrt(vx2**2+vy2**2+vz2**2)
                uy2 = vy2/sqrt(vx2**2+vy2**2+vz2**2)
                uz2 = vz2/sqrt(vx2**2+vy2**2+vz2**2)

                ux1 = vx1/sqrt(vx1**2+vy1**2+vz1**2)
                uy1 = vy1/sqrt(vx1**2+vy1**2+vz1**2)
                uz1 = vz1/sqrt(vx1**2+vy1**2+vz1**2)
                # add to S
                S += 1./8.*(35.* (ux2*ux1 + uy2*uy1 + uz2*uz1)**4. \
                           -30.* (ux2*ux1 + uy2*uy1 + uz2*uz1)**2.  + 3. )

                n = n + 1
            return S, dx, dy, dz, n

        @numba.jit(nopython=True,cache=True)
        def p6_dot(indexes, _x, _y, _z):
            S = 0.
            n = 0

            dx = 0
            dy = 0
            dz = 0

            for k in range(1,indexes-1):
                # vector
                vx2 = _x[k+1] - _x[k]
                vy2 = _y[k+1] - _y[k]
                vz2 = _z[k+1] - _z[k]

                vx1 = _x[k] - _x[k-1]
                vy1 = _y[k] - _y[k-1]
                vz1 = _z[k] - _z[k-1]

                # chord vector
                dx += _x[k+1] - _x[k-1]
                dy += _y[k+1] - _y[k-1]
                dz += _z[k+1] - _z[k-1]

                # unit vectors
                ux2 = vx2/sqrt(vx2**2+vy2**2+vz2**2)
                uy2 = vy2/sqrt(vx2**2+vy2**2+vz2**2)
                uz2 = vz2/sqrt(vx2**2+vy2**2+vz2**2)

                ux1 = vx1/sqrt(vx1**2+vy1**2+vz1**2)
                uy1 = vy1/sqrt(vx1**2+vy1**2+vz1**2)
                uz1 = vz1/sqrt(vx1**2+vy1**2+vz1**2)
                # add to S
                S += 1./16.*(231.* (ux2*ux1 + uy2*uy1 + uz2*uz1)**6. \
                            -315.* (ux2*ux1 + uy2*uy1 + uz2*uz1)**4. \
                            +105.* (ux2*ux1 + uy2*uy1 + uz2*uz1)**2. - 5. )

                n = n + 1
            return S, dx, dy, dz, n


        def polymer_loop(i):
            for j in range(monomers):
                ind = j+i*monomers

                if j < (len_coeff+1):
                    # [0 ... n]
                    # [-1 0 ... n]
                    # ...
                    # [-n ... n]
                    idx = j-1
                    index = arange(-idx, len_coeff+1)+ind

                elif j > (monomers-len_coeff)-2:
                    # [-n ... end]
                    idx = monomers-j-1
                    index = arange(-len_coeff, idx)+ind

                else:
                    index = arange(-len_coeff, len_coeff+1)+ind

                _x = self.xu[index[0]-1:index[-1]+1]
                _y = self.yu[index[0]-1:index[-1]+1]
                _z = self.zu[index[0]-1:index[-1]+1]

                if legendre_type == 'P2':
                    S, dx, dy, dz, n = p2_dot(len(index), _x, _y, _z)
                elif legendre_type == 'P4':
                    S, dx, dy, dz, n = p4_dot(len(index), _x, _y, _z)
                elif legendre_type == 'P6':
                    S, dx, dy, dz, n = p6_dot(len(index), _x, _y, _z)

                # final unit vector
                ux = dx/sqrt(dx**2+dy**2+dz**2)
                uy = dy/sqrt(dx**2+dy**2+dz**2)
                uz = dz/sqrt(dx**2+dy**2+dz**2)

                if ux < 0:
                    ux = -1*ux
                if uy < 0:
                    uy = -1*uy
                if uz < 0:
                    uz = -1*uz

                lam = 1.0/(n)*S

                vectorarray[j+i*monomers] = [ux, uy, uz]

                if lam > self.legendre_cutoff:
                    vectorlist[j+i*monomers] = True
                    vectorarray[j+i*monomers] = [ux, uy, uz]

        Parallel(n_jobs=self.ncpus, require='sharedmem')(
            delayed(polymer_loop)(i) for i in track(range(polymers),  description= "{:<25}".format('Processing '+self.align_type+' ...')))

        vectorlist = array(vectorlist)
        vectorarray = array(vectorarray)

        self.instantaneous = vectorarray

        return vectorlist, vectorarray

    '''
    #########################################
    # External Facing
    #########################################
    '''
    def centroid_id(self, avgvector):
        '''
        Determines centroid of each crystal

        Inputs :
        avgvector - array of vectors for each crystal
        self.labels - list of labels for all crystals
        self.moltype - global, molecule type/ crystal type after clustering
        self.x - global, x location data
        self.y - global, y location data
        self.z - global, z location data

        Returns :
        exist_list - list of booleans as to whether the crystal exists or not
        centroid - centroid data for each crystal
                i - crystal number
                cx - centroid x location
                cy - centroid y location
                cz - centroid z location
                n  - number of beads in crystal
        statistics - statistics on the data

        Prints :
        Centroid data file 'filename.directions' which is read by the python
            script controlling abaqus behaviour

        '''
        from numpy import abs, array, min, max, arange
        from joblib import Parallel, delayed
        from rich.progress import track
        import numba
        # crystal info
        c_loc = []
        len_mol = len(self.moltype)
        for i in range(len_mol):
            if int(abs(self.moltype[i])) > 1:  # != 1 or 0 or -1:
                c_loc.append(
                    [self.moltype[i], self.x[i], self.y[i], self.z[i]])
        c_loc = array(c_loc)

        # find centroids
        centroid = []
        centroid_dist = []
        exist_list = []

        # skip -1, the noise
        @numba.jit(nopython=True,cache=True)
        def centroid_loop(i, c_loc):
            cx = 0
            cy = 0
            cz = 0
            Exists = False
            n = 0

            for j in range(len(c_loc)):
                if i == c_loc[j, 0]:
                    cx = cx + c_loc[j, 1]
                    cy = cy + c_loc[j, 2]
                    cz = cz + c_loc[j, 3]
                    n = n + 1
                    Exists = True

            if n != 0:
                cx = cx/n
                cy = cy/n
                cz = cz/n

            # centroid.append([i, cx, cy, cz, n])
            # exist_list.append(Exists)
            return [Exists, i, cx, cy, cz, n]


        min_label = min(self.labels)
        max_label = max(self.labels)+2

        label_range = arange(min_label, max_label, 1)
        centroids = Parallel(n_jobs=self.ncpus, require='sharedmem')(
            delayed(centroid_loop)(i, c_loc) for i in track(label_range,  description= "{:<25}".format('Calculating Centroids ...')))

        # print(centroids)
        centroid = []
        exist_list = []
        for _i in centroids:
            exist_list.append(int(_i[0]))
            centroid.append(_i[1:6])

        centroid = array(centroid)
        # Centroid and Vector of crystals for Abaqus
        total_exist = int(sum(exist_list))
        cent_vect = [[0.,0.,0.,0.,0.,0.,0.]]*total_exist
        statistics = [[0.,0.,0.,0.]]*total_exist
        len_exist = len(exist_list)
        n = 0
        for i in range(len_exist):
            if exist_list[i]:
                cent_vect[n] = [centroid[i, 0],
                                  centroid[i, 1],
                                  centroid[i, 2],
                                  centroid[i, 3],
                                  centroid[i, 4],
                                  avgvector[i, 1],
                                  avgvector[i, 2],
                                  avgvector[i, 3]]

                statistics[n] = [centroid[i, 4],
                                   avgvector[i, 1],
                                   avgvector[i, 2],
                                   avgvector[i, 3]]
                n = n+1

        cent_vect = array(cent_vect, dtype='object')
        statistics = array(statistics, dtype='object')

        # header
        cent_str = ''
        f = open(self.totaloutname, 'r')
        n = 0
        for line in f.readlines()[0:8]:
            if n == 3:
                cent_str += str(len(c_loc))+'\n'
            else:
                cent_str += line
            n = n+1
        f.close()

        # centroid string
        cent_str +='# Crystal Centroids, Sizes, and Avg Vectors \n'
        cent_str +='Crystal, cx, cy, cz, count, vx, vy, vz \n'

        for i in cent_vect:
            vect_str = ''
            for j in i:
                vect_str += str(j) + ' '
            cent_str += vect_str + '\n'

        seq = open(self.vectorname, 'w+')
        seq.write(cent_str)
        seq.close()

        return exist_list, centroid, statistics

    def crystal_dump(self, series, n_atoms, _r):
        '''
        Inputs:
        none, reads from globals

        Returns:
        none, atom prints

        Prints :
        Crystalline dump file
            atom number
            atom id
            crystal type

            x location
            y location
            z location
            xu unwrapped x location
            yu unwrapped y location
            zu unwrapped z location

            self.klk - chord vector, average vector, etc. Colors in ovito
        '''
        from numpy import array
        from rich.progress import track
        self.totaloutname = self.prefix+'_align'+self.custom_str+'.dump'
        #########################################
        # Full Crystalline Dump
        #########################################

        if series:
            # Reading
            f = open(self.filename, 'r')
            if _r == 0:
                box = f.readlines()[5:8]
            else:
                box = f.readlines()[_r*(n_atoms+9)+5:_r*(n_atoms+9)+8]
            f.close()
        else:
            # Reading
            f = open(self.filename, 'r')
            box = f.readlines()[5:8]
            f.close()

        if len(box[0].split()) == 2:
            # regular box
            xlo, xhi = box[0].split()
            ylo, yhi = box[1].split()
            zlo, zhi = box[2].split()

        elif len(box[0].split()) == 3:
            # triclinic box
            xlo, xhi, _xt= box[0].split()
            ylo, yhi, _yt= box[1].split()
            zlo, zhi, _zt= box[2].split()

        len_atoms = len(self.vectorarray)

        if self.just_atoms:
            lenmol = len(self.moltype)
        else:
            lenmol = len(self.moltype) + len(self.x_atom)


        xlo = float(xlo)
        xhi = float(xhi)
        ylo = float(ylo)
        yhi = float(yhi)
        zlo = float(zlo)
        zhi = float(zhi)
        if len(box[0].split()) == 3:
            _xt = float(_xt)
            _yt = float(_yt)
            _zt = float(_zt)

        xhi = xhi-xlo
        yhi = yhi-ylo
        zhi = zhi-zlo

        # crystal info
        from numpy import array
        c_loc = []

        for i in track(range(lenmol), description= "{:<25}".format('Writing Align ...')):
            if i < len_atoms:
                if self.data_condition == 'unwrapped':
                    c_loc.append(
                        [self.atom[i],
                         self.id[i],
                         self.moltype[i],
                         self.x[i]-xlo,
                         self.y[i]-ylo,
                         self.z[i]-zlo,
                         self.xu[i]-xlo,
                         self.yu[i]-ylo,
                         self.zu[i]-zlo,
                         self.vectorarray[i,0],
                         self.vectorarray[i,1],
                         self.vectorarray[i,2]])
                elif self.data_condition == 'index':
                    c_loc.append(
                        [self.atom[i],
                         self.id[i],
                         self.moltype[i],
                         self.x[i]-xlo,
                         self.y[i]-ylo,
                         self.z[i]-zlo,
                         self.ix[i],
                         self.iy[i],
                         self.iz[i],
                         self.vectorarray[i,0],
                         self.vectorarray[i,1],
                         self.vectorarray[i,2]])

            else:
                j = i - len_atoms
                if self.data_condition == 'unwrapped':
                    c_loc.append(
                        [self.x_atom[j],
                         self.x_id[j],
                         self.x_moltype[j],
                         self.x_x[j]-xlo,
                         self.x_y[j]-ylo,
                         self.x_z[j]-zlo,
                         self.x_xu[j]-xlo,
                         self.x_yu[j]-ylo,
                         self.x_zu[j]-zlo,
                         0.,
                         0.,
                         0.])
                elif self.data_condition == 'index':
                    c_loc.append(
                        [self.atom[j],
                         self.id[j],
                         self.moltype[j],
                         self.x[j]-xlo,
                         self.y[j]-ylo,
                         self.z[j]-zlo,
                         self.ix[j],
                         self.iy[j],
                         self.iz[j],
                         0.,
                         0.,
                         0.])


        c_loc = array(c_loc)

        if series:
            # header
            crystal_str = ''
            f = open(self.filename, 'r')
            n = 0

            if _r == 0:
                for line in f.readlines()[0:9]:
                    if n == 3 and len(box[0].split()) == 3:
                        crystal_str += str(lenmol)+'\n'

                    elif n == 5 and len(box[0].split()) == 3:
                        crystal_str += str(0.0) + ' ' + str(xhi)+ ' ' + str(_xt)+'\n'
                    elif n == 6 and len(box[0].split()) == 3:
                        crystal_str += str(0.0) + ' ' + str(yhi)+ ' ' + str(_yt)+'\n'
                    elif n == 7 and len(box[0].split()) == 3:
                        crystal_str += str(0.0) + ' ' + str(zhi)+ ' ' + str(_zt)+'\n'

                    elif n == 5 and len(box[0].split()) == 2:
                        crystal_str += str(0.0) + ' ' + str(xhi)+'\n'
                    elif n == 6 and len(box[0].split()) == 2:
                        crystal_str += str(0.0) + ' ' + str(yhi)+'\n'
                    elif n == 7 and len(box[0].split()) == 2:
                        crystal_str += str(0.0) + ' ' + str(zhi)+'\n'
                    elif n == 8:
                        if self.data_condition == 'unwrapped':
                            crystal_str += f'ITEM: ATOMS id mol type x y z xu yu zu vx vy vz \n'
                        elif self.data_condition == 'index':
                            crystal_str += f'ITEM: ATOMS id mol type x y z ix iy iz vx vy vz \n'
                    else:
                        crystal_str += line
                    n = n+1
            else:
                for line in f.readlines()[_r*(n_atoms+9)+0:_r*(n_atoms+9)+9]:
                    if n == 3 and len(box[0].split()) == 3:
                        crystal_str += str(lenmol)+'\n'

                    elif n == 5 and len(box[0].split()) == 3:
                        crystal_str += str(0.0) + ' ' + str(xhi)+ ' ' + str(_xt)+'\n'
                    elif n == 6 and len(box[0].split()) == 3:
                        crystal_str += str(0.0) + ' ' + str(yhi)+ ' ' + str(_yt)+'\n'
                    elif n == 7 and len(box[0].split()) == 3:
                        crystal_str += str(0.0) + ' ' + str(zhi)+ ' ' + str(_zt)+'\n'

                    elif n == 5 and len(box[0].split()) == 2:
                        crystal_str += str(0.0) + ' ' + str(xhi)+'\n'
                    elif n == 6 and len(box[0].split()) == 2:
                        crystal_str += str(0.0) + ' ' + str(yhi)+'\n'
                    elif n == 7 and len(box[0].split()) == 2:
                        crystal_str += str(0.0) + ' ' + str(zhi)+'\n'
                    elif n == 8:
                        if self.data_condition == 'unwrapped':
                            crystal_str += f'ITEM: ATOMS id mol type x y z xu yu zu vx vy vz \n'
                        elif self.data_condition == 'index':
                            crystal_str += f'ITEM: ATOMS id mol type x y z ix iy iz vx vy vz \n'
                    else:
                        crystal_str += line
                    n = n+1
            f.close()

            for i in range(len(c_loc)):
                cc_loc = c_loc[i]
                c_str = ''
                for j in range(len(cc_loc)):
                    if j < 3:
                        c_str += str(int(cc_loc[j])) + ' '
                    elif 2 < j < 6:
                        c_str += '{:.4f}'.format(cc_loc[j]) + ' '
                    elif self.data_condition == 'unwrapped' and 5 < j < 9:
                        c_str += '{:.4f}'.format(cc_loc[j]) + ' '
                    elif self.data_condition == 'index' and 5 < j < 9:
                        c_str += str(int(cc_loc[j])) + ' '
                    elif j>8:
                        c_str += '{:.4f}'.format(cc_loc[j]) + ' '
                crystal_str += c_str + '\n'

            # Writing file
            if _r == 0:
                seq = open(self.totaloutname, 'w+')
            else:
                seq = open(self.totaloutname, 'a+')
            seq.write(crystal_str)
            seq.close()
        else:
            # header
            crystal_str = ''
            f = open(self.filename, 'r')
            n = 0


            for line in f.readlines()[0:9]:
                if n == 3 and len(box[0].split()) == 3:
                    crystal_str += str(lenmol)+'\n'

                elif n == 5 and len(box[0].split()) == 3:
                    crystal_str += str(0.0) + ' ' + str(xhi)+ ' ' + str(_xt)+'\n'
                elif n == 6 and len(box[0].split()) == 3:
                    crystal_str += str(0.0) + ' ' + str(yhi)+ ' ' + str(_yt)+'\n'
                elif n == 7 and len(box[0].split()) == 3:
                    crystal_str += str(0.0) + ' ' + str(zhi)+ ' ' + str(_zt)+'\n'

                elif n == 5 and len(box[0].split()) == 2:
                    crystal_str += str(0.0) + ' ' + str(xhi)+'\n'
                elif n == 6 and len(box[0].split()) == 2:
                    crystal_str += str(0.0) + ' ' + str(yhi)+'\n'
                elif n == 7 and len(box[0].split()) == 2:
                    crystal_str += str(0.0) + ' ' + str(zhi)+'\n'
                elif n == 8:
                    crystal_str += 'ITEM: ATOMS id mol type x y z xu yu zu vx vy vz\n'
                else:
                    crystal_str += line
                n = n+1
            f.close()

            for i in range(len(c_loc)):
                cc_loc = c_loc[i]
                c_str = ''
                for j in range(len(cc_loc)):
                    if j < 3:
                        c_str += str(int(cc_loc[j])) + ' '
                    elif 2 < j < 6:
                        c_str += '{:.4f}'.format(cc_loc[j]) + ' '
                    elif self.data_condition == 'unwrapped' and 5 < j < 9:
                        c_str += '{:.4f}'.format(cc_loc[j]) + ' '
                    elif self.data_condition == 'index' and 5 < j < 9:
                        c_str += str(int(cc_loc[j])) + ' '
                    elif j>8:
                        c_str += '{:.4f}'.format(cc_loc[j]) + ' '
                crystal_str += c_str + '\n'

            # Writing file
            seq = open(self.totaloutname, 'w+')
            seq.write(crystal_str)
            seq.close()

    def statistics(self):
        '''
        Printing statistics in the terminal

        Inputs :
        statistics - average overall vector (important if doing fiber drawing)

        Returns :
        None

        Prints (terminal):
        passing - number of passing beads (% crystallinity analog)
        outx - overall crystalline x direction
        outy - overall crystalline y direction
        outz - overall crystalline z direction
        '''
        from numpy import sqrt

        if len(self.stats) > 0:
            counting = self.stats[:, 0]

            passing = 0
            vx = 0
            vy = 0
            vz = 0
            for i in range(len(counting)):
                passing += counting[i]

                vx += counting[i] * self.stats[i, 1]
                vy += counting[i] * self.stats[i, 2]
                vz += counting[i] * self.stats[i, 3]

            outx = vx / sqrt(vx**2 + vy**2 + vz**2)
            outy = vy / sqrt(vx**2 + vy**2 + vz**2)
            outz = vz / sqrt(vx**2 + vy**2 + vz**2)

        else:
            passing = 0
            outx = 0
            outy = 0
            outz = 0

        return passing, outx, outy, outz

    def voro_dump(self):
        from numpy import zeros, array, unique
        f = open(self.totaloutname, 'r')
        data = f.readlines()
        f.close()

        box = data[5:8]
        data = data[9:]

        # atom data
        len_data = len(data)
        atom = [0]*len_data
        id   = [0]*len_data
        mol  = [0]*len_data
        type = [0]*len_data
        x    = [0]*len_data
        y    = [0]*len_data
        z    = [0]*len_data
        xu   = [0]*len_data
        yu   = [0]*len_data
        zu   = [0]*len_data

        for i in range(len_data):
            idx = data[i].split()
            atom[i] = float(idx[0])
            id[i] = float(idx[1])
            type[i] = float(idx[2])
            x[i] = float(idx[3])
            y[i] = float(idx[4])
            z[i] = float(idx[5])
            xu[i] = float(idx[6])
            yu[i] = float(idx[7])
            zu[i] = float(idx[8])

        atom = array(atom)
        id = array(id)
        moltype = array(type)
        x = array(x)
        y = array(y)
        z = array(z)
        xu = array(xu)
        yu = array(yu)
        zu = array(zu)

        # box data
        xlo, xhi = box[0].split()
        ylo, yhi = box[1].split()
        zlo, zhi = box[2].split()

        xlo = float(xlo)
        xhi = float(xhi)
        ylo = float(ylo)
        yhi = float(yhi)
        zlo = float(zlo)
        zhi = float(zhi)

        # normalize to 0,0,0
        xlen = xhi - xlo
        ylen = yhi - ylo
        zlen = zhi - zlo

        x = x-xlo
        y = y-ylo
        z = z-zlo

        xhi = xlen
        yhi = ylen
        zhi = zlen

        xlo = 0.0
        ylo = 0.0
        zlo = 0.0

        data = []
        xyz = []
        for i in range(len(x)):
            data.append([x[i],y[i],z[i], moltype[i]])

        lims = [[xlo, xhi], [ylo, yhi], [zlo, zhi]]

        self.uniques = list(unique(moltype))

        return data, lims

    '''
    #########################################
    # FIND
    #########################################
    '''

    def find(self):
        from sys import stdout
        print('\nAlign v0.1.0')
        print("{:<25}".format('========================='))

        self._variable_check()

        try:
            f = open(self.filename, 'r')
            self.n_atoms = int(f.readlines()[3])
            f.close()
        except:
            print("Data is an empty array")
            exit()

        polymers, monomers = self.data_parse(self.filename, self.series)

        if polymers < 0 or polymers == 0:
            print(
                "Polymer chain amount cannot be negative or equal to zero")
            exit()
        elif not isinstance(polymers, int):
            print("Polymer chain amount is not an integer")
            exit()

        if monomers < 0 or monomers == 0:
            print(
                "Monomer chain amount cannot be negative or equal to zero")
            exit()
        elif not isinstance(monomers, int):
            print("Monomer chain amount is not an integer")
            exit()

        if (2*self.length_coeff+1) > monomers:
            print(
                "Length coefficient must be integer below half the chain length, as it requires 2k+1")
            exit()


        # Final Inits
        self.past_labels = None
        self.series_stats = None

        # Finished clustering tag
        for _i in range(self.n_steps):

            if self.series and _i < self.n_steps-1 and self.timesteps is False:

                print("{:<25}".format('-------------------------'))
                print("{:<25}".format(f'Timestep {_i+1} of {self.n_steps-1}'))
                print("{:<25}".format('-------------------------'))
                self.polymers, self.monomers = self.data_parse(self.filename, self.series, _i)

            elif self.series and _i < self.n_steps-1 and self.timesteps is not False:

                print("{:<25}".format('-------------------------'))
                print("{:<25}".format(f'Timestep {_i+1} of timestep array'))
                print("{:<25}".format('-------------------------'))
                self.polymers, self.monomers = self.data_parse(self.filename, self.series, _i)

            elif not self.series:
                self.polymers, self.monomers = self.data_parse(self.filename, self.series)

            else:
                print('Data parse failed. Is this a data series or a single timestep?')
                exit()

            #########################################
            # Application of Alignment Functions / Order Parameters
            #########################################
            if self.align_type == 'verho':
                vectorlist, vectorarray = self.verho(
                    self.polymers, self.monomers)
            elif self.align_type == 'P2':
                vectorlist, vectorarray = self.legendre(
                    'P2', self.polymers, self.monomers)
            elif self.align_type == 'P4':
                vectorlist, vectorarray = self.legendre(
                    'P4', self.polymers, self.monomers)
            elif self.align_type == 'P6':
                vectorlist, vectorarray = self.legendre(
                    'P6', self.polymers, self.monomers)
            else:
                warning_str  = "Orientation parameter not supported. "
                warning_str += "Current methods : \n Verho :\'verho\'\n "
                warning_str += "Legendre P2 : \'P2\' \n "
                warning_str += "Legendre P4 : \'P4\' \n "
                warning_str += "Legendre P6 : \'P6\' \n "

                print(warning_str)
                exit()

            # refines end of chains
            # vectorlist, vectorarray = self.refine(vectorlist, vectorarray)

            #########################################
            # Minimum Length Enforcement
            #########################################
            vectorlist = self.length_filter(vectorlist)

            #########################################
            # Chain Averaging
            #########################################
            data, chords, chain_id = self.avg_vectors(vectorlist, vectorarray)
            num_pass = sum(vectorlist)

            self.chain_id = chain_id

            self.vectorarray = vectorarray

            #########################################
            # Clustering
            #########################################
            # crystal interval starts at 2

            # validate data
            passed_clustering = True
            self.clustered = False
            if data.size == 0:
                print("Alignment returns no passes. Skipping clustering.")
                self.labels  = [-1]*len(vectorlist)
                passed_clustering = False
            elif num_pass < self.crystalmin:
                print("Passing bead count below crystal minimum. Skipping clustering.")
                self.labels  = [-1]*len(vectorlist)
                passed_clustering = False
            else:
                while self.clustered is False:
                    # try:
                    if self.verbose:
                        print('\n vdbscan \n')
                    VDB = VDBSCAN(data,
                                  chords,
                                  self.ncpus,
                                  self.cluster_method,
                                  self.neighlist_method,
                                  self.eps,
                                  self.lam,
                                  self.leaf_size,
                                  self.min_pts,
                                  self.crystalmin)
                    self.labels, core_loc = VDB.fit()
                    if self.verbose:
                        print('Final leaf size : '+str(self.leaf_size)+'\n')
                    self.clustered = True
                    # except:
                        # if self.verbose:
                        #     print('Modifying Leaf count \n')
                        # self.leaf_size = self.leaf_size + 2

            #########################################
            # Type Filter
            #########################################
            moltype = self.type_filter(vectorlist, self.labels)

            #########################################
            # Optional Second Length Filter
            #########################################
            if self.second_filt:
                vectorlist = self.length_filter(vectorlist)
                self.segments = self.length_stats(vectorlist)
            else:
                self.segments = self.length_stats(vectorlist)

            #########################################
            # Minimum Crystal Size Filter
            #########################################
            self.vectorlist, moltype, avgvector = self.quantity_filter(
                vectorlist, vectorarray, moltype, self.labels)

            #########################################
            # Renumbering Crystals
            #########################################
            if _i == 0:
                self.moltype, uniques, self.labels, self.past_labels = self.renumber(moltype, vectorlist)
            elif self.past_labels is None:
                self.moltype, uniques, self.labels, self.past_labels = self.renumber(moltype, vectorlist)
            else:
                self.moltype, uniques, self.labels, self.past_labels = self.renumber(moltype, vectorlist, self.past_labels)

            #########################################
            # Write PointCloud Dump File
            #########################################
            self.crystal_dump(self.series, self.n_atoms, _i)

            #########################################
            # Centroid data printing
            #########################################
            exist_list, centroid, self.stats = self.centroid_id(avgvector)

            out = list(self.statistics())

            self.uniques = uniques

            out.append(uniques)

            if _i == 0:
                self.series_stats = [out]
            elif self.series_stats is None:
                self.series_stats = [out]
            else:
                self.series_stats.append(out)

            print('=========================')
            if passed_clustering:
                print('\nNumber of Crystals : '+str(uniques))
                print('\nPercent Crystal : '+str(round(out[0]/len(self.vectorlist),5)*100.0)+'% \n')
                print('Average Vectors ')
                print('ux : '+str(out[1]))
                print('uy : '+str(out[2]))
                print('uz : '+str(out[3]))
            else:
                print('\nInput did not pass.')
                print('\nIf the input is amorphous, ignore this message.')
                print('\nNumber of Crystals : 0')
                print('\nPercent Crystal : '+str(round(out[0]/len(self.vectorlist),5)*100.0)+'% \n')
            print('=========================\n')
