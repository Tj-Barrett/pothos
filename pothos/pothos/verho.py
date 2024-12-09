from pothos.data_parse import data_parse

class verho():

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
        self.totaloutname = self.prefix+'_Verho.dump'

        #----------------------------------
        # Defaults
        #----------------------------------
        # Verho
        self.verho_cutoff=0.9
        self.length_coeff=7

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
            id, atom, moltype, x, y, z, xu, yu, zu = map(list, zip(*sorted(zip(id, atom, moltype, x, y, z, xu, yu, zu), reverse=False)))


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
            id, atom, moltype, x, y, z, xu, yu, zu = map(list, zip(*sorted(zip(id, atom, moltype, x, y, z, xu, yu, zu), reverse=False)))


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

    def dump(self, series, n_atoms, _r):
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
        if isinstance(self.custom_str, str):
            self.totaloutname = self.prefix+'_Verho'+self.custom_str+'.dump'
        else:
            self.totaloutname = self.prefix+'_Verho.dump'
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

        for i in track(range(lenmol), description= "{:<25}".format('Writing Verho ...')):
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
                        [self.atom[i],
                         self.id[i],
                         self.moltype[i],
                         self.x[i]-xlo,
                         self.y[i]-ylo,
                         self.z[i]-zlo,
                         self.ix[i],
                         self.iy[i],
                         self.iz[i],
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
                        crystal_str += str(len(c_loc))+'\n'

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
                        crystal_str += str(len(c_loc))+'\n'

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
                    c_str += str(cc_loc[j]) + ' '
                crystal_str += c_str + '\n'

            # Writing file
            if _r == 0:
                seq = open(self.totaloutname, 'w+')
            else:
                seq = open(self.totaloutname, 'a+')
            seq.write(crystal_str)
            seq.close()

    def vector_average(self):
        vx = 0
        vy = 0
        vz = 0

        for va in self.vectorarray:
            vx+= va[0]
            vy+= va[1]
            vz+= va[2]

        vx = vx/len(self.vectorarray)
        vy = vy/len(self.vectorarray)
        vz = vz/len(self.vectorarray)

        return vx, vy, vz

    def stats(self, indx = [[1, 0, 0], [0, 1, 0], [0, 0, 1]]):
        import numpy as np

        def p2_dot(indx, avg):
            costheta = (indx[0]*avg[0] + indx[1]*avg[1] + indx[2]*avg[2]) \
                        /( (indx[0]**2 + indx[1]**2 + indx[2]**2)**(0.5) \
                        *(avg[0]**2 + avg[1]**2 + avg[2]**2)**(0.5) )
            return 0.5*(3.* (costheta)**2.  - 1. )

        def p4_dot(indx, avg):
            costheta = (indx[0]*avg[0] + indx[1]*avg[1] + indx[2]*avg[2]) \
                        /( (indx[0]**2 + indx[1]**2 + indx[2]**2)**(0.5) \
                        *(avg[0]**2 + avg[1]**2 + avg[2]**2)**(0.5) )
            return 1./8.*(35.* (costheta)**4. \
                           -30.* (costheta)**2.  + 3. )

        def p6_dot(indx, avg):
            costheta = (indx[0]*avg[0] + indx[1]*avg[1] + indx[2]*avg[2]) \
                        /( (indx[0]**2 + indx[1]**2 + indx[2]**2)**(0.5) \
                        *(avg[0]**2 + avg[1]**2 + avg[2]**2)**(0.5) )
            return 1./16.*(231.* (costheta)**6. \
                            -315.* (costheta)**4. \
                            +105.* (costheta)**2. - 5. )

        def cosines_compute(indx, avg):
            return (indx[0]*avg[0] + indx[1]*avg[1] + indx[2]*avg[2]) \
                  /( (indx[0]**2 + indx[1]**2 + indx[2]**2)**(0.5) \
                  *(avg[0]**2 + avg[1]**2 + avg[2]**2)**(0.5) )


        vx = 0
        vy = 0
        vz = 0

        for va in self.vectorarray:
            vx+= va[0]
            vy+= va[1]
            vz+= va[2]

        vx = vx/len(self.vectorarray)
        vy = vy/len(self.vectorarray)
        vz = vz/len(self.vectorarray)

        P2 = []
        P4 = []
        P6 = []
        cosines = []
        for i in indx:
            _cosine = cosines_compute(i,[vx, vy, vz])
            cosines.append(_cosine)

            P2.append(p2_dot(i,[vx, vy, vz]))
            P4.append(p4_dot(i,[vx, vy, vz]))
            P6.append(p6_dot(i,[vx, vy, vz]))

        return P2, P4, P6, cosines

    def compute(self, ncpus = -1, verho_cutoff=0.9, length_coeff=7, atom_types=False, nevery=1, just_atoms=True, series=False,custom_str=None):
        import time
        start = time.time()
        print('\nPothos Verho v0.1.0')
        print("{:<25}".format('========================='))

        if isinstance(ncpus, int):
            self.ncpus = ncpus
        else:
            print(
                    "ncpus is not an integer")
            exit()

        if isinstance(just_atoms, bool):
            if just_atoms:
                self.just_atoms = True
            else:
                self.just_atoms = False

        if isinstance(custom_str, str):
            self.custom_str = '-'+custom_str
        else:
            self.custom_str = None

        if isinstance(atom_types, (int,list) ):
            if isinstance(atom_types, int ) and atom_types is not False:
                self.atom_types=[atom_types]
            else:
                self.atom_types=atom_types
        else:
            self.atom_types=False

        if isinstance(nevery, int) and nevery > 0:
            self.nevery = nevery
        else:
            print(
                    "nevery is not an integer above 0")
            exit()

        if isinstance(series, bool):
            self.series = series
            if not self.series:
                self.n_steps = 1
        else:
            print("Data series variable is not a boolean")
            exit()

        try:
            f = open(self.filename, 'r')
            self.n_atoms = int(f.readlines()[3])
            f.close()
        except:
            print("Data is an empty array")
            exit()

        polymers, monomers = self.data_parse(self.filename, self.series)

        ####################################
        # Verho Equation Information
        ####################################
        #
        # Verho Coefficient
        #
        if self.length_coeff < 0 or self.length_coeff == 0:
            print(
                "Verho coefficient cannot be negative or equal to zero")
            exit()
        elif (2*self.length_coeff+1) > monomers:
            print(
                "Verho coefficient must be integer below half the chain length, as it requires 2k+1")
            exit()
        elif not isinstance(self.length_coeff, int):
            print("Verho coefficient is not an integer")
            exit()

        #
        # Verho Cutoff Lambda
        #
        if not 0.0 < self.verho_cutoff < 1.0:
            print("Verho Cutoff must be between 0 and 1")
            exit()

        for _i in range(self.n_steps):
            self.vectorlist, self.vectorarray = self.verho(polymers, monomers)

            if _i == 0:
                self.vec_avg = [self.vector_average()]
                self.statistics = [self.stats()]
            else:
                self.vec_avg.append(self.vector_average())
                self.statistics.append(self.stats())

            self.dump(self.series, self.n_atoms, _i)

            if self.series and _i < self.n_steps-1:
                print("{:<25}".format('-------------------------'))
                print("{:<25}".format(f'Timestep {_i+1} of {self.n_steps}'))
                print("{:<25}".format('-------------------------'))
                polymers, monomers = self.data_parse(self.filename, self.series, _i+1)


        end = time.time()

        total = end-start
        print("{:<25}".format('========================='))
        print("{:<25}".format(f"Verho with {self.ncpus} cpus : {total} s"))
        print("{:<25}".format('========================='))
