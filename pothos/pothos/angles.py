class angles():

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
        # self.totaloutname = self.prefix+'_Verho.dump'

        self.atom_types=False
        #----------------------------------
        # Defaults
        #----------------------------------
        # Legendre

        self.ncpus = -1

    def angle_compute(self, polymers, monomers):
        from numpy import sqrt, arccos, pi, abs
        from rich.progress import track
        from joblib import Parallel, delayed
        import numba
        @numba.jit(nopython=True,cache=True)

        def angles(x2, y2, z2, x1, y1, z1, x0, y0, z0):
            #for i in range(1,len(id)-1):
            #if id[i-1] == id[i+1]:
            vx1 = x2 - x1
            vy1 = y2 - y1
            vz1 = z2 - z1

            vx2 = x0 - x1
            vy2 = y0 - y1
            vz2 = z0 - z1

            dot = vx1*vx2+vy1*vy2+vz1*vz2
            mag1 = sqrt(vx1**2+vy1**2+vz1**2)
            mag2 = sqrt(vx2**2+vy2**2+vz2**2)

            if abs(mag1) < 1E-3:
                mag1 = 1.
            if abs(mag2) < 1E-3:
                mag2 = 1.

            step = abs(arccos( dot / (mag1*mag2) ))

            return 180.0*step/pi

        backbone_angle = [0]*(len(self.id))

        def polymer_loop(i):

            for j in range(monomers):
                ind = j+i*monomers
                if j < (self.length_coeff_b+1):
                    # [0 ... n]
                    # [-1 0 ... n]
                    # ...
                    # [-n ... n]
                    idx = j-1
                    jf = self.length_coeff_f
                    jb = idx
                elif j > (monomers-self.length_coeff_f)-2:
                    # [-n ... end]
                    idx = monomers-j-1
                    jf = idx
                    jb = self.length_coeff_b
                else:
                    jf = self.length_coeff_f
                    jb = self.length_coeff_b

                backbone_angle[ind] = angles(self.xu[ind+jf],self.yu[ind+jf],self.yu[ind+jf],
                                             self.xu[ind],  self.yu[ind],  self.yu[ind],
                                             self.xu[ind-jb],self.yu[ind-jb],self.yu[ind-jb])

        Parallel(n_jobs=self.ncpus, require='sharedmem')(
            delayed(polymer_loop)(i) for i in track(range(polymers),  description= "{:<25}".format('Computing Angles ...')))

        self.backbone_angle = backbone_angle

        kink_type = [-1]*len(backbone_angle)

        # DOI 10.1021/acsmacrolett.7b00808
        for i in track(range(len(backbone_angle)),  description= "{:<25}".format('Sorting Angles ...')):
            back = 180. - backbone_angle[i]
            if back > 90.:
                # return G
                kink_type[i] = 1
            elif  90. >= back > 60.:
                kink_type[i] = 2
            elif  60. >= back > 30.:
                kink_type[i] = 3
            elif  30. >= back > 15.:
                kink_type[i] = 4
            elif 15 >= back >= 0.:
                kink_type[i] = 5

        # Remove Ends
        for i in range(polymers):
            ind1 = i*monomers
            ind2 = (i+1)*monomers - 1
            kink_type[ind1] = -1
            kink_type[ind2] = -1
        self.ppa = kink_type

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
        self.totaloutname = self.prefix+'_angles'+self.custom_str+'.dump'
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

        len_atoms = len(self.moltype)

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
        for i in track(range(lenmol), description= "{:<25}".format('Writing Angles ...')):
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
                         self.backbone_angle[i],
                         self.ppa[i],
                         0])
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
                         self.backbone_angle[i],
                         self.ppa[i],
                         0])
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
                        [self.x_atom[j],
                         self.x_id[j],
                         self.x_moltype[j],
                         self.x_x[j]-xlo,
                         self.x_y[j]-ylo,
                         self.x_z[j]-zlo,
                         self.x_ix[j],
                         self.x_iy[j],
                         self.x_iz[j],
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
                            crystal_str += 'ITEM: ATOMS id mol type x y z xu yu zu vx vy vz\n'
                        elif self.data_condition == 'index':
                            crystal_str += 'ITEM: ATOMS id mol type x y z ix iy iz vx vy vz\n'
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
                            crystal_str += 'ITEM: ATOMS id mol type x y z xu yu zu vx vy vz\n'
                        elif self.data_condition == 'index':
                            crystal_str += 'ITEM: ATOMS id mol type x y z ix iy iz vx vy vz\n'
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

        else:
            # header
            crystal_str = ''
            f = open(self.filename, 'r')
            n = 0
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
                        crystal_str += 'ITEM: ATOMS id mol type x y z xu yu zu vx vy vz\n'
                    elif self.data_condition == 'index':
                        crystal_str += 'ITEM: ATOMS id mol type x y z ix iy iz vx vy vz\n'
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
            seq = open(self.totaloutname, 'w+')
            seq.write(crystal_str)
            seq.close()

    def stats(self):
        print('Stats')

    def compute(self, ncpus = -1, len_forward=5, len_back=1, length_coeff = None, series=False, atom_types=False, nevery=1, just_atoms=True, custom_str=None, timesteps=None):

        print('\nPothos Angles v0.1.0')
        print("{:<25}".format('========================='))

        if isinstance(ncpus, int) and ncpus > 0:
            self.ncpus = ncpus
        else:
            print(
                    "ncpus is not an integer above 0")
            exit()

        if isinstance(just_atoms, bool):
            if just_atoms:
                self.just_atoms = True
            else:
                self.just_atoms = False

        if isinstance(length_coeff, int) and length_coeff>0:
            if just_atoms:
                self.length_coeff_f = length_coeff
                self.length_coeff_b = length_coeff
        elif length_coeff is None:

            if isinstance(len_forward, int) and len_forward>0:
                self.length_coeff_f = len_forward
            else:
                print(
                        "Check length forward coefficient value. Must be above 0 and an int.")
                exit()

            if isinstance(len_back, int) and len_forward>0:
                self.length_coeff_b = len_back
            else:
                print(
                        "Check length back coefficient value. Must be above 0 and an int.")
                exit()

        else:
            print(
                    "Check length coefficient value. Must be above 0 and an int.")
            exit()



        if isinstance(atom_types, (int,list) ):
            if isinstance(atom_types, int ) and atom_types is not False:
                self.atom_types=[atom_types]
            else:
                self.atom_types=atom_types
        else:
            self.atom_types=False

        if isinstance(custom_str, str):
            self.custom_str = '-'+custom_str
        else:
            self.custom_str = ''

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

        if isinstance(nevery, int) and nevery > 0:
            self.nevery = nevery
        else:
            print(
                    "nevery is not an integer above 0")
            exit()

        if timesteps is None:
            pass
        elif timesteps > 0 and timesteps < self.n_steps and isinstance(timesteps, int):
            self.n_steps = timesteps
        else:
            print("Check range of timesteps selected")
            exit()

        for _i in range(self.n_steps):
            self.angle_compute(polymers, monomers)

            self.dump(self.series, self.n_atoms, _i)

            if self.series and _i < self.n_steps-1:
                print("{:<25}".format('-------------------------'))
                print("{:<25}".format(f'Timestep {_i+1} of {self.n_steps}'))
                print("{:<25}".format('-------------------------'))
                polymers, monomers = self.data_parse(self.filename, self.series, _i+1)
