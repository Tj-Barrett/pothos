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
