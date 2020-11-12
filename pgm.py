import numpy as np

def readline(pgmf):
    tmp = pgmf.readline()
    while(tmp[0] == ord('#')):
        tmp = pgmf.readline()
    return tmp

def read_pgm(pgmf):
    assert readline(pgmf) == b'P5\n'
    (width, height) = [int(i) for i in readline(pgmf).split()]
    assert int(readline(pgmf)) <= 255

    return np.fromfile(pgmf, dtype=np.uint8).reshape((height, width))

def read_pgm_1d_array(pgmf):
    assert readline(pgmf) == b'P5\n'
    (width, height) = [int(i) for i in readline(pgmf).split()]
    assert int(readline(pgmf)) <= 255

    return np.fromfile(pgmf, dtype=np.uint8).reshape((1, height*width)).tolist()

# test read_pgm
""" if __name__ == "__main__":
    pgmfile = str(sys.argv[1])
    with open(pgmfile,'rb') as pgmf:
        data = read_pgm(pgmf)
    print(data.shape)
    im = Image.fromarray(data)
    im.show()
 """
        