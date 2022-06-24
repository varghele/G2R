import numpy as np
import mendeleev as me

#____________________________ Quaternion functions #__________________________
def q_mult(q1, q2):
    w1, x1, y1, z1 = q1
    w2, x2, y2, z2 = q2
    w = w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2
    x = w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2
    y = w1 * y2 + y1 * w2 + z1 * x2 - x1 * z2
    z = w1 * z2 + z1 * w2 + x1 * y2 - y1 * x2
    return w, x, y, z
def q_conjugate(q):
    w, x, y, z = q
    return (w, -x, -y, -z)
def qv_mult(q1, v1):
    q2 = (0.0,) + v1
    return q_mult(q_mult(q1, q2), q_conjugate(q1))[1:]
#_____________________________________________________________________________


#_________________________ Useful functions for Data processing #______________
def helper_for_KOS(in_pth):
    tripos_ATOM=False
    mol2_atom_lines=[]
    with open(in_pth) as fh:
        for line in fh:
            if "ATOM" in line:
                tripos_ATOM = True
            if "BOND" in line:
                tripos_ATOM = False
                tripos_BONDS = True
            if tripos_ATOM and not "ATOM" in line:
                mol2_atom_lines.append(line.split())
    return mol2_atom_lines

def helper_rotate_and_transpose_to_KOS(mol2_atom_lines):
    coordinates=np.array(mol2_atom_lines)[:,2:5].astype(float)
    moltypes=[i.split(".")[0] for i in np.array(mol2_atom_lines)[:,5]]

    
    # Calculate Z-Vector, LONGEST vector
    zvec,zi,zy=[],[],[]
    for i in np.array(coordinates):
        for j in np.array(coordinates):
            vec=j-i
            if np.linalg.norm(vec)>np.linalg.norm(zvec):
                zvec,zi,zj=vec,i,j
    ## SCALE the new vectors
    zvec=np.array(zvec)/np.linalg.norm(zvec)
    
    # Calculate Y-Vector, SECOND-LONGEST vector, perpendicular to Z
    p_yvec=[]
    for j in np.array(coordinates):
        #Get vector to molecule
        mvec=j-zi
        #Get projection vector
        pvec=(np.dot(mvec,zvec)/(np.linalg.norm(zvec)))*zvec/np.linalg.norm(zvec)
        #Get y vector, positive
        vec=mvec-pvec
        if np.linalg.norm(vec)>np.linalg.norm(p_yvec):
            p_yvec=vec
    ## SCALE the new vectors
    p_yvec=np.array(p_yvec)/np.linalg.norm(p_yvec)
    
    # Calculate X-Vector, (Perpendicular Z,Y)
    p_xvec=[]
    for j in np.array(coordinates):
        #Get vector to molecule
        mvec=j-zi
        #Get x unit vector
        xuni=np.cross(zvec,p_yvec)/np.linalg.norm(np.cross(zvec,p_yvec))
        #Get x_vector
        vec=np.dot(mvec,xuni)*xuni
        if np.linalg.norm(vec)>np.linalg.norm(p_xvec):
            p_xvec=vec
    ## SCALE the new vectors
    p_xvec=np.array(p_xvec)/np.linalg.norm(p_xvec)
    
    ## SCALE the new vectors
    zvec=np.array(zvec)/np.linalg.norm(zvec)
    p_yvec=np.array(p_yvec)/np.linalg.norm(p_yvec)
    p_xvec=np.array(p_xvec)/np.linalg.norm(p_xvec)
    
    
    ##ROTATION
    #Calculate KOS matrixes
    KOS_M_1=np.array([[1,0,0],[0,1,0],[0,0,1]])
    KOS_M_2=np.array([list(p_xvec),list(p_yvec),list(zvec)])
    #Calculate rotation matrix between KOS matrixes
    R=np.matmul(np.linalg.inv(KOS_M_1),KOS_M_2)
    #Apply rotation
    rot_coordinates=[]
    for c in coordinates:
        rot_coordinates.append(np.matmul(R,np.array(c)))
    # Calculate CENTER OF MASS and TRANSLATE
    molweights=[me.element(i).atomic_weight for i in moltypes]
    CoM=np.dot(np.array(molweights),np.array(rot_coordinates))/np.sum(molweights)
    translate_coordinates=rot_coordinates-CoM
        
    ## WRITE TO MOL2 ATOM LINES
    new_mol2_atom_lines=[i for i in mol2_atom_lines]
    test=[]
    for lin in range(len(new_mol2_atom_lines)):
        for c in range(len(new_mol2_atom_lines)):
            new_mol2_atom_lines[lin][2]=translate_coordinates[lin][0]
            new_mol2_atom_lines[lin][3]=translate_coordinates[lin][1]
            new_mol2_atom_lines[lin][4]=translate_coordinates[lin][2]
    
    return new_mol2_atom_lines

def load_data(pth):
    #load_data converts data from mol2 files. Reads out type, position, charge, bonds
    typ=[]
    position=[]
    charge=[]
    bonds=[]

    tripos_ATOM=False
    tripos_BONDS=False

    with open(pth) as fh:
        for line in fh:
            if "ATOM" in line:
                tripos_ATOM = True
            if "BOND" in line:
                tripos_ATOM = False
                tripos_BONDS = True
            if tripos_ATOM and not "ATOM" in line:
                # This is necessary, as atom names can be weird.
                # The type has a format like C.3, where we take the first part ["C","3"]
                typ.append(line.split()[5].split(".")[0])
                position.append([float(i) for i in line.split()[2:5]])
                charge.append(float(line.split()[-1]))
            if tripos_BONDS and not "BOND" in line:
                bonds.append(line.split()[1:])
    return [typ,position,charge,bonds]
#_____________________________________________________________________________