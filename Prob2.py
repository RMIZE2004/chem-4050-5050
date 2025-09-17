import sys
print("Python:", sys.executable)
import numpy as np
#all cartesian coordinates in Angstroms for H2O, C6H6, and H2
molecule1 = {"O": [0.000000, 0.000000, 0.1173]
            , "H1":[0.0000,	0.7572,	-0.4692],"H2":[	0.0000,	-0.7572, -0.4692]}
molecule2 = {
    "C1": [0.0000, 1.3970, 0.0000],
    "C2": [1.2098, 0.6985, 0.0000],
    "C3": [1.2098, -0.6985, 0.0000],
    "C4": [0.0000, -1.3970, 0.0000],
    "C5": [-1.2098, -0.6985, 0.0000],
    "C6": [-1.2098, 0.6985, 0.0000],
    "H1": [0.0000, 2.4810, 0.0000],
    "H2": [2.1486, 1.2405, 0.0000],
    "H3": [2.1486, -1.2405, 0.0000],
    "H4": [0.0000, -2.4810, 0.0000],
    "H5": [-2.1486, -1.2405, 0.0000],
    "H6": [-2.1486, 1.2405, 0.0000]
}
molecule3 = {"H1": [0.0000, 0.0000, 0.0000],
             "H2": [0.0000, 0.0000, 0.7414]}

print(molecule1)
#condition for warning long bond length only once (otherwise will return for every combination of atoms in benzene)
warned_long_bond = False
def compute_bond_length(coord1, coord2,vertrue=False):
    #distance formula and bond length condition
    global warned_long_bond
    r1 = np.array(coord1, dtype=float)
    r2 = np.array(coord2, dtype=float)
    d = float(np.linalg.norm(r2 - r1))
    if d > 2.0 and not warned_long_bond:
        print("Warning: Bond length Too long! (>2.0 Angstroms)")
        warned_long_bond = True
    if vertrue:
        print(f"Bond length: {d:.2f} Angstroms")

    return d
#computing bond angle with dot product calculation
def compute_bond_angle(coord1, coord2, coord3, right_atol=1e-2):
    r1=np.array(coord1,dtype=float)
    r2=np.array(coord2,dtype=float)
    r3=np.array(coord3,dtype=float)
    v1=r1-r2
    v2=r3-r2
    cos_theta=np.dot(v1,v2)/(np.linalg.norm(v1)*np.linalg.norm(v2))
    angle=np.arccos(cos_theta)
    angle_deg=np.degrees(angle)
    #classifying angle type for the calculated values
    if angle_deg < 90- right_atol:
        kind = "acute"
    elif 90-right_atol<= angle_deg <=90+right_atol:
        kind = "right"
    else:
        kind = "obtuse"
    return angle_deg, kind
#all molecules in a dictionary for use in loop over all of them
molecules= {"H2O": molecule1, "C6H6": molecule2, "H2": molecule3}
#cutoff lengths that will allow displaying of only real bond length connections and calculation of angles of real bonds
def cutoff_for(name):
     if name=="H2O":
          return 1.10
     if name=="C6H6": 
          return 1.50
     return 1.05
def calculate_all_bond_lengths(molecule, mol_name):
    max_pair_length=cutoff_for(mol_name)
    names=list(molecule.keys())
    results=[]
    #loop over all pairs of atoms in a molecule with the condition so that only real bonds will be output
    for i in range (len(names)):
         for j in range(i+1, len(names)):
              a1, a2= names[i], names[j]
              d= compute_bond_length(molecule[a1], molecule[a2])
              if d<= max_pair_length:
                    results.append((a1, a2, d))
    results.sort(key=lambda t: t[2])
    return results
#final list of all bond lengths for each named molecule
for name, mol in molecules.items():
     bonds= calculate_all_bond_lengths(mol, name)
     print(f"\n{name} bond lengths (<= {cutoff_for(name):.3f} Angstroms):")
     for a1, a2, d in bonds:
        print(f"{a1}-{a2}: {d:.2f} Angstroms")
#calculating all bond angles using only the bonds of real connectivties in the molecules
from itertools import combinations
def calculate_all_bond_angles(molecule, mol_name):
    bonds= calculate_all_bond_lengths(molecule, mol_name)
    adj={name: set() for name in molecule.keys()}
    for a1, a2, _ in bonds:
        adj[a1].add(a2)
        adj[a2].add(a1)
    angles=[]
    for B, nbrs in adj.items():
        if len(nbrs) < 2:
            continue
        for A, C in combinations(nbrs, 2):
            angle_deg, kind = compute_bond_angle(molecule[A], molecule[B], molecule[C])
            angles.append((A, B, C, angle_deg, kind))
    angles.sort(key=lambda t: t[3])
    return angles
for name, mol in molecules.items():
    angles= calculate_all_bond_angles(mol, name)
    print(f"\n{name} bond angles:")
    for A, B, C, angle_deg, kind in angles:
        print(f"{A}-{B}-{C}: {angle_deg:.2f} degrees ({kind})")
