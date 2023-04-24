import pyscf.gto
import pyscf.scf
import pyscf.fci
import vayesta
import vayesta.dmet
from vayesta.misc.molecules import ring


# H6 ring
mol = pyscf.gto.Mole()
mol.atom = ring(atom='H', natom=6, bond_length=2.0)
mol.basis = 'sto-6g'
mol.output = 'pyscf.out'
mol.build()

# Hartree-Fock
mf = pyscf.scf.RHF(mol)
mf.kernel()

# Reference FCI
fci = pyscf.fci.FCI(mf)
fci.kernel()

# One-shot DMET
dmet = vayesta.dmet.DMET(mf, solver='VQE', maxiter=1)
with dmet.sao_fragmentation() as f:
    f.add_atomic_fragment([0,1])
    f.add_atomic_fragment([2,3])
    f.add_atomic_fragment([4,5])
dmet.kernel()

print("ZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZ")
print("ZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZ")
print("ZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZ")

# Self-consistent DMET
dmet_sc = vayesta.dmet.DMET(mf, solver='VQE', maxiter=7)
with dmet_sc.sao_fragmentation() as f:
    f.add_atomic_fragment([0,1])
    f.add_atomic_fragment([2,3])
    f.add_atomic_fragment([4,5])
dmet_sc.kernel()

#          |  E(corr)=                  -0.01928979 Ha
#          |  E(MF)=                    -2.42648134 Ha
#          |  E(nuc)=                   +2.90097636 Ha
#          |  E(tot)=                   -2.44577113 Ha
#Energies
#========
#  HF:                         -2.44362848 Ha
#  FCI:                        -2.87843154 Ha
#  DMET-VQE(1 iteration):          -2.69585884 Ha  (error= 182.6 mHa)
#  DMET-VQE(1 iteration, it=5,5):  -2.84857242 Ha  (error= 29.9 mHa)
#  DMET-VQE(self-consistent):      -2.44577113 Ha  (error= 432.7 mHa)

#THIS IS WHEN FCI IS USED FOR THE FRAGMENT SOLVER
#  HF:                         -2.44362848 Ha
#  FCI:                        -2.87843154 Ha
#  DMET-FCI(1 iteration):      -2.87133716 Ha  (error= 7.1 mHa)
#  DMET-FCI(self-consistent):  -2.87755919 Ha  (error= 0.9 mHa)

print("Energies")
print("========")
print("  HF:                    %+16.8f Ha" % mf.e_tot)
print("  FCI:                   %+16.8f Ha" % fci.e_tot)
print("  DMET-VQE(1 it.):       %+16.8f Ha  (error= %.1f mHa)" % (dmet.e_tot, 1000*(dmet.e_tot-fci.e_tot)))
print("  DMET-VQE(sc):          %+16.8f Ha  (error= %.1f mHa)" % (dmet_sc.e_tot, 1000*(dmet_sc.e_tot-fci.e_tot)))
