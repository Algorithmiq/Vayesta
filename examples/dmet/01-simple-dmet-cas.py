import pyscf.gto
import pyscf.scf
import pyscf.fci
import vayesta
import vayesta.dmet


mol = pyscf.gto.M(atom = 'Li 0 0 0; H 2 0 0', basis = 'sto-6g', verbose=1)

# Hartree-Fock
mf = pyscf.scf.RHF(mol)
mf.kernel()

# Reference FCI
fci = pyscf.fci.FCI(mf)
fci.kernel()

# One-shot DMET
dmet = vayesta.dmet.DMET(mf, solver='FCI', maxiter=1)
with dmet.cas_fragmentation() as f:
    f.add_cas_fragment(4,2)
dmet.kernel()

print("ZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZ")
print("ZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZ")
print("ZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZ")

# Self-consistent DMET
dmet_sc = vayesta.dmet.DMET(mf, solver='FCI', maxiter=20)
with dmet_sc.cas_fragmentation() as f:
    f.add_cas_fragment(4,2)
dmet_sc.kernel()

print("Energies")
print("========")
print("  HF:                    %+16.8f Ha" % mf.e_tot)
print("  FCI:                   %+16.8f Ha" % fci.e_tot)
print("  DMET-VQE(1 it.):       %+16.8f Ha  (error= %.1f mHa)" % (dmet.e_tot, 1000*(dmet.e_tot-fci.e_tot)))
print("  DMET-VQE(sc):          %+16.8f Ha  (error= %.1f mHa)" % (dmet_sc.e_tot, 1000*(dmet_sc.e_tot-fci.e_tot)))
