import unittest
import numpy as np
from vayesta import lattmod, dmet
import pyscf.gto
import pyscf.scf
import pyscf.tools.ring

def make_test_molecular(atoms, basis, solver, fragment_type, cc, known_values, fragments=None, **kwargs):

    class DMETMolecularTests(unittest.TestCase):

        @classmethod
        def setUpClass(cls):
            cls.mol = pyscf.gto.Mole()
            cls.mol.atom = atoms
            cls.mol.basis = basis
            cls.mol.build()

            cls.mf = pyscf.scf.RHF(cls.mol)
            cls.mf.kernel()

            cls.dmet = dmet.DMET(cls.mf, solver=solver, charge_consistent=cc, **kwargs)
            # Ensure that we don't spam with output.
            cls.dmet.log.setLevel(50)
            if fragments is None:
                cls.dmet.make_all_atom_fragments()
            else:
                for x in fragments:
                    cls.dmet.make_atom_fragment(x)
            cls.dmet.kernel()

        @classmethod
        def tearDownClass(cls):
            del cls.mol, cls.mf, cls.dmet

        def test_energy(self):
            self.assertAlmostEqual(self.dmet.e_tot, known_values['e_tot'], 6)

    return DMETMolecularTests

def make_test_Hring(natom, d, fragsize, *args, **kwargs):
    ring = pyscf.tools.ring.make(natom, d)
    atom = [('H %f %f %f' % xyz) for xyz in ring]
    fragments = [list(range(x, x+fragsize)) for x in range(0, natom, fragsize)]
    return make_test_molecular(atom, *args, fragments=fragments, **kwargs)


dmet_Hring_sto6g_FCI_IAO_cc_Test = make_test_Hring(
        10,
        1.0,
        2,
        "sto-6g",
        "FCI",
        "IAO",
        True,
        {'e_tot': -5.421103410208376},
)

dmet_Hring_sto6g_FCI_IAO_nocc_Test = make_test_Hring(
        10,
        1.0,
        2,
        "sto-6g",
        "FCI",
        "IAO",
        False,
        {'e_tot': -5.421192647967002},
)

dmet_Hring_sto6g_FCI_IAO_all_nocc_Test = make_test_Hring(
        10,
        1.0,
        2,
        "sto-6g",
        "FCI",
        "IAO",
        False,
        {'e_tot': -5.422668582405825},
        bath_type='ALL',
)

dmet_Hring_sto6g_FCI_IAO_BNO_nocc_Test = make_test_Hring(
        10,
        1.0,
        2,
        "sto-6g",
        "FCI",
        "IAO",
        False,
        {'e_tot': -5.421192648085972},
        bath_type='MP2-BNO',
        bno_threshold=np.inf,
)


if __name__ == '__main__':
    print('Running %s' % __file__)
    unittest.main()
