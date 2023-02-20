import dataclasses

import numpy as np

import pyscf
import pyscf.ao2mo
import pyscf.ci
import pyscf.cc
import pyscf.mcscf
import pyscf.fci
import pyscf.fci.addons

from vayesta.core.util import *
from vayesta.core.types import Orbitals
from vayesta.core.types import VQE_WaveFunction
from vayesta.core.qemb.scrcoulomb import get_screened_eris_full
from .solver import ClusterSolver
from .cisd import CISD_Solver
from .cisd import UCISD_Solver

from aurora.chemistry.eos.vqe_scf import Qassolver
from aurora.chemistry.eos.vqe_scf import qasscf

from qiskit_nature.mappers.second_quantization import (
    BravyiKitaevMapper,
    JordanWignerMapper,
)



class VQE_Solver(ClusterSolver):

    @dataclasses.dataclass
    class Options(ClusterSolver.Options):
        threads: int = 1            # Number of threads for multi-threaded FCI
        max_cycle: int = 300
        lindep: float = None        # Linear dependency tolerance. If None, use PySCF default
        conv_tol: float = 1e-12     # Convergence tolerance. If None, use PySCF default
        solver_spin: bool = True    # Use direct_spin1 if True, or direct_spin0 otherwise
        fix_spin: bool = True           # If True, the given S^2 expectation value will be targeted
        fix_spin_value: float = None    # S^2 expectation value (None: Sz*(Sz+1), where 2*Sz = n(alpha)-n(beta))
        fix_spin_penalty: float = 1.0
        davidson_only: bool = True
        init_guess: str = 'default'
        init_guess_noise: float = 1e-5

    cisd_solver = CISD_Solver

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # MK: we need to use extra arguments for our Quasolver to mimic FullCI
        # FIXME better way to pass nelecas

#       solver = qasscf(
#           self.mf,
#           self.ncas,
#           (self.nelec//2, self.nelec//2),
#           kernel_max_iters=5,                #number of gates added in ADAPT-VQE before orbital rotation
#           approx_kernel_max_iters=1,
#           mapper=JordanWignerMapper(),
#           pool="fermionic",
#           method="statevector",
#       )
        solver = Qassolver(mol=self.mol, mapper=JordanWignerMapper())


#       solver = pyscf.mcscf.mc1step.CASSCF(mf_or_mol=self.mol, ncas=self.ncas, nelecas=self.nelec)
 #      pyscf.fci.direct_spin1.FCISolver
        self.log.debugv("type(solver)= %r", type(solver))
        # Set options
        if self.opts.init_guess == 'default':
            self.opts.init_guess = 'CISD'
        if self.opts.threads is not None:
            solver.threads = self.opts.threads
        if self.opts.conv_tol is not None:
            solver.conv_tol = self.opts.conv_tol
        if self.opts.lindep is not None:
            solver.lindep = self.opts.lindep
        if self.opts.max_cycle is not None:
            solver.max_cycle = self.opts.max_cycle
        if self.opts.davidson_only is not None:
            solver.davidson_only = self.opts.davidson_only
        # MK: is NONE for the example H6 case
        if self.opts.fix_spin:
            value = self.opts.fix_spin_value
            penalty = self.opts.fix_spin_penalty
            self.log.info("Fixing S^2 expectation value of FCI solver (value= %r, penalty= %f)", value, penalty)
            solver = pyscf.fci.addons.fix_spin_(solver, shift=penalty, ss=value)
        self.solver = solver

        # --- Results
        self.civec = None
        self.c0 = None
        self.c1 = None
        self.c2 = None

    def reset(self):
        super().reset()
        self.civec = None
        self.c0 = None
        self.c1 = None
        self.c2 = None

    @property
    def ncas(self):
        return self.cluster.norb_active

    @property
    def nelec(self):
        return 2*self.cluster.nocc_active

    def get_init_guess(self):
        return {'ci0' : self.civec}

    @deprecated()
    def get_t1(self):
        return self.get_c1(intermed_norm=True)

    @deprecated()
    def get_t2(self):
        return (self.c2 - einsum('ia,jb->ijab', self.c1, self.c1))/self.c0

    @deprecated()
    def get_c1(self, intermed_norm=False):
        norm = 1/self.c0 if intermed_norm else 1
        return norm*self.c1

    @deprecated()
    def get_c2(self, intermed_norm=False):
        norm = 1/self.c0 if intermed_norm else 1
        return norm*self.c2

    @deprecated()
    def get_l1(self, **kwargs):
        return None

    @deprecated()
    def get_l2(self, **kwargs):
        return None

    def get_cisd_init_guess(self):
        self.log.info("Generating intitial guess from CISD.")
        cisd = self.cisd_solver(self.mf, self.fragment, self.cluster)
        cisd.kernel()
        ci = cisd.wf.as_fci().ci
        if self.opts.init_guess_noise:
            ci += self.opts.init_guess_noise * np.random.random(ci.shape)
        return ci

    def kernel(self, ci0=None, eris=None, seris_ov=None):
        """Run FCI kernel."""

        if eris is None: eris = self.get_eris()
        heff = self.get_heff(eris)
        # Screening
        if seris_ov is not None:
            eris = get_screened_eris_full(eris, seris_ov, log=self.log)

        #MK: we don't want a CISD init guess now
#       if ci0 is None and self.opts.init_guess == 'CISD':
#           ci0 = self.get_cisd_init_guess()

        t0 = timer()
        #self.solver.verbose = 10
#       e_fci, self.civec = self.solver.kernel(heff, eris, self.ncas, (self.nelec//2,self.nelec//2), ci0=ci0)
        e_fci, self.civec = self.solver.kernel(heff, eris, self.ncas, (self.nelec//2,self.nelec//2) )
        # FIXME fix parameter converged
   #    if not self.solver.converged:
   #        self.log.error("FCI not converged!")
   #    else:
   #        self.log.debugv("FCI converged.")
        self.log.timing("Time for FCI: %s", time_string(timer()-t0))
        self.log.debug("E(CAS)= %s", energy_string(e_fci))
        # TODO: This requires the E_core energy (and nuc-nuc repulsion)
        self.e_corr = np.nan
        # FIXME add the converged attribute
   #    self.converged = self.solver.converged
   #    self.c0, self.c1, self.c2 = self.get_cisd_amps(self.civec)
        nocc, nvir = self.cluster.nocc_active, self.cluster.nvir_active
        self.c0 = 1.00
        self.c1 = np.zeros((nocc, nvir))
        self.c2 = np.zeros((nocc, nvir, nocc, nvir))

        self.log.info("FCI: weight of reference determinant= %.8g", abs(self.c0))
#       s2, mult = self.solver.spin_square(self.civec, self.ncas, self.nelec)
#       print(self.civec.shape, self.ncas, self.nelec)
        s2, mult = 0.00, 1.00
#       s2, mult = pyscf.fci.direct_spin1.FCISolver.spin_square(self.civec, self.ncas, self.nelec)
        if not isinstance(self, UVQE_Solver) and (abs(s2) > 1e-8):
            if abs(s2) > 0.1:
                self.log.critical("FCI: S^2= %.10f  multiplicity= %.10f", s2, mult)
                raise RuntimeError("Spin restricted FCI encountered solution with S^2 >> 0")
            self.log.warning("FCI: S^2= %.10f  multiplicity= %.10f", s2, mult)
        else:
            self.log.info("FCI: S^2= %.10f  multiplicity= %.10f", s2, mult)
        mo = Orbitals(self.cluster.c_active, occ=self.cluster.nocc_active)
        self.wf = VQE_WaveFunction(qas_solver=self.solver, mo=mo, state=self.civec)

    #def get_cisd_amps(self, civec):
    #    cisdvec = pyscf.ci.cisd.from_fcivec(civec, self.ncas, self.nelec)
    #    c0, c1, c2 = pyscf.ci.cisd.cisdvec_to_amplitudes(cisdvec, self.ncas, self.cluster.nocc_active)
    #    c1 = c1/c0
    #    c2 = c2/c0
    #    return c0, c1, c2

    def get_cisd_amps(self, civec, intermed_norm=False):
        nocc, nvir = self.cluster.nocc_active, self.cluster.nvir_active
        t1addr, t1sign = pyscf.ci.cisd.t1strs(self.ncas, nocc)
        c0 = civec[0,0]
        c1 = civec[0,t1addr] * t1sign
        c2 = einsum('i,j,ij->ij', t1sign, t1sign, civec[t1addr[:,None],t1addr])
        c1 = c1.reshape(nocc,nvir)
        c2 = c2.reshape(nocc,nvir,nocc,nvir).transpose(0,2,1,3)
        if intermed_norm:
            c1 = c1/c0
            c2 = c2/c0
            c0 = 1.0
        return c0, c1, c2

    def _debug_exact_wf(self, wf):
        from pyscf.fci.addons import transform_ci
        mo = Orbitals(self.cluster.c_active, occ=self.cluster.nocc_active)
        nelec = wf.mo.nelec
        if mo.nelec != nelec:
            raise NotImplementedError
        u = self.fragment.get_overlap('mo|cluster')
        ci = transform_ci(wf.ci, nelec, u)
        wf = FCI_WaveFunction(mo, ci)
        self.wf = wf
        self.converged = True


class UVQE_Solver(VQE_Solver):
    """VQE with UHF orbitals."""

    pass

