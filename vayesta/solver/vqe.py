import dataclasses

import numpy as np

from vayesta.core.util import *
from vayesta.core.types import Orbitals
from vayesta.core.types import VQE_WaveFunction
from vayesta.core.qemb.scrcoulomb import get_screened_eris_full
from .solver import ClusterSolver

from aurora.chemistry.eos.vqe_scf import Qassolver

from qiskit_nature.mappers.second_quantization import (
    BravyiKitaevMapper,
    JordanWignerMapper,
)



class VQE_Solver(ClusterSolver):

    @dataclasses.dataclass
    class Options(ClusterSolver.Options):
        threads: int = 1            # Number of threads for multi-threaded VQE
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

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        solver = Qassolver(mol=self.mol, mapper=JordanWignerMapper(), kernel_max_iters=10, approx_kernel_max_iters=1)
        self.log.debugv("type(solver)= %r", type(solver))
        # Set options
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
        self.solver = solver

        # --- Results
        self.state = None

    def reset(self):
        super().reset()
        self.state = None

    @property
    def ncas(self):
        return self.cluster.norb_active

    @property
    def nelec(self):
        return 2*self.cluster.nocc_active

    def get_init_guess(self):
        return {'state0' : self.state}

    def kernel(self, eris=None, seris_ov=None):
        """Run VQE kernel."""

        if eris is None: eris = self.get_eris()
        heff = self.get_heff(eris)
        # Screening
        if seris_ov is not None:
            eris = get_screened_eris_full(eris, seris_ov, log=self.log)

        t0 = timer()
        #self.solver.verbose = 10
        # FIXME better way of passing nelec tuple
        e_vqe, self.state = self.solver.kernel(heff, eris, self.ncas, (self.nelec//2,self.nelec//2) )
        self.log.timing("Time for VQE: %s", time_string(timer()-t0))
        self.log.debug("E(CAS)= %s", energy_string(e_vqe))
        # TODO: This requires the E_core energy (and nuc-nuc repulsion)
        self.e_corr = np.nan

        # FIXME use the spin_square function
#       s2, mult = self.solver.spin_square(self.civec, self.ncas, self.nelec)
        s2, mult = 0.00, 1.00
        if abs(s2) > 1e-8:
            if abs(s2) > 0.1:
                self.log.critical("VQE: S^2= %.10f  multiplicity= %.10f", s2, mult)
                raise RuntimeError("Spin restricted VQE encountered solution with S^2 >> 0")
            self.log.warning("VQE: S^2= %.10f  multiplicity= %.10f", s2, mult)
        else:
            self.log.info("VQE: S^2= %.10f  multiplicity= %.10f", s2, mult)
        mo = Orbitals(self.cluster.c_active, occ=self.cluster.nocc_active)
        self.wf = VQE_WaveFunction(qas_solver=self.solver, mo=mo, state=self.state)


class UVQE_Solver(VQE_Solver):
    """VQE with UHF orbitals."""

    pass

