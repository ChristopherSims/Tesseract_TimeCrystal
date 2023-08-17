import numpy as np
import sympy as sp
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib import colors
from matplotlib.animation import FuncAnimation
from matplotlib.ticker import MultipleLocator, AutoMinorLocator
from IPython import display
from typing import Sequence, Tuple, List, Iterator
from typing import AbstractSet, Any, Dict, Optional, Tuple
import cirq
from random import randint
from cirq import Y, PhasedXZGate, ZZ,H,X,Ry,Rx,Rz,XX,YY,ZZPowGate
#import cirq.ops.fsim_gate.PhasedFSimGate
import cirq.ops.fsim_gate as FSIM
#import cirq.PhasedFSimGate 
import cirq_google, qsimcirq
from scipy.interpolate import interp1d
from mpl_toolkits.axes_grid1 import make_axes_locatable


class QubitNoiseModel(cirq.NoiseModel):

    def __init__(self, depolarizing_error_rate, phase_damping_error_rate, amplitude_damping_error_rate):
        self._depolarizing_error_rate = depolarizing_error_rate
        self._phase_damping_error_rate = phase_damping_error_rate
        self._amplitude_damping_error_rate = amplitude_damping_error_rate

    def noisy_operation(self, op):
        n_qubits = len(op.qubits)
        depolarize_channel = cirq.depolarize(self._depolarizing_error_rate, n_qubits=n_qubits)
        phase_damping_channel = cirq.phase_damp(self._phase_damping_error_rate).on_each(op.qubits)
        amplitude_damping_channel = cirq.amplitude_damp(self._amplitude_damping_error_rate).on_each(op.qubits)
        return [op, depolarize_channel.on(*op.qubits), phase_damping_channel, amplitude_damping_channel]
def simulate_circuit_list(
                            circuit_list: Sequence[cirq.Circuit],
                            maxk: int = None
                            ) -> np.ndarray:

    simulator = cirq.Simulator()
    circuit_positions = {len(c) - 1 for c in circuit_list}
    circuit = circuit_list[-1]

    probabilities = []
    for k, step in enumerate(
        simulator.simulate_moment_steps(circuit=circuit)
    ):

        # add the state vector if the number of moments simulated so far is equal
        #   to the length of a circuit in the circuit_list
        #if maxk != None:
        #    print((k/maxk)*100)
        if k in circuit_positions:
            probabilities.append(np.abs(step.state_vector()) ** 2)

    return np.asarray(probabilities)

def get_polarizations(
    probabilities: np.ndarray,
    num_qubits: int,
    initial_states: np.ndarray = None,
) -> np.ndarray:
    """Get polarizations from matrix of probabilities, possibly autocorrelated on
        the initial state.

    A polarization is the marginal probability for a qubit to measure zero or one,
        over all possible basis states, scaled to the range [-1. 1].

    Args:
        probabilities: `np.ndarray` of shape (:, cycles, 2**qubits)
            representing probability to measure each bit string
        num_qubits: the number of qubits in the circuit the probabilities
            were generated from
        initial_states: `np.ndarray` of shape (:, qubits) representing the initial
            state for each dtc circuit list

    Returns:
        `np.ndarray` of shape (:, cycles, qubits) that represents each
            qubit's polarization

    """
    # prepare list of polarizations for each qubit
    polarizations = []
    for qubit_index in range(num_qubits):
        # select all indices in range(2**num_qubits) for which the
        #   associated element of the statevector has qubit_index as zero
        shift_by = num_qubits - qubit_index - 1
        state_vector_indices = [
            i for i in range(2**num_qubits) if not (i >> shift_by) % 2
        ]

        # sum over all probabilities for qubit states for which qubit_index is zero,
        #   and rescale them to [-1,1]
        polarization = (
            2.0
            * np.sum(
                probabilities.take(indices=state_vector_indices, axis=-1),
                axis=-1,
            )
            - 1.0
        )
        polarizations.append(polarization)

    # turn polarizations list into an array,
    #   and move the new, leftmost axis for qubits to the end
    polarizations = np.moveaxis(np.asarray(polarizations), 0, -1)

    # flip polarizations according to the associated initial_state, if provided
    #   this means that the polarization of a qubit is relative to it's initial state
    if initial_states is not None:
        initial_states = 1 - 2.0 * initial_states
        polarizations = initial_states * polarizations

    return polarizations

class FSIM(cirq.PhasedFSimGate):
    def _unitary_(self) -> Optional[np.ndarray]:
        if self._is_parameterized_():
            return None
        a = math.cos(self.theta)
        b = 1j * math.sin(self.theta)
        c = cmath.exp(-1j * self.phi)
        
        f2 = cmath.exp(1j * self.chi)
        f3 = cmath.exp(-1j * self.chi)
        # fmt: off
        return np.array(
            [
                [1, 0, 0, 0],
                [0,  a, f2 * b, 0],
                [0, f3 * b, a, 0],
                [0, 0, 0, c],
            ]
        )

def Circuit_list_2D(
                    THETA: float,
                    PHI: float,
                    qubits, 
                    sq: int,
                    cycles: int,
                    n_qubits: int,
                    gate_rotate: cirq.ops,
                    gate_coupling: cirq.ops
                    ) -> List[cirq.Circuit]:
    circuit = cirq.Circuit()
    ## initial operation
    initial_operations = []
    init_ops = []
    for op in init_ops:
        initial_operations.append(cirq.Moment(op))
    ## Initial U gate
    sequence_operations = []
    for i in range(0, sq):
        for j in range(0,sq):
            sequence_operations.append(
                gate_rotate(rads = THETA).on(qubits[i][j])
            )
    u_cycle = [cirq.Moment(sequence_operations)]

    #FSIM Gate
    even_qubit_moment = []
    odd_qubit_moment = []
    phi = PHI/np.pi
    coupling_gate = gate_coupling**(phi)
    for i in range(1, sq-1, 2):
        for j in range(1,sq-1,2):
            # u_cycle.append(cirq.Moment(coupling_gate.on(qubits[i], qubits[sq*(i+1) + j])))
            # u_cycle.append(cirq.Moment(coupling_gate.on(qubits[i], qubits[sq*(i+1) + j+1])))
            # u_cycle.append(cirq.Moment(coupling_gate.on(qubits[i], qubits[sq*i + j+1])))
            u_cycle.append(cirq.Moment(coupling_gate.on(qubits[i][j], qubits[i+1][j])))   
            u_cycle.append(cirq.Moment(coupling_gate.on(qubits[i][j], qubits[i][j+1])))
            u_cycle.append(cirq.Moment(coupling_gate.on(qubits[i][j], qubits[i][j-1])))
            u_cycle.append(cirq.Moment(coupling_gate.on(qubits[i][j], qubits[i-1][j])))
            
            u_cycle.append(cirq.Moment(coupling_gate.on(qubits[i][j], qubits[i+1][j+1])))
            u_cycle.append(cirq.Moment(coupling_gate.on(qubits[i][j], qubits[i+1][j-1])))
            u_cycle.append(cirq.Moment(coupling_gate.on(qubits[i][j], qubits[i-1][j-1])))
            u_cycle.append(cirq.Moment(coupling_gate.on(qubits[i][j], qubits[i-1][j+1])))
    for i in range(2, sq-1, 2):
        for j in range(2,sq-1,2):
            # u_cycle.append(cirq.Moment(coupling_gate.on(qubits[i], qubits[sq*(i+1) + j])))
            # u_cycle.append(cirq.Moment(coupling_gate.on(qubits[i], qubits[sq*(i+1) + j+1])))
            # u_cycle.append(cirq.Moment(coupling_gate.on(qubits[i], qubits[sq*i + j+1])))
            u_cycle.append(cirq.Moment(coupling_gate.on(qubits[i][j], qubits[i+1][j])))   
            u_cycle.append(cirq.Moment(coupling_gate.on(qubits[i][j], qubits[i][j+1])))
            u_cycle.append(cirq.Moment(coupling_gate.on(qubits[i][j], qubits[i][j-1])))
            u_cycle.append(cirq.Moment(coupling_gate.on(qubits[i][j], qubits[i-1][j])))
            
            u_cycle.append(cirq.Moment(coupling_gate.on(qubits[i][j], qubits[i+1][j+1])))
            u_cycle.append(cirq.Moment(coupling_gate.on(qubits[i][j], qubits[i+1][j-1])))
            u_cycle.append(cirq.Moment(coupling_gate.on(qubits[i][j], qubits[i-1][j-1])))
            u_cycle.append(cirq.Moment(coupling_gate.on(qubits[i][j], qubits[i-1][j+1])))
    u_cycle.append(cirq.Moment(even_qubit_moment))
    u_cycle.append(cirq.Moment(odd_qubit_moment))
    after_moment = []
    # for i in range(0,sq):
    #     for j in range(0,sq):
    #         after_moment.append(H(qubits[i][j]))
    #u_cycle.append(cirq.Moment(after_moment))
    circuit_list = []
    total_circuit = cirq.Circuit(initial_operations)
    circuit_list.append(total_circuit.copy())
    for _ in range(cycles):
        for moment in u_cycle:
            total_circuit.append(moment)
        circuit_list.append(total_circuit.copy())

    return circuit_list, len(circuit_list[1])*len(circuit_list)


def Plot_Pol(dtc_z,num_cycles,N_QUBITS):

    #print(dtc_z.shape)
    #fig,ax = plt.figure(figsize=(12,8))
    fig,ax = plt.subplots(ncols=1,nrows=1,figsize=(12,8))
    # divider = make_axes_locatable(ax)
    # cax = divider.append_axes('right', size='5%', pad=0.05)
    cm ='bwr'
    im = ax.imshow(dtc_z,cmap = cm,vmin=-1,vmax=1)
    #im = ax.imshow(dtc_z)
    plt.rcParams.update({'font.size': 15})
    #plt.rcParams['text.usetex'] = True
    # cbar = mpl.colorbar.ColorbarBase(ax, cmap=cm,
    #                        norm=mpl.colors.Normalize(vmin=-1, vmax=1),boundaries=np.linspace(-1,1,5))
    #plt.colorbar(im)
    #fig.colorbar(im,orientation='vertical')
    # cbar0 = plt.colorbar(im)
    # cbar0.set_ticks([-1,0,1])
    # Graph results
    ax.set_xlabel('Floquet time step',fontsize=24)
    ax.xaxis.labelpad = 10
    ax.set_ylabel('Qubit',fontsize=24)
    ax.set_xticks(np.append(np.arange(0, num_cycles , 10),num_cycles),fontsize=18)
    #yticks_new = np.arange(0, N_QUBITS, sq)
    yticks_new = np.array([0,3,5,6,9,10,12,15])
    ax.set_yticks(yticks_new)
    ax.set_yticklabels(yticks_new+1,fontsize=18)
    ax.tick_params(axis='both', which='major', labelsize=24)
    plt.ylim([N_QUBITS-0.5,-0.5])
    ax.xaxis.set_label_position('bottom')
    #plt.savefig('1D_POL_.png',bbox_inches='tight')
    #plt.show()
    return fig