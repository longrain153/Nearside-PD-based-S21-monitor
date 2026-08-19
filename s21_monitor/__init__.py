"""In-service monitoring of a high-baud-rate coherent transmitter using a
low-bandwidth photodetector (PD).

The transmitter is modeled as a 2x2 widely linear FIR filter L_ij[l]
(the channel of interest, COI) acting on the digital drive signals
(xI, xQ).  The PD is modeled as an ideal square-law detector followed by
an unknown low-pass FIR filter M[m] (the observation channel, OC, which
also absorbs the ADC response).  Because the square law is nonlinear,
both L_ij[l] and M[m] can be recovered jointly from (xI, xQ, z) with an
LMS / error-backpropagation scheme, even though M[m] has far less
bandwidth than the signal.
"""

from .filters import (
    lowpass_fir,
    fractional_delay_fir,
    rrc_fir,
    freq_response,
    causal_conv,
)
from .transmitter import WidelyLinearTransmitter, make_transmitter
from .photodetector import photodetect
from .monitor import MonitorResult, fit_monitor
from .metrics import analyze_iq, branch_responses

__all__ = [
    "lowpass_fir",
    "fractional_delay_fir",
    "rrc_fir",
    "freq_response",
    "causal_conv",
    "WidelyLinearTransmitter",
    "make_transmitter",
    "photodetect",
    "MonitorResult",
    "fit_monitor",
    "analyze_iq",
    "branch_responses",
]
