# Block B description
Note: Block B, also referred as final block (FB), and class in the MadWeight world. 
Description inspired by MadWeight.

This block corresponds to an initial state with Bjorken fractions \f$q_1\f$ and \f$q_2\f$, 
and a final state with a visible particle with momenta \f$p_2\f$, and an invisible particle 
with momenta \f$p_1\f$, both coming from a resonance with mass \f$s_{12}\f$. Extra radiation (ISR) it is also allowed.

The goal of this Block is to address the change of variables needed to pass from the standard phase-space
parametrization to the \f$\frac{1}{4\pi E_1} ds_{12} \times J\f$ parametrization. Per integration point 
in CUBA, this Block outputs the value of the jacobian, J, and the four momenta of the invisible particle, 
\f$p_1\f$. The system of equations needed to compute \f$p_1\f$ is described below. 


### Change of variables

From the standard phase-space parametrisation:

\f[
  dq_1 dq_2\frac{d^3 p_1}{(2\pi)^3 2E_1}(2\pi)^4 \delta^4 (P_{in}-P_{out})
\f]

where \f$P_{in} and P_{out}\f$ are the total four-momenta in the initial and final states, to the 
following parametrisation: 

\f[
  \frac{1}{4\pi E_1} ds_{12} \times J
\f]

where the jacobian, J, is given by:

\f[
  J = \frac{E_1}{s} \left| p_{2z} E_1 -  E_2 p_{1z} \right|^{-1}
\f]

### Parameters

- Collision energy.

### Inputs

- Visible particle, with 4-momentum \f$p_2\f$
- The invariant \f$s_{12}\f$ as output from the Flatter or NarrowWidthApproximation module.

### Outputs

- Invisible particle, with 4-momentum \f$p_1\f$ (up to two solutions possible)
- Jacobian, one per solution.    

### System of equations to compute \f$p_1\f$

The integrator throws random points in the invariant (\f$s_{12}\f$) while \f$p_2\f$ is a known quantity. The equations to 
compute \f$p_1\f$ are:

\f{eqnarray}{
 s_{12} &=& (p_1 + p_2)^2 = M_{1}^{2} + M_{2}^2 + 2 E_1 E_2 + 2 p_{1x}p_{2x} + 2p_{1y}p_{2y} + p_{1z}p_{2z} \\
 p_{1x} &=& - p_{Tx} \\
 p_{1y} &=& - p_{Ty} \\
 p_1^2 &=& 0 \Leftrightarrow E_{1}^2 = p_{1x}^2 + p_{1y}^2 + p_{1z}^2
\f}

Where \f$\vec{p}_T\f$ is the total transverse momentum of the visible particles. Using the values of \f$p_{1x}, p_{1y}\f$ from equations (2) and (3), equation (1) can be written as \f$p_{1z} = A - B E_1\f$, where A and B are:
\f{eqnarray}{
 A &=& \frac{s_{12} - M_{2}^2 + 2(p_{Tx}p_{2x} + p_{Ty}p_{2y})}{2 p_{2z}} \\
 B &=& \frac{E_2}{p_{2z}} \\
\f}

Finally equation (4) can be written as \f$(1 - B) E_{1}^2 + 2AB E_1 - C = 0\f$, where \f$C = p_{Tx}^{2} + p_{Ty}^{2}\f$.

Each solution of the quadratic equation with a positive value of \f$E_1\f$ is taken.

