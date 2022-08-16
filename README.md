{
 "cells": [
  {
   "cell_type": "markdown",
   "id": "ae52dfe4",
   "metadata": {},
   "source": [
    "# Project: Snark That"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "8317b5cd",
   "metadata": {},
   "source": [
    "## Overview"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "e68b24f0",
   "metadata": {},
   "source": [
    "This script serves as an analogy to the world of zkSNARKs."
   ]
  },
  {
   "cell_type": "markdown",
   "id": "b0acb513",
   "metadata": {},
   "source": [
    "zero knowledge (zk), alludes to confirmation that some exclusive information is shared between two parties; without revealing the information to either party."
   ]
  },
  {
   "cell_type": "markdown",
   "id": "eec3eba4",
   "metadata": {},
   "source": [
    "SNARKs;\n",
    "\n",
    "Succinct: relative size of information transferred.\n",
    "Non-interactive: none-to-little interaction.\n",
    "ARguments: Verifier protected against computationally limited provers.\n",
    "Knowledge: Proof constructed with 'witness' at its core. \n"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "84888184",
   "metadata": {},
   "source": [
    "## Features"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "2434851f",
   "metadata": {},
   "source": [
    "Interface:"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "4d234e84",
   "metadata": {},
   "source": [
    "The interface acts as the UI that imports all the functionality from the snarkthat.py module."
   ]
  },
  {
   "cell_type": "markdown",
   "id": "94d0bf9e",
   "metadata": {},
   "source": [
    "Export R1CS"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "94f1c8a3",
   "metadata": {},
   "source": [
    "Prints a Rank-1 Constraint System specific to this function: (a**3 * 2b - 5)"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "a6caf644",
   "metadata": {},
   "source": [
    "Prove "
   ]
  },
  {
   "cell_type": "markdown",
   "id": "a1035b63",
   "metadata": {},
   "source": [
    "Generates a solution vector, specific to inputs."
   ]
  },
  {
   "cell_type": "markdown",
   "id": "0a7318fe",
   "metadata": {},
   "source": [
    "Verify"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "c60e82ef",
   "metadata": {},
   "source": [
    "computes logic gates; confirms prover-verifier validity or not."
   ]
  },
  {
   "cell_type": "markdown",
   "id": "c9b5aceb",
   "metadata": {},
   "source": [
    "## Why this is not secure?"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "00164963",
   "metadata": {},
   "source": [
    "'Succinct' is a holy grail in a world of ever-increasing transaction speeds and information-transfer. Hence, the Quadratic Arithmetic Program benefits R1CS conceptually; via processing the relationship (via some polynomials) as opposed to individual data points.\n",
    "\n",
    "In theory, this is apt for encryption technology. Though, the security risk is introduced due to measuring the function relationship among many polynomials over one datapoint (x,y) as opposed to a set of datapoints [xi,yi]. There is still a layer of security, however this metric over one data point allows a falsifier more probability to produce a trivial outcome. \n",
    "\n",
    "Hence, degree of confidentiality lost due to increased probability of 'guessing' inputs that will lead to a NullSpace, without having access to the 'secret' information."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "80ba41e8",
   "metadata": {},
   "outputs": [],
   "source": []
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.8.8"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 5
}
