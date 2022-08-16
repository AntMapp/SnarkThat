{
 "cells": [
  {
   "cell_type": "markdown",
   "id": "1d7a2faf",
   "metadata": {},
   "source": [
    "# Project: Snark That"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "c50eae11",
   "metadata": {},
   "source": [
    "## Overview"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "40bba144",
   "metadata": {},
   "source": [
    "This script serves as an analogy to the world of zkSNARKs."
   ]
  },
  {
   "cell_type": "markdown",
   "id": "5bfc5a32",
   "metadata": {},
   "source": [
    "zero knowledge (zk), alludes to confirmation that some exclusive information is shared between two parties; without revealing the information to either party."
   ]
  },
  {
   "cell_type": "markdown",
   "id": "458a41a9",
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
   "id": "d491ac32",
   "metadata": {},
   "source": [
    "## Features"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "bab947cf",
   "metadata": {},
   "source": [
    "Export R1CS"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "d2958cb4",
   "metadata": {},
   "source": [
    "Prints a Rank-1 Constraint System specific to this function: (a**3 * 2b - 5)"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "44989813",
   "metadata": {},
   "source": [
    "Prove "
   ]
  },
  {
   "cell_type": "markdown",
   "id": "db8b498b",
   "metadata": {},
   "source": [
    "Generates a solution vector, specific to inputs."
   ]
  },
  {
   "cell_type": "markdown",
   "id": "6bffa15e",
   "metadata": {},
   "source": [
    "Verify"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "b5ed8fca",
   "metadata": {},
   "source": [
    "computes logic gates; confirms prover-verifier validity or not."
   ]
  },
  {
   "cell_type": "markdown",
   "id": "074fdd28",
   "metadata": {},
   "source": [
    "## Why this is not secure?"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "e2cdb2af",
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
   "id": "bd01b9e2",
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
