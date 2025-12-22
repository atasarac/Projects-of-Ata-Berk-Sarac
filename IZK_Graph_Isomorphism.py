import numpy as np
import networkx as nx
import random

"""
This is a project of mine for the course Computational Mathematics I've completed.
"""

# --- Mathematical Utilities and Verification ---

def random_permutation(n: int) -> np.ndarray:
    """Generates a random permutation of size n."""
    sigma = np.arange(n)
    np.random.shuffle(sigma)
    return sigma

def apply_permutation(G_matrix: np.ndarray, sigma: np.ndarray) -> np.ndarray:
    """
    Apply the permutation sigma to the rows and columns of G.
    Mathematically: H = P * G * P^T
    """
    return G_matrix[sigma][:, sigma]

def invert_permutation(sigma: np.ndarray) -> np.ndarray:
    """Calculate the inverse permutation sigma^{-1}."""
    return np.argsort(sigma)

# --- Actors of the Protocol ---

class HonestProver:
    def __init__(self, G1: np.ndarray, G2: np.ndarray, phi: np.ndarray):
        self.G1 = G1
        self.G2 = G2
        self.phi = phi # The secret (isomorphism such that G2 = phi(G1))
        self.n = len(G1)
        self.sigma = None # Temporary commitment

    def commit(self) -> np.ndarray:
        """Step 1 : The Prover generates an isomorphic graph H."""
        self.sigma = random_permutation(self.n)
        H = apply_permutation(self.G1, self.sigma)
        return H

    def respond(self, challenge: int) -> np.ndarray:
        """Step 2 : The Prover responds to the challenge c."""
        if challenge == 0:
            return self.sigma
        else:
            phi_inv = invert_permutation(self.phi)
            return phi_inv[self.sigma]

class Cheater:
    def __init__(self, G1: np.ndarray, G2: np.ndarray):
        self.G1 = G1
        self.G2 = G2
        self.n = len(G1)
        self.sigma = None
        self.predicted_challenge = None

    def commit(self) -> np.ndarray:
        self.predicted_challenge = random.randint(0, 1)
        self.sigma = random_permutation(self.n)
        
        if self.predicted_challenge == 0:
            return apply_permutation(self.G1, self.sigma)
        else:
            return apply_permutation(self.G2, self.sigma)

    def respond(self, challenge: int) -> np.ndarray:
        if challenge != self.predicted_challenge:
            return self.sigma 
        return self.sigma

class StrictVerifier:
    def __init__(self, G1: np.ndarray, G2: np.ndarray):
        self.G1 = G1
        self.G2 = G2

    def check(self, H: np.ndarray, rho: np.ndarray, challenge: int) -> bool:
        """Verify if rho transforms the target graph (G1 or G2) into H."""
        target_graph = self.G1 if challenge == 0 else self.G2
        
        # Calculate H_prime = rho(target_graph)
        # We removed the try/except block because performance validation was removed
        H_prime = apply_permutation(target_graph, rho)
            
        return np.array_equal(H_prime, H)

    def run_protocol(self, prover, rounds: int = 20) -> bool:
        print(f"--- Start of the protocol ({rounds} rounds) ---")
        for i in range(rounds):
            H = prover.commit()
            c = random.randint(0, 1)
            rho = prover.respond(c)
            
            if not self.check(H, rho, c):
                print(f"Failure at round {i+1} (Challenge c={c})")
                return False
                
        print("Success: The proof is accepted.")
        return True

# --- Simulation ---

if __name__ == "__main__":
    n = 1000
    p = 0.3
    G1_nx = nx.erdos_renyi_graph(n, p)
    G1_matrix = nx.to_numpy_array(G1_nx)

    phi_secret = random_permutation(n)
    G2_matrix = apply_permutation(G1_matrix, phi_secret)

    verifier = StrictVerifier(G1_matrix, G2_matrix)

    print("Simulation with Cheater: ")
    cheater = Cheater(G1_matrix, G2_matrix)
    verifier.run_protocol(cheater, rounds=30)

    print("\nSimulation with Honest Prover: ")
    prover = HonestProver(G1_matrix, G2_matrix, phi_secret)
    verifier.run_protocol(prover, rounds=30)
