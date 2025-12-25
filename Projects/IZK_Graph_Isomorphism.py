import numpy as np
import networkx as nx
import random
import time
import multiprocessing
import queue 
from networkx.algorithms import isomorphism

# --- Mathematical Helpers ---

def random_permutation(n: int) -> np.ndarray:
    sigma = np.arange(n)
    np.random.shuffle(sigma)
    return sigma

def apply_permutation(G_matrix: np.ndarray, sigma: np.ndarray) -> np.ndarray:
    """
    H = P * G * P^T
    Optimized using np.ix_ for faster mesh indexing.
    """
    return G_matrix[np.ix_(sigma, sigma)]

def invert_permutation(sigma: np.ndarray) -> np.ndarray:
    """Computes sigma^{-1} in O(N)."""
    n = len(sigma)
    inv = np.empty(n, dtype=int)
    inv[sigma] = np.arange(n)
    return inv

# --- Actors ---

class HonestProver:
    def __init__(self, G1: np.ndarray, G2: np.ndarray, phi: np.ndarray = None):
        if phi is None:
            raise ValueError("HonestProver requires the secret 'phi'!")
        self.G1 = G1
        self.G2 = G2
        self.phi = phi 
        self.n = len(G1)
        self.sigma = None 

    def commit(self) -> np.ndarray:
        self.sigma = random_permutation(self.n)
        return apply_permutation(self.G1, self.sigma)

    def respond(self, challenge: int) -> np.ndarray:
        if challenge == 0:
            return self.sigma
        else:
            phi_inv = invert_permutation(self.phi)
            return phi_inv[self.sigma]

class LazySmartCheater:
    def __init__(self, G1: np.ndarray, G2: np.ndarray, phi: np.ndarray = None):
        self.G1 = G1
        self.G2 = G2
        self.n = len(G1)
        self.sigma = None
        self.predicted_challenge = None
        self.psi = None 

    def _solve_isomorphism(self):
        # Fail fast for simulation purposes
        if self.n > 500:
            return None
            
        GM = isomorphism.GraphMatcher(
            nx.from_numpy_array(self.G1), 
            nx.from_numpy_array(self.G2)
        )
        if GM.is_isomorphic():
            mapping = GM.mapping
            return np.array([mapping[i] for i in range(self.n)])
        return None

    def commit(self) -> np.ndarray:
        self.sigma = random_permutation(self.n)
        self.predicted_challenge = random.randint(0, 1)
        target = self.G1 if self.predicted_challenge == 0 else self.G2
        return apply_permutation(target, self.sigma)

    def respond(self, challenge: int) -> np.ndarray:
        # 1. Lucky Guess
        if challenge == self.predicted_challenge:
            return self.sigma
            
        # 2. Wrong Guess -> Panic Solve
        if self.psi is None:
            self.psi = self._solve_isomorphism()
            
        if self.psi is not None:
            if self.predicted_challenge == 0 and challenge == 1:
                return invert_permutation(self.psi)[self.sigma]
            if self.predicted_challenge == 1 and challenge == 0:
                return self.psi[self.sigma]
        
        # 3. Failed to solve -> Return garbage
        return self.sigma

class SmartCheater:
    def __init__(self, G1: np.ndarray, G2: np.ndarray, phi: np.ndarray = None):
        self.G1 = G1
        self.G2 = G2
        self.n = len(G1)
        self.sigma = None
        self.psi = self._solve_isomorphism_init() 
        
    def _solve_isomorphism_init(self):
        if self.n > 500:
            time.sleep(1.0) 
            return None

        GM = isomorphism.GraphMatcher(
            nx.from_numpy_array(self.G1), 
            nx.from_numpy_array(self.G2)
        )
        if GM.is_isomorphic():
            mapping = GM.mapping
            return np.array([mapping[i] for i in range(self.n)])
        return None

    def commit(self) -> np.ndarray:
        self.sigma = random_permutation(self.n)
        if self.psi is not None:
            return apply_permutation(self.G1, self.sigma)
        
        self.predicted_challenge = random.randint(0, 1)
        target = self.G1 if self.predicted_challenge == 0 else self.G2
        return apply_permutation(target, self.sigma)

    def respond(self, challenge: int) -> np.ndarray:
        if self.psi is not None:
            if challenge == 0:
                return self.sigma
            else:
                return invert_permutation(self.psi)[self.sigma]
        else:
            return self.sigma

# --- Verification Worker ---

def verification_worker(G1, G2, ProverClass, phi, rounds, result_queue):
    try:
        prover = ProverClass(G1, G2, phi=phi)
        result_queue.put("INIT_DONE") 
    except Exception as e:
        result_queue.put(f"ERROR: {e}")
        return

    for i in range(rounds):
        try:
            H = prover.commit()
            c = random.randint(0, 1)
            
            rho = prover.respond(c)
            
            target = G1 if c == 0 else G2
            H_check = target[np.ix_(rho, rho)]
            
            if not np.array_equal(H_check, H):
                # UPDATE: Send the challenge 'c' back with the error
                result_queue.put(f"WRONG_ANSWER:{c}")
                return
            
            result_queue.put("ROUND_PASSED")
                
        except Exception as e:
            result_queue.put(f"ERROR: {e}")
            return

    result_queue.put("SUCCESS")

# --- Strict Verifier ---

class TotalVerifier:
    def __init__(self, G1: np.ndarray, G2: np.ndarray):
        self.G1 = G1
        self.G2 = G2
        self.n = len(G1)
        
        scale_factor = (self.n / 1000.0) ** 2
        self.init_limit = 0.5 + (2.0 * scale_factor)
        self.round_limit = 0.2 + (0.5 * scale_factor)
        
        print(f"[System] Limits (N={self.n}): Init={self.init_limit:.2f}s, Round={self.round_limit:.2f}s")

    def verify_prover_class(self, ProverClass, phi_secret: np.ndarray = None, rounds: int = 20):
        print(f"\n--- Verifying {ProverClass.__name__} ---")
        
        q = multiprocessing.Queue()
        p = multiprocessing.Process(
            target=verification_worker, 
            args=(self.G1, self.G2, ProverClass, phi_secret, rounds, q)
        )
        
        print(f"Checking Initialization...")
        t0 = time.time()
        p.start()
        
        try:
            msg = q.get(timeout=self.init_limit)
            if msg != "INIT_DONE":
                print(f"[FAILED] Initialization Error: {msg}")
                p.terminate(); p.join()
                return False
            print(f"[PASSED] Init okay ({time.time() - t0:.4f}s).")
        except queue.Empty:
            print(f"[FAILED] Initialization Timeout!")
            p.terminate(); p.join()
            return False

        print(f"Running Protocol ({rounds} rounds)...")
        for i in range(rounds):
            try:
                result = q.get(timeout=self.round_limit)
                
                if result == "ROUND_PASSED":
                    continue
                
                # UPDATE: Parse the challenge failure message
                elif result.startswith("WRONG_ANSWER"):
                    parts = result.split(":")
                    failed_c = parts[1] if len(parts) > 1 else "?"
                    print(f"[FAILED] Wrong Answer at Round {i+1} (Challenge: {failed_c}).")
                    p.terminate(); p.join()
                    return False
                    
                else:
                    print(f"[FAILED] Error: {result}")
                    p.terminate(); p.join()
                    return False
            except queue.Empty:
                print(f"[FAILED] Timeout at Round {i+1}!")
                p.terminate(); p.join()
                return False

        print("[SUCCESS] The Proof is Accepted.")
        p.join()
        return True

if __name__ == "__main__":
    n = 10000
    p = 0.5
    
    print(f"Generating Graph (N={n})...")
    G1_nx = nx.erdos_renyi_graph(n, p)
    G1_matrix = nx.to_numpy_array(G1_nx, dtype=np.int8)
    phi_secret = random_permutation(n)
    G2_matrix = apply_permutation(G1_matrix, phi_secret)
    
    verifier = TotalVerifier(G1_matrix, G2_matrix)

    verifier.verify_prover_class(LazySmartCheater, phi_secret=None)
    verifier.verify_prover_class(SmartCheater, phi_secret=None)
    verifier.verify_prover_class(HonestProver, phi_secret=phi_secret)
