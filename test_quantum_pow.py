import asyncio
import hashlib
import time
from quantum_blockchain.consensus.quantum_proof_of_work import QuantumProofOfWork, QuantumMiningOracle, QNPUSimulator

async def test():
    # Create a QNPU simulator
    qnpu = QNPUSimulator(qubits_available=8)
    print(f"Created QNPU simulator with {qnpu.qubits_available} qubits")
    
    # Create a mining oracle
    oracle = QuantumMiningOracle(qnpu=qnpu)
    print("Created quantum mining oracle")
    
    # Create the quantum proof of work
    qpow = QuantumProofOfWork(difficulty=3, oracle=oracle)
    print(f"Created quantum proof of work with difficulty {qpow.difficulty}")
    
    # Simple hash function for testing
    def calculate_hash(header, nonce):
        header_copy = header.copy()
        header_copy["nonce"] = nonce
        header_str = str(header_copy)
        return hashlib.sha256(header_str.encode()).hexdigest()
    
    # Example block header
    block_header = {
        "index": 1,
        "previous_hash": "0" * 64,
        "timestamp": int(time.time()),
        "merkle_root": "sample_merkle_root"
    }
    
    # Time standard mining
    standard_start = time.time()
    standard_nonce = 0
    while True:
        block_hash = calculate_hash(block_header, standard_nonce)
        if block_hash.startswith("0" * qpow.difficulty):
            break
        standard_nonce += 1
        if standard_nonce > 100000:  # Limit for testing
            break
    standard_end = time.time()
    standard_time = standard_end - standard_start
    
    # If standard mining was successful, use the same block header for quantum mining
    # Otherwise, use a simpler one for testing
    if standard_nonce > 100000:
        print("Standard mining took too long, using simpler block header for quantum mining")
        block_header = {
            "index": 1, 
            "previous_hash": "0",
            "timestamp": 12345
        }
    
    # Time quantum-enhanced mining
    quantum_start = time.time()
    quantum_nonce, quantum_hash = qpow.find_nonce(block_header, calculate_hash, max_iterations=100000)
    quantum_end = time.time()
    quantum_time = quantum_end - quantum_start
    
    # Print results
    print("\n=== Mining Results ===")
    if standard_nonce <= 100000:
        print(f"Standard mining: nonce={standard_nonce}, time={standard_time:.2f}s")
        print(f"Standard hash: {calculate_hash(block_header, standard_nonce)}")
    else:
        print("Standard mining: Did not complete within iteration limit")
    
    if quantum_nonce > 0:
        print(f"Quantum mining: nonce={quantum_nonce}, time={quantum_time:.2f}s")
        print(f"Quantum hash: {quantum_hash}")
        
        # Get mining efficiency stats
        stats = oracle.get_efficiency_stats()
        print("\n=== Mining Efficiency ===")
        print(f"Total attempts: {stats['total_attempts']}")
        print(f"Oracle approvals: {stats['oracle_approvals']} ({stats['oracle_approvals']/stats['total_attempts']:.2%})")
        print(f"Successful mines: {stats['successful_mines']}")
        print(f"Energy saved: {stats['energy_efficiency']:.2%}")
        print(f"Quantum operations: {stats['quantum_operations']}")
    else:
        print("Quantum mining: Did not complete within iteration limit")
    
    # Verify the proof of work
    if quantum_nonce > 0:
        is_valid = qpow.verify_pow(block_header, quantum_nonce, calculate_hash)
        print(f"\nProof of work valid: {is_valid}")

if __name__ == "__main__":
    asyncio.run(test())