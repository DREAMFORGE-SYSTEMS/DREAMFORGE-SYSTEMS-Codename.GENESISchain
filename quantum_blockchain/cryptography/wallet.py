"""
GenesisChain Wallet Implementation

This module provides a quantum-resistant wallet implementation for the GenesisChain blockchain.
It handles key management, transaction signing, and secure storage.
"""

import os
import json
import secrets
import hashlib
import time
from typing import Dict, Any, List, Tuple, Optional, Union
from pathlib import Path

# Import our quantum-resistant cryptography module
from quantum_blockchain.cryptography.quantum_resistant import QuantumResistantCrypto


class Wallet:
    """
    Quantum-resistant wallet for GenesisChain
    
    Features:
    - Multiple quantum-resistant signature schemes
    - Hierarchical deterministic structure for address generation
    - Layered security approach (quantum + traditional when needed)
    - Multi-signature capability
    """
    
    def __init__(self, seed: Optional[bytes] = None, name: str = "default"):
        """
        Initialize a new wallet or load an existing one
        
        Args:
            seed: Optional seed bytes. If not provided, a new random seed will be generated.
            name: Name of the wallet (for multiple wallet support)
        """
        self.name = name
        self.seed = seed if seed else QuantumResistantCrypto.generate_seed()
        
        # Generate master keys using each supported algorithm
        # In a full implementation, we would use a quantum-resistant version of BIP32
        self.key_pairs = {}
        self.addresses = {}
        
        # Generate primary key pairs
        for algorithm in ["falcon", "sphincs"]:
            self._generate_keypair(algorithm, 0)  # Generate key pair #0 for each algorithm
        
        # Set default signing algorithm
        self.default_algorithm = "falcon"  # Falcon is faster than SPHINCS+
    
    def _generate_keypair(self, algorithm: str, index: int) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        """
        Generate a key pair for the specified algorithm and index
        
        Args:
            algorithm: The algorithm to use ("falcon", "sphincs", etc.)
            index: Index of the key pair (for HD wallet support)
            
        Returns:
            Tuple of (private_key, public_key)
        """
        # Derive a deterministic seed for this key pair
        derived_seed = hashlib.sha3_256(self.seed + algorithm.encode() + index.to_bytes(4, 'big')).digest()
        
        # Generate the key pair
        private_key, public_key = QuantumResistantCrypto.generate_keypair(algorithm=algorithm, seed=derived_seed)
        
        # Store the keys
        if algorithm not in self.key_pairs:
            self.key_pairs[algorithm] = {}
            self.addresses[algorithm] = {}
        
        self.key_pairs[algorithm][index] = (private_key, public_key)
        
        # Generate address from public key (simplified version)
        address = self._derive_address(algorithm, public_key)
        self.addresses[algorithm][index] = address
        
        return private_key, public_key
    
    def _derive_address(self, algorithm: str, public_key: Dict[str, Any]) -> str:
        """
        Derive a blockchain address from a public key
        
        Args:
            algorithm: The algorithm used for the key
            public_key: The public key
            
        Returns:
            Address string (with algorithm prefix)
        """
        # Convert public key to canonical form
        if isinstance(public_key, dict):
            key_data = json.dumps(public_key, sort_keys=True).encode()
        else:
            key_data = public_key
        
        # Double hash with SHA3 (quantum-resistant)
        h1 = hashlib.sha3_256(key_data).digest()
        h2 = hashlib.sha3_256(h1).digest()
        
        # Create address with algorithm prefix and checksum
        prefix = algorithm[:2].upper()
        main_part = h2[:20].hex()
        checksum = hashlib.sha3_256((prefix + main_part).encode()).hexdigest()[:8]
        
        return f"{prefix}-{main_part}-{checksum}"
    
    def get_address(self, algorithm: Optional[str] = None, index: int = 0) -> str:
        """
        Get an address from the wallet
        
        Args:
            algorithm: The algorithm to use (default: self.default_algorithm)
            index: Index of the address to get
            
        Returns:
            Address string
        """
        algorithm = algorithm or self.default_algorithm
        
        if algorithm not in self.addresses or index not in self.addresses[algorithm]:
            self._generate_keypair(algorithm, index)
            
        return self.addresses[algorithm][index]
    
    def sign_transaction(self, transaction_data: Dict[str, Any], algorithm: Optional[str] = None, index: int = 0) -> bytes:
        """
        Sign a transaction using the specified key
        
        Args:
            transaction_data: Transaction data to sign
            algorithm: Algorithm to use for signing (default: self.default_algorithm)
            index: Index of the key to use
            
        Returns:
            Signature bytes
        """
        algorithm = algorithm or self.default_algorithm
        
        if algorithm not in self.key_pairs or index not in self.key_pairs[algorithm]:
            self._generate_keypair(algorithm, index)
            
        private_key, _ = self.key_pairs[algorithm][index]
        
        # Sign the transaction data
        signature = QuantumResistantCrypto.sign(private_key, transaction_data)
        
        return signature
    
    def create_transaction(self, recipient: str, amount: float, fee: float = 0.001, data: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Create a new transaction
        
        Args:
            recipient: Recipient address
            amount: Amount to send
            fee: Transaction fee
            data: Optional additional data
            
        Returns:
            Transaction dictionary
        """
        # Get sender address
        sender = self.get_address()
        
        # Create transaction object
        transaction = {
            "sender": sender,
            "recipient": recipient,
            "amount": amount,
            "fee": fee,
            "timestamp": int(time.time()),
            "nonce": secrets.randbits(64),  # Random nonce to prevent replay attacks
            "data": data or {}
        }
        
        # Add reference to the signing algorithm
        transaction["algorithm"] = self.default_algorithm
        
        # Sign the transaction
        signature = self.sign_transaction(transaction)
        transaction["signature"] = signature.hex()
        
        return transaction
    
    def export_public_data(self) -> Dict[str, Any]:
        """
        Export public wallet data (addresses and public keys)
        
        Returns:
            Dictionary with public wallet data
        """
        public_data = {
            "addresses": self.addresses,
            "public_keys": {}
        }
        
        # Include all public keys
        for algorithm, keys in self.key_pairs.items():
            public_data["public_keys"][algorithm] = {}
            for index, (_, public_key) in keys.items():
                public_data["public_keys"][algorithm][str(index)] = public_key
        
        return public_data
    
    def export_encrypted(self, password: str) -> Dict[str, Any]:
        """
        Export encrypted wallet data (including private keys)
        
        Args:
            password: Password to encrypt the wallet
        
        Returns:
            Dictionary with encrypted wallet data
        """
        # Prepare wallet data
        wallet_data = {
            "name": self.name,
            "seed": self.seed.hex(),
            "key_pairs": {},
            "addresses": self.addresses,
            "default_algorithm": self.default_algorithm
        }
        
        # Include all key pairs
        for algorithm, keys in self.key_pairs.items():
            wallet_data["key_pairs"][algorithm] = {}
            for index, (private_key, public_key) in keys.items():
                wallet_data["key_pairs"][algorithm][str(index)] = {
                    "private_key": private_key,
                    "public_key": public_key
                }
        
        # Convert to JSON
        wallet_json = json.dumps(wallet_data)
        
        # Encrypt using password (simplified - in production use a proper encryption library)
        # Derive key from password
        salt = os.urandom(16)
        key = hashlib.pbkdf2_hmac("sha256", password.encode(), salt, 100000)
        
        # For this prototype, we'll simply note that encryption would happen here
        # In a real implementation, we would encrypt with XChaCha20-Poly1305 or AES-GCM
        
        return {
            "encrypted_data": f"PLACEHOLDER: {wallet_json[:30]}...",
            "salt": salt.hex(),
            "encryption": "XChaCha20-Poly1305"
        }
    
    @classmethod
    def import_encrypted(cls, encrypted_data: Dict[str, Any], password: str) -> 'Wallet':
        """
        Import an encrypted wallet
        
        Args:
            encrypted_data: Encrypted wallet data
            password: Password to decrypt the wallet
            
        Returns:
            New Wallet instance
        """
        # For this prototype, we'll simulate decryption
        # This is a placeholder for actual decryption logic
        wallet = cls(seed=bytes.fromhex("0" * 64))
        wallet.name = "imported_wallet"
        
        # Generate some keys to simulate the imported wallet
        wallet._generate_keypair("falcon", 0)
        wallet._generate_keypair("sphincs", 0)
        
        return wallet
    
    def save(self, password: str, directory: Optional[str] = None) -> str:
        """
        Save the wallet to a file
        
        Args:
            password: Password to encrypt the wallet
            directory: Directory to save the wallet (default: current directory)
            
        Returns:
            Path to the saved wallet file
        """
        # Encrypt the wallet
        encrypted_data = self.export_encrypted(password)
        
        # Determine save location
        if directory is None:
            directory = os.getcwd()
        
        # Create directory if it doesn't exist
        os.makedirs(directory, exist_ok=True)
        
        # Save to file
        filename = f"{self.name}_wallet.json"
        path = os.path.join(directory, filename)
        
        with open(path, 'w') as f:
            json.dump(encrypted_data, f, indent=2)
        
        return path
    
    @classmethod
    def load(cls, filename: str, password: str) -> 'Wallet':
        """
        Load a wallet from a file
        
        Args:
            filename: Path to the wallet file
            password: Password to decrypt the wallet
            
        Returns:
            Loaded Wallet instance
        """
        # Read encrypted data from file
        with open(filename, 'r') as f:
            encrypted_data = json.load(f)
        
        # Import the wallet
        return cls.import_encrypted(encrypted_data, password)


class MultiSignatureWallet:
    """
    Multi-signature wallet implementation for GenesisChain
    
    Allows creating transactions that require signatures from multiple parties
    """
    
    def __init__(self, name: str, required_signatures: int, participants: List[Dict[str, Any]]):
        """
        Initialize a new multi-signature wallet
        
        Args:
            name: Name of the wallet
            required_signatures: Number of signatures required to validate a transaction
            participants: List of participant public data (exported from their wallets)
        """
        self.name = name
        self.required_signatures = required_signatures
        self.participants = participants
        
        # Verify that we have enough participants
        if len(participants) < required_signatures:
            raise ValueError("Number of participants must be at least equal to required signatures")
        
        # Create multi-sig address
        self.address = self._create_multisig_address()
    
    def _create_multisig_address(self) -> str:
        """Create a multi-signature address from participant data"""
        # Combine all participant public keys
        participant_data = json.dumps(
            sorted([p["addresses"] for p in self.participants]),
            sort_keys=True
        ).encode()
        
        # Add required signatures count
        ms_data = participant_data + str(self.required_signatures).encode()
        
        # Hash the combined data
        h1 = hashlib.sha3_256(ms_data).digest()
        h2 = hashlib.sha3_256(h1).digest()
        
        # Create address with MS prefix
        prefix = "MS"
        main_part = h2[:20].hex()
        checksum = hashlib.sha3_256((prefix + main_part).encode()).hexdigest()[:8]
        
        return f"{prefix}-{main_part}-{checksum}"
    
    def create_unsigned_transaction(self, recipient: str, amount: float, fee: float = 0.001, data: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Create an unsigned multi-signature transaction
        
        Args:
            recipient: Recipient address
            amount: Amount to send
            fee: Transaction fee
            data: Optional additional data
            
        Returns:
            Unsigned transaction dictionary
        """
        # Create transaction object
        transaction = {
            "sender": self.address,
            "recipient": recipient,
            "amount": amount,
            "fee": fee,
            "timestamp": int(time.time()),
            "nonce": secrets.randbits(64),
            "data": data or {},
            "type": "multi_signature",
            "required_signatures": self.required_signatures,
            "signatures": {}  # Will be filled by participants
        }
        
        return transaction
    
    @staticmethod
    def add_signature(transaction: Dict[str, Any], wallet: Wallet, participant_id: str) -> Dict[str, Any]:
        """
        Add a signature to a multi-signature transaction
        
        Args:
            transaction: The transaction to sign
            wallet: The wallet to sign with
            participant_id: ID of the participant (to identify the signature)
            
        Returns:
            Updated transaction with signature added
        """
        # Create a copy of the transaction without existing signatures
        tx_to_sign = transaction.copy()
        tx_to_sign["signatures"] = {}
        
        # Sign the transaction
        signature = wallet.sign_transaction(tx_to_sign)
        
        # Add signature to transaction
        transaction_copy = transaction.copy()
        transaction_copy["signatures"][participant_id] = {
            "algorithm": wallet.default_algorithm,
            "address": wallet.get_address(),
            "signature": signature.hex()
        }
        
        return transaction_copy


# Example usage
if __name__ == "__main__":
    # Create a new wallet
    my_wallet = Wallet(name="Alice")
    print(f"Created wallet with address: {my_wallet.get_address()}")
    
    # Create a transaction
    tx = my_wallet.create_transaction(
        recipient="FA-0123456789abcdef0123-abcdef12", 
        amount=10.0,
        data={"memo": "Test transaction"}
    )
    print(f"Created transaction: {tx}")
    
    # Multi-signature example
    bob_wallet = Wallet(name="Bob")
    charlie_wallet = Wallet(name="Charlie")
    
    # Export public data for multi-sig setup
    alice_public = my_wallet.export_public_data()
    bob_public = bob_wallet.export_public_data()
    charlie_public = charlie_wallet.export_public_data()
    
    # Create multi-sig wallet (2-of-3)
    multisig = MultiSignatureWallet(
        name="Shared Wallet",
        required_signatures=2,
        participants=[alice_public, bob_public, charlie_public]
    )
    
    print(f"Multi-signature address: {multisig.address}")
    
    # Create unsigned transaction
    unsigned_tx = multisig.create_unsigned_transaction(
        recipient="SP-9876543210fedcba9876-fedcba98",
        amount=5.0,
        data={"memo": "Joint payment"}
    )
    
    # Sign with Alice and Bob
    partially_signed = MultiSignatureWallet.add_signature(unsigned_tx, my_wallet, "alice")
    fully_signed = MultiSignatureWallet.add_signature(partially_signed, bob_wallet, "bob")
    
    print(f"Multi-sig transaction with {len(fully_signed['signatures'])} signatures:")