from fastapi import FastAPI, HTTPException, Body
from fastapi.responses import JSONResponse
from dotenv import load_dotenv
from starlette.middleware.cors import CORSMiddleware
from motor.motor_asyncio import AsyncIOMotorClient
import uvicorn
import os
import logging
from pathlib import Path
import hashlib
import json
import time
import uuid
from typing import List, Dict, Any, Optional
from pydantic import BaseModel, Field

# /backend 
ROOT_DIR = Path(__file__).parent
load_dotenv(ROOT_DIR / '.env')

# MongoDB connection
mongo_url = os.environ.get('MONGO_URL')
db_name = os.environ.get('DB_NAME', 'genesischain')
client = AsyncIOMotorClient(mongo_url)
db = client[db_name]

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_credentials=True,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Models
class Transaction(BaseModel):
    sender: str
    recipient: str
    amount: float
    timestamp: float = Field(default_factory=time.time)
    transaction_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    
class Block(BaseModel):
    index: int
    timestamp: float
    transactions: List[Transaction]
    proof: int
    previous_hash: str
    block_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    
class Blockchain(BaseModel):
    chain: List[Block] = []
    current_transactions: List[Transaction] = []
    
    class Config:
        arbitrary_types_allowed = True

# Initialize blockchain
blockchain = Blockchain(chain=[], current_transactions=[])

# Create genesis block
async def create_genesis_block():
    # Check if blockchain already exists in database
    existing_blocks = await db.blocks.count_documents({})
    
    if existing_blocks == 0:
        # Create genesis block if blockchain doesn't exist
        genesis_block = Block(
            index=1,
            timestamp=time.time(),
            transactions=[],
            proof=100,
            previous_hash="1"
        )
        
        # Insert genesis block to database
        await db.blocks.insert_one(genesis_block.dict())
        
        logger.info("Genesis block created")
    else:
        logger.info("Blockchain already exists, skipping genesis block creation")

# Blockchain methods
async def get_last_block():
    last_block = await db.blocks.find_one(
        sort=[("index", -1)]
    )
    return last_block

# QNodeOS-inspired Quantum Network Processing Unit (QNPU) simulator
class QNPUSimulator:
    """Simulates the behavior of a Quantum Network Processing Unit as described in QNodeOS"""
    
    def __init__(self):
        self.qubits_available = 4  # Simulated quantum memory
        self.qubit_states = [None] * self.qubits_available
        self.operations_log = []
        self.operation_counter = 0
        
    def execute_quantum_operation(self, operation_type, params=None):
        """Simulates execution of a quantum operation"""
        self.operation_counter += 1
        
        operation_record = {
            "id": self.operation_counter,
            "type": operation_type,
            "params": params,
            "timestamp": time.time()
        }
        
        self.operations_log.append(operation_record)
        
        # In a real quantum processor, this would perform the actual quantum operation
        # Here we just log it for simulation purposes
        return operation_record
    
    def get_operations_log(self, limit=10):
        """Return recent operations for operational data collection"""
        return self.operations_log[-limit:] if self.operations_log else []
    
    def clear_operations_log(self):
        """Clear operations log"""
        self.operations_log = []

# Enhanced Quantum-inspired AI Oracle for mining optimization
class MiningOracle:
    def __init__(self, qnpu_simulator=None):
        """Initialize the mining oracle with a QNodeOS-inspired architecture"""
        self.training_data = []
        self.model = self._create_default_model()
        self.metrics = {
            "total_attempts": 0,
            "oracle_approvals": 0,
            "successful_mines": 0,
            "energy_saved": 0,
            "quantum_operations": 0
        }
        self.qnpu = qnpu_simulator if qnpu_simulator else QNPUSimulator()
        self.operational_data_buffer = []
        
    def _create_default_model(self):
        """Create a quantum-inspired decision tree model for initial predictions"""
        # In a real implementation, this would be a trained model
        # For this simulation, we use a rule-based system inspired by QBT's Method C
        return {
            "min_one_bits": 8,      # Minimum number of 1 bits
            "max_one_bits": 24,     # Maximum number of 1 bits
            "min_consec_zeros": 3,  # Minimum consecutive zeros
            "min_consec_ones": 2,   # Minimum consecutive ones
            "xor_threshold": 16,    # XOR threshold with last proof
            "hamming_weight_threshold": 12,  # New QNodeOS-inspired parameter
            "quantum_pattern_match": 0.65    # New QNodeOS-inspired parameter
        }
    
    def extract_features(self, last_proof: int, proof: int):
        """
        Extract predictive features from the proof parameters using 
        simulated quantum operations inspired by QNodeOS
        """
        # Simulate quantum feature extraction by logging quantum operations
        self.qnpu.execute_quantum_operation(
            "feature_extraction", 
            {"last_proof": last_proof, "proof": proof}
        )
        
        binary = bin(proof)[2:].zfill(32)
        features = {
            "one_bits": binary.count('1'),
            "consec_zeros": self._max_consecutive_chars(binary, '0'),
            "consec_ones": self._max_consecutive_chars(binary, '1'),
            "xor_with_last": bin(last_proof ^ proof).count('1'),
            "last_proof_relation": abs(last_proof - proof) % 256,
            # New quantum-inspired features
            "hamming_weight": self._calculate_hamming_weight(last_proof, proof),
            "quantum_pattern": self._simulate_quantum_pattern_match(last_proof, proof)
        }
        
        self.metrics["quantum_operations"] += 1
        return features
    
    def _max_consecutive_chars(self, s: str, char: str) -> int:
        """Count maximum consecutive occurrences of a character in a string"""
        max_count = 0
        current_count = 0
        
        for c in s:
            if c == char:
                current_count += 1
                max_count = max(max_count, current_count)
            else:
                current_count = 0
                
        return max_count
    
    def _calculate_hamming_weight(self, last_proof: int, proof: int) -> int:
        """Calculate Hamming weight between two proofs (simulates quantum calculation)"""
        # Simulate quantum operation
        self.qnpu.execute_quantum_operation(
            "hamming_calculation", 
            {"last_proof": last_proof, "proof": proof}
        )
        
        # Classical calculation to simulate the result
        xor_result = last_proof ^ proof
        return bin(xor_result).count('1')
    
    def _simulate_quantum_pattern_match(self, last_proof: int, proof: int) -> float:
        """Simulate a quantum pattern matching algorithm"""
        # Simulate quantum operation
        self.qnpu.execute_quantum_operation(
            "pattern_matching", 
            {"last_proof": last_proof, "proof": proof}
        )
        
        # Use hash of the inputs to generate a pseudo-random but deterministic result
        combined = str(last_proof) + str(proof)
        hash_val = int(hashlib.md5(combined.encode()).hexdigest(), 16)
        return (hash_val % 1000) / 1000.0  # Range 0.0-1.0
    
    def predict(self, last_proof: int, proof: int) -> bool:
        """
        Predict if this mining attempt is likely to succeed using 
        QNodeOS-inspired quantum simulation
        """
        # Simulate a quantum prediction operation
        self.qnpu.execute_quantum_operation(
            "oracle_prediction", 
            {"last_proof": last_proof, "proof": proof}
        )
        
        # Extract features
        features = self.extract_features(last_proof, proof)
        
        # Enhanced rule-based prediction (inspired by QBT's Method C)
        model = self.model
        if (
            model["min_one_bits"] <= features["one_bits"] <= model["max_one_bits"] and
            features["consec_zeros"] >= model["min_consec_zeros"] and
            features["consec_ones"] >= model["min_consec_ones"] and
            features["xor_with_last"] <= model["xor_threshold"] and
            # New quantum-inspired conditions
            features["hamming_weight"] <= model["hamming_weight_threshold"] and
            features["quantum_pattern"] >= model["quantum_pattern_match"]
        ):
            self.metrics["oracle_approvals"] += 1
            
            # Collect operational data for self-replication
            self._collect_operational_data("oracle_approval", features)
            return True
        
        # Save energy by skipping unlikely candidates
        self.metrics["energy_saved"] += 1
        return False
    
    def update_metrics(self, successful: bool):
        """Update mining metrics"""
        self.metrics["total_attempts"] += 1
        if successful:
            self.metrics["successful_mines"] += 1
            
            # Collect operational data for successful mining
            self._collect_operational_data("successful_mining", {"success": True})
    
    def _collect_operational_data(self, event_type, data):
        """Collect operational data for self-replication"""
        event = {
            "type": event_type,
            "timestamp": time.time(),
            "data": data,
            "quantum_operations": self.qnpu.get_operations_log(5)  # Get last 5 operations
        }
        
        self.operational_data_buffer.append(event)
        
        # When buffer reaches threshold, process for self-replication
        if len(self.operational_data_buffer) >= 5:  # Small threshold for simulation
            self._process_operational_data()
    
    def _process_operational_data(self):
        """Process collected operational data for self-replication"""
        if not self.operational_data_buffer:
            return
            
        # Convert operational data to a hash for content generation
        data_string = json.dumps(self.operational_data_buffer, sort_keys=True)
        operational_hash = hashlib.sha256(data_string.encode()).hexdigest()
        
        # Clear buffer after processing
        self.operational_data_buffer = []
        
        # Return the hash for self-replication
        return operational_hash
    
    def get_efficiency_stats(self):
        """Return efficiency statistics"""
        metrics = self.metrics
        
        if metrics["oracle_approvals"] > 0:
            success_rate = metrics["successful_mines"] / metrics["oracle_approvals"]
        else:
            success_rate = 0
            
        energy_efficiency = metrics["energy_saved"] / (metrics["total_attempts"] or 1)
        
        return {
            "success_rate": success_rate,
            "energy_efficiency": energy_efficiency,
            "total_attempts": metrics["total_attempts"],
            "oracle_approvals": metrics["oracle_approvals"],
            "successful_mines": metrics["successful_mines"],
            "energy_saved": metrics["energy_saved"],
            "quantum_operations": metrics["quantum_operations"]
        }
        
    def get_operational_hash(self):
        """Get a hash from current operational data for immediate use"""
        return self._process_operational_data()

# Initialize the mining oracle
mining_oracle = MiningOracle()

async def quantum_enhanced_proof_of_work(last_proof: int) -> int:
    """
    Quantum-inspired Proof of Work Algorithm using AI Oracle:
    - Uses an AI Oracle to pre-screen potential solutions
    - Only performs SHA-256 hashing on promising candidates
    - Tracks energy savings and efficiency metrics
    """
    proof = 0
    while True:
        # Pre-screening with AI Oracle
        if mining_oracle.predict(last_proof, proof):
            # Only perform expensive SHA-256 if Oracle approves
            if valid_proof(last_proof, proof):
                mining_oracle.update_metrics(True)
                
                # Log efficiency metrics
                stats = mining_oracle.get_efficiency_stats()
                logger.info(f"Mining efficiency: Success rate={stats['success_rate']:.2%}, Energy saved={stats['energy_efficiency']:.2%}")
                
                return proof
            else:
                mining_oracle.update_metrics(False)
        
        proof += 1

async def proof_of_work(last_proof: int) -> int:
    """
    Wrapper for the quantum-enhanced proof of work algorithm
    """
    return await quantum_enhanced_proof_of_work(last_proof)

def valid_proof(last_proof: int, proof: int) -> bool:
    """
    Validates the Proof: Does hash(last_proof, proof) contain 4 leading zeroes?
    """
    guess = f'{last_proof}{proof}'.encode()
    guess_hash = hashlib.sha256(guess).hexdigest()
    return guess_hash[:4] == "0000"

# Routes
@app.get("/api")
async def root():
    return {"message": "GenesisChain API"}

@app.get("/api/chain")
async def get_chain():
    blocks = await db.blocks.find().sort("index", 1).to_list(length=None)
    # Convert ObjectId to string
    for block in blocks:
        block["_id"] = str(block["_id"])
    return {"chain": blocks, "length": len(blocks)}

@app.post("/api/transactions/new")
async def new_transaction(transaction: Transaction = Body(...)):
    # Add a new transaction to the list of transactions
    await db.transactions.insert_one(transaction.dict())
    
    # Also add to current transactions (for next block)
    blockchain.current_transactions.append(transaction)
    
    return JSONResponse(
        status_code=201,
        content={"message": f"Transaction will be added to Block {(await get_last_block())['index'] + 1}"}
    )

@app.get("/api/mine")
async def mine():
    # Get the last block
    last_block = await get_last_block()
    
    # Convert MongoDB ObjectId to string to make it JSON serializable
    if '_id' in last_block:
        last_block['_id'] = str(last_block['_id'])
    
    # Calculate the proof of work
    last_proof = last_block['proof']
    proof = await proof_of_work(last_proof)
    
    # Create a new transaction to award the miner
    miner_transaction = Transaction(
        sender="0",  # "0" signifies that this node has mined a new coin
        recipient="miner-address",  # This would be the miner's address in a real implementation
        amount=1.0
    )
    
    # Store miner transaction in database
    await db.transactions.insert_one(miner_transaction.dict())
    
    # Add to current transactions
    blockchain.current_transactions.append(miner_transaction)
    
    # Collect current transactions
    current_transactions = blockchain.current_transactions.copy()
    blockchain.current_transactions = []
    
    # Create a new Block - first create a serializable version of last_block
    # by removing ObjectId
    clean_last_block = {k: v for k, v in last_block.items() if k != '_id'}
    previous_hash = hashlib.sha256(json.dumps(clean_last_block, sort_keys=True).encode()).hexdigest()
    
    block = Block(
        index=last_block['index'] + 1,
        timestamp=time.time(),
        transactions=current_transactions,
        proof=proof,
        previous_hash=previous_hash,
    )
    
    # Reset current transactions
    blockchain.current_transactions = []
    
    # Save the new block to database
    block_dict = block.dict()
    await db.blocks.insert_one(block_dict)
    
    # Also save all transactions as confirmed
    for transaction in current_transactions:
        await db.transactions.update_one(
            {"transaction_id": transaction.transaction_id},
            {"$set": {"confirmed": True, "block_id": block.block_id}}
        )
    
    return {
        "message": "New Block Forged",
        "index": block.index,
        "transactions": [tx.dict() for tx in current_transactions],
        "proof": block.proof,
        "previous_hash": block.previous_hash,
    }

@app.get("/api/transactions")
async def get_transactions():
    transactions = await db.transactions.find().to_list(length=None)
    # Convert ObjectId to string
    for tx in transactions:
        tx["_id"] = str(tx["_id"])
    return {"transactions": transactions}

# Functions for simulating AI content generation
async def generate_image(prompt_hash: str) -> Dict[str, Any]:
    """
    Simulates AI image generation based on a hash.
    In a full implementation, this would call an actual AI API.
    """
    # Use the hash to create deterministic but unique "AI-generated" content
    seed = int(prompt_hash[:8], 16)  # Convert first 8 chars of hash to integer
    
    # Simulate different image properties based on the hash
    width = 400 + (seed % 400)  # 400-800px width
    height = 300 + (seed % 500)  # 300-800px height
    style = ["abstract", "landscape", "portrait", "futuristic", "digital", "geometric"][seed % 6]
    
    return {
        "image_id": f"img_{prompt_hash[:10]}",
        "prompt_hash": prompt_hash,
        "width": width,
        "height": height,
        "style": style,
        "url": f"https://example.com/ai-images/{prompt_hash[:10]}.jpg",
        "created_at": time.time()
    }

async def generate_audio(prompt_hash: str) -> Dict[str, Any]:
    """
    Simulates AI audio generation based on a hash.
    In a full implementation, this would call an actual AI API.
    """
    # Use the hash to create deterministic but unique "AI-generated" content
    seed = int(prompt_hash[:8], 16)  # Convert first 8 chars of hash to integer
    
    # Simulate different audio properties based on the hash
    duration = 30 + (seed % 120)  # 30-150 seconds
    genre = ["ambient", "electronic", "cinematic", "jazz", "rock", "classical"][seed % 6]
    
    return {
        "audio_id": f"audio_{prompt_hash[:10]}",
        "prompt_hash": prompt_hash,
        "duration": duration,
        "genre": genre,
        "url": f"https://example.com/ai-audio/{prompt_hash[:10]}.mp3",
        "created_at": time.time()
    }

# Self-replication mechanism
@app.post("/api/data-input")
async def process_data_input(data: Dict[str, Any] = Body(...)):
    """
    This endpoint receives external data, hashes it, generates AI content,
    and implements the self-replication mechanism.
    """
    # Generate a hash of the input data
    data_string = json.dumps(data, sort_keys=True)
    data_hash = hashlib.sha256(data_string.encode()).hexdigest()
    
    # Generate AI content based on the hash (simulated)
    images = [await generate_image(data_hash + str(i)) for i in range(4)]
    audio_tracks = [await generate_audio(data_hash + str(i)) for i in range(2)]
    
    # Store the data, hash, and generated content
    data_record = {
        "original_data": data,
        "hash": data_hash,
        "timestamp": time.time(),
        "processed": True,
        "data_id": str(uuid.uuid4()),
        "source": "user_input",
        "generated_content": {
            "images": images,
            "audio": audio_tracks
        }
    }
    
    await db.data_inputs.insert_one(data_record)
    
    # Self-replication: Use the generated content as new input data
    # This simulates how the system can feed its own outputs back as inputs
    replications = []
    
    # 1. Content-based replications (from generated images)
    for img in images:
        replication_data = {
            "source_type": "image",
            "source_id": img["image_id"],
            "content": f"AI-generated image with style {img['style']}",
            "timestamp": time.time()
        }
        
        # Create a new hash from the AI-generated content
        repl_data_string = json.dumps(replication_data, sort_keys=True)
        repl_hash = hashlib.sha256(repl_data_string.encode()).hexdigest()
        
        repl_record = {
            "original_data": replication_data,
            "hash": repl_hash,
            "timestamp": time.time(),
            "processed": False,
            "data_id": str(uuid.uuid4()),
            "parent_data_id": data_record["data_id"],
            "source": "content_replication"
        }
        
        await db.data_inputs.insert_one(repl_record)
        replications.append(repl_record)
    
    # 2. Operational data replication (from system activity)
    # Get operational hash from mining oracle (if available)
    operational_hash = mining_oracle.get_operational_hash()
    if operational_hash:
        # Generate AI content based on the operational hash
        op_images = [await generate_image(operational_hash + str(i)) for i in range(2)]
        op_audio = [await generate_audio(operational_hash + str(i)) for i in range(1)]
        
        op_record = {
            "original_data": {
                "source_type": "operational_data",
                "content": "System operational data from mining and quantum operations",
                "timestamp": time.time()
            },
            "hash": operational_hash,
            "timestamp": time.time(),
            "processed": True,
            "data_id": str(uuid.uuid4()),
            "parent_data_id": data_record["data_id"],
            "source": "operational_replication",
            "generated_content": {
                "images": op_images,
                "audio": op_audio
            }
        }
        
        await db.data_inputs.insert_one(op_record)
        replications.append(op_record)
    
    # Collect system logs as another source of operational data
    system_data = {
        "timestamp": time.time(),
        "memory_usage": os.popen('free -m').readlines(),
        "disk_usage": os.popen('df -h').readlines(),
        "system_load": os.getloadavg(),
        "process_id": os.getpid()
    }
    
    # Hash system operational data
    sys_data_string = json.dumps(system_data, sort_keys=True)
    sys_hash = hashlib.sha256(sys_data_string.encode()).hexdigest()
    
    # Create system operational data record
    sys_record = {
        "original_data": {
            "source_type": "system_metrics",
            "content": "System performance metrics and logs",
            "timestamp": time.time()
        },
        "hash": sys_hash,
        "timestamp": time.time(),
        "processed": False,
        "data_id": str(uuid.uuid4()),
        "parent_data_id": data_record["data_id"],
        "source": "system_replication"
    }
    
    await db.data_inputs.insert_one(sys_record)
    replications.append(sys_record)
    
    return {
        "message": "Data processed successfully",
        "hash": data_hash,
        "data_id": data_record["data_id"],
        "generated_content": {
            "images": len(images),
            "audio": len(audio_tracks)
        },
        "replications": len(replications)
    }

@app.get("/api/data-inputs")
async def get_data_inputs():
    """
    Returns all data inputs with their generated content and self-replications.
    """
    data_inputs = await db.data_inputs.find().sort("timestamp", -1).to_list(length=100)
    
    # Convert ObjectId to string
    for data in data_inputs:
        data["_id"] = str(data["_id"])
    
    return {"data_inputs": data_inputs}

@app.on_event("startup")
async def startup_event():
    logger.info("Starting up GenesisChain API")
    await create_genesis_block()

@app.on_event("shutdown")
async def shutdown_db_client():
    client.close()

if __name__ == "__main__":
    uvicorn.run("server:app", host="0.0.0.0", port=8001, reload=True)
