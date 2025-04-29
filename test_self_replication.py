import asyncio
import time
from quantum_blockchain.core.self_replication import SelfReplicationEngine, ContentGenerator

async def test():
    # Create content generator
    content_generator = ContentGenerator()
    print("Created content generator")
    
    # Generate a sample image from a test hash
    test_hash = "0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef"
    image = content_generator.generate_image(test_hash)
    print(f"Generated image: {image['image_id']}, style: {image['style']}, size: {image['width']}x{image['height']}")
    
    # Generate a sample audio from a test hash
    audio = content_generator.generate_audio(test_hash)
    print(f"Generated audio: {audio['audio_id']}, genre: {audio['genre']}, duration: {audio['duration']}s")
    
    # Create self-replication engine
    replication_engine = SelfReplicationEngine()
    print("Created self-replication engine")
    
    # Process some test data with limited replication depth
    test_data = {
        "content": "Test data for GenesisChain self-replication",
        "timestamp": time.time(),
        "tags": ["test", "self-replication", "genesis"]
    }
    
    print("\nProcessing test data (max level 1)...")
    result = await replication_engine.process_data_input(test_data, max_level=1)
    print(f"Data processed with ID: {result['data_id']}")
    print(f"Generated {result['assets_generated']} assets")
    print(f"Created {result['replications_created']} replications")
    
    # Process operational data
    operational_data = {
        "mining_time": 1.23,
        "nonce": 12345,
        "difficulty": 4,
        "success": True
    }
    
    print("\nProcessing operational data...")
    op_result = await replication_engine.process_operational_data("mining", operational_data)
    print(f"Operational data processed with ID: {op_result['data_id']}")
    print(f"Generated {op_result['assets_generated']} assets")
    print(f"Created {op_result['replications_created']} replications")
    
    # Process system metrics
    system_metrics = {
        "memory_usage": {"blockchain": 1024, "transactions": 256},
        "cpu_usage": 0.35,
        "uptime": 3600,
        "blockchain_length": 10
    }
    
    print("\nProcessing system metrics...")
    metrics_result = await replication_engine.process_system_metrics(system_metrics)
    print(f"System metrics processed with ID: {metrics_result['data_id']}")
    print(f"Generated {metrics_result['assets_generated']} assets")
    print(f"Created {metrics_result['replications_created']} replications")
    
    # Get assets
    images = replication_engine.get_assets_by_type("image")
    audio = replication_engine.get_assets_by_type("audio")
    text = replication_engine.get_assets_by_type("text")
    
    print(f"\nTotal assets generated: {len(images)} images, {len(audio)} audio, {len(text)} text")
    
    # Get metrics
    metrics = replication_engine.get_metrics()
    print("\nSelf-replication metrics:")
    print(f"Data inputs processed: {metrics['data_inputs_processed']}")
    print(f"Assets generated: {metrics['assets_generated']}")
    print(f"Replications created: {metrics['replications_created']}")
    print(f"Max chain depth: {metrics['max_chain_depth']}")
    
    # Get the replication chain for the first data input
    root_id = result['data_id']
    chain = replication_engine.get_replication_chain(root_id)
    print(f"\nReplication chain for {root_id}:")
    print(f"Root ID: {chain['root_id']}")
    print(f"Children: {len(chain['children'])}")

if __name__ == "__main__":
    asyncio.run(test())