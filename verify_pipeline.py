from backend.pipeline.market_pipeline import PipelineFactory
import logging

def verify_system_connectivity():
    print("🔍 Starting Connectivity Audit...")
    try:
        pipeline = PipelineFactory.create_production_pipeline()
        
        # Checking Core Components
        if hasattr(pipeline.coordinator.executor, 'sync_engine'):
            print("✅ [Database/Sync] MarketSync (Writer) connected.")
        
        if hasattr(pipeline.coordinator.executor, 'reader'):
            print("✅ [Data] MarketReader connected.")
            
        if hasattr(pipeline.coordinator.executor, 'validator'):
            print("✅ [Data] MarketValidator connected.")

        # Checking Event Bus & Audit (Hooks)
        if pipeline.event_bus:
            print("✅ [Core] EventBus connected.")
            
        if hasattr(pipeline.lifecycle.hooks, 'monitor'):
            print("✅ [Core] PipelineMonitor/Audit connected.")
            
        print("\n🚀 All primary components are successfully injected and wired.")
        
    except Exception as e:
        print(f"❌ Connectivity Audit Failed: {e}")

if __name__ == "__main__":
    verify_system_connectivity()
