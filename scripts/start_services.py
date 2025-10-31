#!/usr/bin/env python3
"""
Restart Lattice Services Script
Helps restart the web server and check lattice service status
"""
import subprocess
import sys
import time
import requests
import os

def check_port(port, service_name):
    """Check if a service is running on a port"""
    try:
        result = subprocess.run(
            f"netstat -ano | findstr :{port}",
            shell=True, capture_output=True, text=True
        )
        if result.returncode == 0 and result.stdout.strip():
            print(f"✅ {service_name} is running on port {port}")
            return True
        else:
            print(f"❌ {service_name} is NOT running on port {port}")
            return False
    except Exception as e:
        print(f"❌ Error checking {service_name}: {e}")
        return False

def check_lattice_health():
    """Check if lattice service is healthy"""
    try:
        response = requests.get("http://127.0.0.1:11434/health", timeout=5)
        if response.status_code == 200:
            data = response.json()
            print(f"✅ Lattice service is healthy: {data.get('status', 'unknown')}")
            return True
        else:
            print(f"❌ Lattice health check failed: HTTP {response.status_code}")
            return False
    except Exception as e:
        print(f"❌ Lattice health check error: {e}")
        return False

def start_web_server():
    """Start the web server"""
    try:
        print("🌐 Starting web server on port 8080...")
        # Kill any existing web server process
        subprocess.run("taskkill /F /IM python.exe /FI \"WINDOWTITLE eq*web_server*\"", 
                      shell=True, capture_output=True)
        
        # Start new web server
        subprocess.Popen([sys.executable, "web_server.py"], 
                        creationflags=subprocess.CREATE_NEW_CONSOLE)
        time.sleep(2)
        
        if check_port(8080, "Web Server"):
            print("✅ Web server started successfully!")
            return True
        else:
            print("❌ Web server failed to start")
            return False
    except Exception as e:
        print(f"❌ Error starting web server: {e}")
        return False

def main():
    print("🚀 Lattice Services Restart Tool")
    print("=" * 50)
    
    # Change to lattice directory
    try:
        os.chdir(os.path.dirname(os.path.abspath(__file__)))
        print(f"📁 Working directory: {os.getcwd()}")
    except Exception as e:
        print(f"❌ Error changing directory: {e}")
        return
    
    # Check current status
    print("\n🔍 Checking current service status...")
    lattice_running = check_port(11434, "Lattice Service")
    web_running = check_port(8080, "Web Server")
    text_gen_running = check_port(5000, "Text-Generation-WebUI API")
    
    if lattice_running:
        lattice_healthy = check_lattice_health()
    else:
        lattice_healthy = False
    
    print("\n📊 Service Status Summary:")
    print(f"   Text-Generation-WebUI API: {'✅' if text_gen_running else '❌'}")
    print(f"   Lattice Service:           {'✅' if lattice_healthy else '❌'}")
    print(f"   Web Server:                {'✅' if web_running else '❌'}")
    
    # Recommendations
    print("\n💡 Recommendations:")
    
    if not text_gen_running:
        print("   ⚠️  Start text-generation-webui first!")
        print("   📝 Make sure it's running with OpenAI API enabled")
        
    if not lattice_running or not lattice_healthy:
        print("   ⚠️  Start lattice service: python lattice_service.py")
        
    if not web_running:
        print("   🔄 Restarting web server...")
        if start_web_server():
            print("   ✅ Web server restarted successfully!")
        else:
            print("   ❌ Web server restart failed")
    
    # Final status
    print("\n🌐 Web Interface URLs:")
    print("   Simple Interface:  http://localhost:8080")
    print("   Full Interface:    http://localhost:8080/full")
    print("   Lattice Health:    http://localhost:11434/health")
    print("   Memory Stats:      http://localhost:11434/v1/memories/stats")
    
    print("\n🎯 Next Steps:")
    print("   1. Open http://localhost:8080 in your browser")
    print("   2. Try sending a message to test the system")
    print("   3. Check the browser console (F12) for any errors")

if __name__ == "__main__":
    main() 