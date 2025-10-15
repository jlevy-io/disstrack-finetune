#!/usr/bin/env python3
"""
Sync data folder to RunPod instance
Usage: python tools/sync_data_to_runpod.py "ssh root@213.181.122.217 -p 13186 -i ~/.ssh/id_ed25519"
"""

import sys
import re
import subprocess
import os
from pathlib import Path

def parse_ssh_command(ssh_cmd: str) -> dict:
    """
    Parse SSH connection string into components
    
    Example input: "ssh root@213.181.122.217 -p 13186 -i ~/.ssh/id_ed25519"
    Returns: {
        'user': 'root',
        'host': '213.181.122.217',
        'port': '13186',
        'identity_file': '~/.ssh/id_ed25519'
    }
    """
    # Extract user@host
    user_host_match = re.search(r'ssh\s+(\w+)@([\d.]+)', ssh_cmd)
    if not user_host_match:
        raise ValueError(f"Could not parse user@host from: {ssh_cmd}")
    
    user = user_host_match.group(1)
    host = user_host_match.group(2)
    
    # Extract port
    port_match = re.search(r'-p\s+(\d+)', ssh_cmd)
    port = port_match.group(1) if port_match else "22"
    
    # Extract identity file
    identity_match = re.search(r'-i\s+([\S]+)', ssh_cmd)
    identity_file = identity_match.group(1) if identity_match else None
    
    return {
        'user': user,
        'host': host,
        'port': port,
        'identity_file': identity_file
    }

def sync_data(ssh_info: dict, local_data_dir: str = "data", remote_path: str = "/workspace/disstrack-finetune/data"):
    """
    Rsync data folder to RunPod instance
    """
    
    # Expand home directory in identity file path
    identity_file = ssh_info['identity_file']
    if identity_file and identity_file.startswith('~'):
        identity_file = os.path.expanduser(identity_file)
    
    # Check if local data directory exists
    local_path = Path(local_data_dir)
    if not local_path.exists():
        print(f"❌ Local data directory not found: {local_data_dir}")
        sys.exit(1)
    
    # Build rsync command
    rsync_cmd = [
        "rsync",
        "-avz",
        "--progress",
        "-e",
        f"ssh -p {ssh_info['port']}" + (f" -i {identity_file}" if identity_file else ""),
        f"{local_data_dir}/",  # Trailing slash is important!
        f"{ssh_info['user']}@{ssh_info['host']}:{remote_path}/"
    ]
    
    print("=" * 70)
    print("📤 SYNCING DATA TO RUNPOD")
    print("=" * 70)
    print()
    print(f"Source:      {local_path.absolute()}")
    print(f"Destination: {ssh_info['user']}@{ssh_info['host']}:{remote_path}")
    print(f"Port:        {ssh_info['port']}")
    if identity_file:
        print(f"SSH Key:     {identity_file}")
    print()
    print("Command:")
    print(" ".join(rsync_cmd))
    print()
    print("=" * 70)
    print()
    
    # Confirm before syncing
    response = input("Proceed with sync? (yes/no): ").strip().lower()
    if response != "yes":
        print("❌ Sync cancelled")
        sys.exit(0)
    
    print()
    print("🚀 Starting sync...")
    print()
    
    # Execute rsync
    try:
        result = subprocess.run(rsync_cmd, check=True)
        print()
        print("=" * 70)
        print("✅ SYNC COMPLETE!")
        print("=" * 70)
        print()
        print("Next steps:")
        print(f"  1. SSH into RunPod: ssh {ssh_info['user']}@{ssh_info['host']} -p {ssh_info['port']}" + (f" -i {identity_file}" if identity_file else ""))
        print("  2. cd /workspace/disstrack-finetune")
        print("  3. bash scripts/finetune_roastme_simple.sh")
        print()
        return 0
    except subprocess.CalledProcessError as e:
        print()
        print("=" * 70)
        print("❌ SYNC FAILED")
        print("=" * 70)
        print()
        print(f"Error: {e}")
        print()
        return 1
    except KeyboardInterrupt:
        print()
        print("⚠️  Sync interrupted by user")
        return 1

def main():
    if len(sys.argv) != 2:
        print("Usage: python tools/sync_data_to_runpod.py \"<ssh-connection-string>\"")
        print()
        print("Example:")
        print("  python tools/sync_data_to_runpod.py \"ssh root@213.181.122.217 -p 13186 -i ~/.ssh/id_ed25519\"")
        print()
        sys.exit(1)
    
    ssh_cmd = sys.argv[1]
    
    try:
        ssh_info = parse_ssh_command(ssh_cmd)
        exit_code = sync_data(ssh_info)
        sys.exit(exit_code)
    except ValueError as e:
        print(f"❌ Error parsing SSH command: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
