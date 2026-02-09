#!/usr/bin/env python3

"""
Verify blockchain addresses for TRX, Solana, ETH.

Usage: python verify_address.py <chain> <address>

Chains: trx, sol, eth
"""

import sys
import hashlib
import base58

def sha256(data):
    return hashlib.sha256(data).digest()

def double_sha256(data):
    return sha256(sha256(data))

def verify_trx_address(address):
    """Verify TRX address (base58 with checksum)."""
    try:
        decoded = base58.b58decode(address)
        if len(decoded) != 25:
            return False
        data = decoded[:-4]
        checksum = decoded[-4:]
        expected = double_sha256(data)[:4]
        return checksum == expected
    except:
        return False

def verify_solana_address(address):
    """Verify Solana address (base58, 32-44 bytes)."""
    try:
        decoded = base58.b58decode(address)
        return 32 <= len(decoded) <= 44
    except:
        return False

def verify_eth_address(address):
    """Verify ETH address (hex with checksum)."""
    if not address.startswith('0x') or len(address) != 42:
        return False
    try:
        int(address, 16)
        return True
    except:
        return False

def main():
    if len(sys.argv) != 3:
        print("Usage: python verify_address.py <chain> <address>")
        print("Chains: trx, sol, eth")
        sys.exit(1)

    chain = sys.argv[1].lower()
    address = sys.argv[2]

    if chain == 'trx':
        valid = verify_trx_address(address)
    elif chain == 'sol':
        valid = verify_solana_address(address)
    elif chain == 'eth':
        valid = verify_eth_address(address)
    else:
        print(f"Unknown chain: {chain}")
        sys.exit(1)

    if valid:
        print(f"Valid {chain.upper()} address: {address}")
    else:
        print(f"Invalid {chain.upper()} address: {address}")

if __name__ == "__main__":
    main()