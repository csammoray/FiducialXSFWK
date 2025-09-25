#!/usr/bin/env python3
"""
Script to read and display profiling data from test_profile.prof
"""

import pstats
import sys
import os

def read_profile(prof_file):
    """Read and display profiling statistics from a .prof file"""
    
    if not os.path.exists(prof_file):
        print(f"Error: Profile file '{prof_file}' not found!")
        return
    
    print(f"Reading profiling data from: {prof_file}")
    print("=" * 80)
    
    # Load the profile statistics
    stats = pstats.Stats(prof_file)
    
    # Print different views of the data
    print("\n1. TOP 30 FUNCTIONS BY CUMULATIVE TIME:")
    print("-" * 50)
    stats.sort_stats('cumulative').print_stats(30)
    
    print("\n2. TOP 20 FUNCTIONS BY TOTAL TIME:")
    print("-" * 50)
    stats.sort_stats('tottime').print_stats(20)
    
    print("\n3. TOP 20 MOST CALLED FUNCTIONS:")
    print("-" * 50)
    stats.sort_stats('ncalls').print_stats(20)


if __name__ == "__main__":
    # Check if a specific profile file was provided as argument
    prof_file = "test_profile.prof"
    
    read_profile(prof_file)
