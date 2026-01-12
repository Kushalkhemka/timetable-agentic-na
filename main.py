#!/usr/bin/env python3
"""
Multi-Agent College Timetable Scheduling System
Based on PlanGEN framework from Google

Usage:
    python main.py                  # Run full scheduling
    python main.py --test           # Run with subset of data
    python main.py --max-iter 10    # Set max iterations
"""
import argparse
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from models import SchedulingConfig, Day
from scheduler import SchedulingOrchestrator


def main():
    parser = argparse.ArgumentParser(
        description="Multi-Agent College Timetable Scheduling System"
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        default=".",
        help="Directory containing CSV data files"
    )
    parser.add_argument(
        "--max-iter",
        type=int,
        default=5,
        help="Maximum iterations for scheduling"
    )
    parser.add_argument(
        "--rooms",
        type=int,
        default=10,
        help="Number of available rooms"
    )
    parser.add_argument(
        "--room-capacity",
        type=int,
        default=90,
        help="Capacity per room"
    )
    parser.add_argument(
        "--start-hour",
        type=int,
        default=10,
        help="Start hour (24h format)"
    )
    parser.add_argument(
        "--end-hour",
        type=int,
        default=18,
        help="End hour (24h format)"
    )
    parser.add_argument(
        "--test",
        action="store_true",
        help="Run in test mode with reduced iterations"
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Suppress verbose output"
    )
    
    args = parser.parse_args()
    
    # Configure scheduling
    config = SchedulingConfig(
        days=[Day.MONDAY, Day.TUESDAY, Day.WEDNESDAY, Day.THURSDAY, Day.FRIDAY],
        start_hour=args.start_hour,
        end_hour=args.end_hour,
        num_rooms=args.rooms,
        room_capacity=args.room_capacity
    )
    
    max_iterations = 2 if args.test else args.max_iter
    
    # Run orchestrator
    orchestrator = SchedulingOrchestrator(
        data_dir=args.data_dir,
        config=config,
        max_iterations=max_iterations
    )
    
    try:
        proposal = orchestrator.run(verbose=not args.quiet)
        
        if not args.quiet:
            print("\n" + "=" * 60)
            print("🎉 Scheduling Complete!")
            print("=" * 60)
            print(f"\nScheduled {len(proposal.entries)} sessions")
            print(f"Output directory: {orchestrator.output_dir}")
            
    except KeyboardInterrupt:
        print("\n\n⚠️ Scheduling interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Error: {e}")
        raise


if __name__ == "__main__":
    main()
