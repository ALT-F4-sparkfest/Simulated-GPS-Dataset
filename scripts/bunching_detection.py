import pandas as pd
import numpy as np
import math
from datetime import datetime
from typing import List, Dict, Tuple, Optional
from itertools import combinations


# Distance Helper (same Haversine, but returns METERS)

def haversine_distance_meters(lat1: float, lon1: float,
                               lat2: float, lon2: float) -> float:
    """
    Calculate distance between two GPS coordinates IN METERS.

    This is the same Haversine formula used in ETA, but we return
    meters (not km) because bunching thresholds are small (<200m).

    Args:
        lat1, lon1: First vehicle's position
        lat2, lon2: Second vehicle's position

    Returns:
        float: Distance in METERS

    Example:
    >>> dist = haversine_distance_meters(14.5700, 121.1050, 14.5710, 121.1060)
    >>> print(f"{dist:.1f} m")  # Output: ~148.3 m (very close, bunching!)
    """
    R = 6371000  # Earth radius in METERS (not km)

    lat1_r = math.radians(lat1)
    lon1_r = math.radians(lon1)
    lat2_r = math.radians(lat2)
    lon2_r = math.radians(lon2)

    dlat = lat2_r - lat1_r
    dlon = lon2_r - lon1_r

    a = (math.sin(dlat / 2) ** 2 +
         math.cos(lat1_r) * math.cos(lat2_r) * math.sin(dlon / 2) ** 2)
    c = 2 * math.asin(math.sqrt(a))

    return R * c


# Core Bunching Detection Logic

# Thresholds (defined as constants for easy tuning)
BUNCHING_THRESHOLD_METERS = 200    # Alert if closer than this
RESOLUTION_THRESHOLD_METERS = 500  # Clear alert if farther than this
MIN_SPEED_KMH = 5                  # Below this = vehicle is stopped/terminal
GPS_STALE_SECONDS = 60             # Ignore vehicles with GPS older than this

# Terminal stop names (to exclude from detection)
# Backend Lead: update this list with actual terminal names from route definitions
TERMINAL_KEYWORDS = [
    "Araneta Center",
    "Tutuban Center",
    "Makati CBD",
    "Marikina CBD",
    "Pasig Blvd",
    "San Juan CBD",
]


def is_at_terminal(stop_name: str) -> bool:
    """
    Check if a vehicle is currently at a terminal stop.

    Vehicles at terminals are normally clustered together (queuing),
    so we exclude them from bunching detection.

    Args:
        stop_name: Current stop name from GPS reading

    Returns:
        bool: True if at a terminal (should exclude from detection)

    Example:
    >>> is_at_terminal("Araneta Center, Cubao")
    True  # Terminal — exclude
    >>> is_at_terminal("Ortigas Center")
    False  # Mid-route stop — include
    """
    if not stop_name:
        return False
    # Check if any terminal keyword appears in the stop name
    return any(keyword.lower() in stop_name.lower() for keyword in TERMINAL_KEYWORDS)


def is_gps_fresh(timestamp_str: str, current_time: datetime,
                  max_age_seconds: int = GPS_STALE_SECONDS) -> bool:
    """
    Check if a GPS reading is recent enough to trust.

    Stale GPS data can cause false alerts (vehicle might have moved).
    We only use readings from the last 60 seconds.

    Args:
        timestamp_str: ISO 8601 timestamp string from GPS reading
        current_time: Current time for comparison
        max_age_seconds: Maximum acceptable age in seconds

    Returns:
        bool: True if GPS data is fresh (within max_age_seconds)

    Example:
    >>> is_gps_fresh("2026-06-29T08:15:00Z", datetime(2026, 6, 29, 8, 15, 45))
    True  # 45 seconds old — fresh enough
    >>> is_gps_fresh("2026-06-29T08:14:00Z", datetime(2026, 6, 29, 8, 15, 45))
    False  # 105 seconds old — too stale
    """
    try:
        # Parse timestamp (handle both 'Z' and '+00:00' formats)
        ts = pd.to_datetime(timestamp_str, utc=True)
        now = pd.to_datetime(current_time, utc=True)

        age_seconds = (now - ts).total_seconds()
        return 0 <= age_seconds <= max_age_seconds
    except Exception:
        return False  # If we can't parse the timestamp, treat as stale


def detect_bunching_for_snapshot(
    vehicle_snapshots: List[Dict],
    current_time: datetime,
    existing_alerts: Dict = None
) -> List[Dict]:
    """
    Detect bunching events from a snapshot of all vehicle positions.

    This is the main function called every 30 seconds by the Backend.
    It checks all active vehicles on each route for proximity.

    Args:
        vehicle_snapshots: List of latest GPS reading for each vehicle.
            Each dict has: vehicle_id, route_id, latitude, longitude,
                          speed_kmh, stop_name, timestamp
        current_time: Current datetime (for freshness check)
        existing_alerts: Dict of currently active alerts (to avoid duplicates)
            Key: "{vehicle_a}_{vehicle_b}", Value: alert dict

    Returns:
        List[Dict]: New or updated bunching alerts

    Algorithm:
    1. Group vehicles by route
    2. Filter out terminals and stale GPS
    3. Check all pairs on same route
    4. Alert if distance < 200m AND both moving (speed > 5 km/h)
    5. Return list of alerts

    Example:
    >>> snapshots = [
    ...     {'vehicle_id': 'CUBAO-MAKATI-V1', 'route_id': 'CUBAO-MAKATI',
    ...      'latitude': 14.5700, 'longitude': 121.1050,
    ...      'speed_kmh': 18, 'stop_name': 'Ortigas Center',
    ...      'timestamp': '2026-06-29T08:15:00Z'},
    ...     {'vehicle_id': 'CUBAO-MAKATI-V2', 'route_id': 'CUBAO-MAKATI',
    ...      'latitude': 14.5710, 'longitude': 121.1060,
    ...      'speed_kmh': 16, 'stop_name': 'Ortigas Center',
    ...      'timestamp': '2026-06-29T08:15:00Z'},
    ... ]
    >>> alerts = detect_bunching_for_snapshot(snapshots, datetime.now())
    >>> print(len(alerts))  # Output: 1 (one bunching event detected)
    """
    if existing_alerts is None:
        existing_alerts = {}

    new_alerts = []

    # Step 1: Group vehicles by route
    vehicles_by_route = {}
    for vehicle in vehicle_snapshots:
        route_id = vehicle['route_id']
        if route_id not in vehicles_by_route:
            vehicles_by_route[route_id] = []
        vehicles_by_route[route_id].append(vehicle)

    # Step 2: Check each route independently
    for route_id, vehicles in vehicles_by_route.items():

        # Filter: only fresh GPS, not at terminals, and moving
        eligible = []
        for v in vehicles:
            if not is_gps_fresh(v['timestamp'], current_time):
                continue  # Skip stale GPS
            if is_at_terminal(v['stop_name']):
                continue  # Skip terminal queuing
            if v['speed_kmh'] < MIN_SPEED_KMH:
                continue  # Skip stopped vehicles
            eligible.append(v)

        # Need at least 2 vehicles to compare
        if len(eligible) < 2:
            continue

        # Step 3: Check ALL pairs of vehicles on this route
        for vehicle_a, vehicle_b in combinations(eligible, 2):
            # Calculate distance between the two vehicles
            distance_m = haversine_distance_meters(
                vehicle_a['latitude'], vehicle_a['longitude'],
                vehicle_b['latitude'], vehicle_b['longitude']
            )

            # Generate a stable key for this pair (alphabetical order)
            pair_key = "_".join(sorted([vehicle_a['vehicle_id'], vehicle_b['vehicle_id']]))

            # Step 4: Check bunching threshold
            if distance_m < BUNCHING_THRESHOLD_METERS:
                # Determine the nearest stop for the alert message
                # (Use vehicle_a's current stop as reference)
                nearest_stop = vehicle_a['stop_name'] or vehicle_b['stop_name']

                # Build the alert message
                message = (
                    f"{vehicle_a['vehicle_id']} and {vehicle_b['vehicle_id']} "
                    f"are bunched near {nearest_stop}. "
                    f"Distance: {distance_m:.0f}m. "
                    f"Consider holding {vehicle_b['vehicle_id']} at the "
                    f"next stop to restore spacing."
                )

                alert = {
                    'alert_id': f"ALERT-{route_id}-{current_time.strftime('%Y%m%d-%H%M')}",
                    'route_id': route_id,
                    'vehicle_a': vehicle_a['vehicle_id'],
                    'vehicle_b': vehicle_b['vehicle_id'],
                    'distance_meters': round(distance_m, 1),
                    'vehicle_a_lat': vehicle_a['latitude'],
                    'vehicle_a_lon': vehicle_a['longitude'],
                    'vehicle_b_lat': vehicle_b['latitude'],
                    'vehicle_b_lon': vehicle_b['longitude'],
                    'nearest_stop': nearest_stop,
                    'speed_a_kmh': vehicle_a['speed_kmh'],
                    'speed_b_kmh': vehicle_b['speed_kmh'],
                    'status': 'ACTIVE',
                    'detected_at': current_time.isoformat(),
                    'resolved_at': None,
                    'message': message,
                    'pair_key': pair_key,
                }
                new_alerts.append(alert)

            # Step 5: Check if an existing alert should be resolved
            elif pair_key in existing_alerts:
                if distance_m > RESOLUTION_THRESHOLD_METERS:
                    # Vehicles have spread out: resolve the alert
                    resolved = existing_alerts[pair_key].copy()
                    resolved['status'] = 'RESOLVED'
                    resolved['resolved_at'] = current_time.isoformat()
                    new_alerts.append(resolved)

    return new_alerts


# PART 3: Simulation Test Using Our Generated Data

def simulate_bunching_from_csv(csv_path: str) -> None:
    """
    Run bunching detection on our simulated GPS dataset.

    This simulates what the Backend does in real time:
    - Every 30 seconds, grab latest GPS for each vehicle
    - Run detect_bunching_for_snapshot()
    - Print any alerts

    Args:
        csv_path: Path to simulated CSV (simulated_trips_multiroute.csv or
                  simulated_trips.csv)

    Why this test matters:
    - Proves the detection algorithm works on our data
    - Shows what the dashboard will display
    - Validates our route + vehicle structure
    - Generates demo-ready alert output for pitch
    """
    print(f"\nLoading GPS data from: {csv_path}")
    df = pd.read_csv(csv_path)

    # Rename columns to match expected structure
    # (simulated_trips.csv uses 'vehicle_id', 'timestamp', etc.)
    print(f"  Loaded {len(df):,} GPS readings across {df['vehicle_id'].nunique()} vehicles")

    # Check if route_id column exists (multiroute CSV has it, single route doesn't)
    if 'route_id' not in df.columns:
        # Single-route CSV: assign route from filename
        df['route_id'] = 'CUBAO-DIVISORIA'

    # Get all unique timestamps (sorted)
    # We'll "replay" the data by stepping through timestamps
    timestamps = sorted(df['timestamp'].unique())
    print(f"  Date range: {timestamps[0]} to {timestamps[-1]}")
    print(f"  Simulating {len(timestamps)} GPS snapshots...\n")

    # Track alerts
    all_alerts = []
    existing_alerts = {}  # For hysteresis (resolve when distance > 500m)
    alert_count = 0

    # Step through each timestamp (every 30 seconds in our data)
    for ts in timestamps:
        # Get all vehicles' position at this timestamp
        snapshot_df = df[df['timestamp'] == ts].copy()

        # Convert to list of dicts (simulating Firebase query result)
        vehicle_snapshots = snapshot_df.to_dict('records')

        # Parse current time
        current_time = pd.to_datetime(ts, utc=True).to_pydatetime()

        # Run bunching detection
        alerts = detect_bunching_for_snapshot(
            vehicle_snapshots=vehicle_snapshots,
            current_time=current_time,
            existing_alerts=existing_alerts
        )

        # Process new alerts
        for alert in alerts:
            pair_key = alert['pair_key']

            if alert['status'] == 'ACTIVE':
                # Only print if this is a NEW alert (not already active)
                if pair_key not in existing_alerts:
                    alert_count += 1
                    all_alerts.append(alert)

                    print(f"🚨 BUNCHING ALERT #{alert_count}")
                    print(f"   Route:    {alert['route_id']}")
                    print(f"   Vehicles: {alert['vehicle_a']} + {alert['vehicle_b']}")
                    print(f"   Distance: {alert['distance_meters']} meters")
                    print(f"   Location: Near {alert['nearest_stop']}")
                    print(f"   Time:     {alert['detected_at']}")
                    print(f"   Message:  {alert['message']}")
                    print()

                # Update or add to existing alerts
                existing_alerts[pair_key] = alert

            elif alert['status'] == 'RESOLVED':
                # Alert resolved: vehicles spread out
                if pair_key in existing_alerts:
                    del existing_alerts[pair_key]
                    print(f"✅ ALERT RESOLVED: {alert['vehicle_a']} + {alert['vehicle_b']} separated")

    # Summary
    print("=" * 80)
    print(f"SIMULATION COMPLETE")
    print(f"  Total GPS snapshots processed: {len(timestamps):,}")
    print(f"  Total bunching alerts raised:  {alert_count}")

    if alert_count > 0:
        # Breakdown by route
        alert_df = pd.DataFrame(all_alerts)
        print(f"\n  Alerts by route:")
        for route_id, count in alert_df['route_id'].value_counts().items():
            print(f"    {route_id}: {count} alert(s)")

        print(f"\n  Sample alert for pitch demo:")
        sample = all_alerts[0]
        print(f"    {sample['message']}")
    else:
        print(f"\n  Note: No bunching detected in simulated data.")
        print(f"  (This is expected — simulated vehicles start at different times.)")
        print(f"  In real-world data, bunching occurs when jeepneys get caught in")
        print(f"  the same traffic wave and gradually close the gap between them.")

    print("=" * 80)

    return all_alerts


# Standalone Example (no CSV needed)

def run_manual_example():
    """
    Run a manual bunching detection example without CSV.

    This creates two fake vehicle snapshots that ARE bunching,
    and two that are NOT, to validate the detection logic.
    """
    print("\n" + "=" * 80)
    print("MANUAL EXAMPLE: Bunching Detection Logic Test")
    print("=" * 80)

    # Scenario A: Two vehicles bunching (145 meters apart, both moving)
    print("\n[Scenario A] Two vehicles on CUBAO-MAKATI, 145m apart, both moving")
    snapshots_a = [
        {
            'vehicle_id': 'CUBAO-MAKATI-V1',
            'route_id': 'CUBAO-MAKATI',
            'latitude': 14.5700,
            'longitude': 121.1050,
            'speed_kmh': 18.5,
            'stop_name': 'Ortigas Center',
            'timestamp': '2026-06-29T08:15:00Z',
        },
        {
            'vehicle_id': 'CUBAO-MAKATI-V2',
            'route_id': 'CUBAO-MAKATI',
            'latitude': 14.5710,
            'longitude': 121.1060,
            'speed_kmh': 16.2,
            'stop_name': 'Ortigas Center',
            'timestamp': '2026-06-29T08:15:00Z',
        },
    ]
    alerts_a = detect_bunching_for_snapshot(
        snapshots_a,
        datetime(2026, 6, 29, 8, 15)
    )
    print(f"  Distance: {haversine_distance_meters(14.5700, 121.1050, 14.5710, 121.1060):.1f} m")
    print(f"  Alerts raised: {len(alerts_a)}")
    if alerts_a:
        print(f"  Message: {alerts_a[0]['message']}")
    print(f"  Expected: 1 alert ✓" if len(alerts_a) == 1 else f"  Expected: 1 alert ✗")

    # Scenario B: Two vehicles on different routes (should NOT alert)
    print("\n[Scenario B] Vehicles on DIFFERENT routes, close together (cross-route, okay)")
    snapshots_b = [
        {
            'vehicle_id': 'CUBAO-MAKATI-V1',
            'route_id': 'CUBAO-MAKATI',  # Route A
            'latitude': 14.5700,
            'longitude': 121.1050,
            'speed_kmh': 20,
            'stop_name': 'Ortigas Center',
            'timestamp': '2026-06-29T08:15:00Z',
        },
        {
            'vehicle_id': 'CUBAO-PASIG-V1',
            'route_id': 'CUBAO-PASIG',  # Route B (different route!)
            'latitude': 14.5710,
            'longitude': 121.1060,
            'speed_kmh': 18,
            'stop_name': 'Ortigas Avenue',
            'timestamp': '2026-06-29T08:15:00Z',
        },
    ]
    alerts_b = detect_bunching_for_snapshot(
        snapshots_b,
        datetime(2026, 6, 29, 8, 15)
    )
    print(f"  Distance: {haversine_distance_meters(14.5700, 121.1050, 14.5710, 121.1060):.1f} m")
    print(f"  Alerts raised: {len(alerts_b)}")
    print(f"  Expected: 0 alerts ✓" if len(alerts_b) == 0 else f"  Expected: 0 alerts ✗")

    # Scenario C: Two vehicles at terminal (should NOT alert)
    print("\n[Scenario C] Two vehicles at terminal (queuing, NOT bunching)")
    snapshots_c = [
        {
            'vehicle_id': 'CUBAO-MAKATI-V1',
            'route_id': 'CUBAO-MAKATI',
            'latitude': 14.5808,
            'longitude': 121.0885,
            'speed_kmh': 0,
            'stop_name': 'Araneta Center, Cubao',  # Terminal!
            'timestamp': '2026-06-29T08:15:00Z',
        },
        {
            'vehicle_id': 'CUBAO-MAKATI-V2',
            'route_id': 'CUBAO-MAKATI',
            'latitude': 14.5810,
            'longitude': 121.0887,
            'speed_kmh': 0,
            'stop_name': 'Araneta Center, Cubao',  # Also at terminal
            'timestamp': '2026-06-29T08:15:00Z',
        },
    ]
    alerts_c = detect_bunching_for_snapshot(
        snapshots_c,
        datetime(2026, 6, 29, 8, 15)
    )
    print(f"  Distance: {haversine_distance_meters(14.5808, 121.0885, 14.5810, 121.0887):.1f} m")
    print(f"  Both at terminal: Araneta Center, Cubao")
    print(f"  Alerts raised: {len(alerts_c)}")
    print(f"  Expected: 0 alerts ✓" if len(alerts_c) == 0 else f"  Expected: 0 alerts ✗")

    # Scenario D: Two vehicles far apart on same route (no bunching)
    print("\n[Scenario D] Two vehicles on same route, 800m apart (ideal spacing)")
    snapshots_d = [
        {
            'vehicle_id': 'CUBAO-MAKATI-V1',
            'route_id': 'CUBAO-MAKATI',
            'latitude': 14.5700,
            'longitude': 121.1050,
            'speed_kmh': 22,
            'stop_name': 'Ortigas Center',
            'timestamp': '2026-06-29T08:15:00Z',
        },
        {
            'vehicle_id': 'CUBAO-MAKATI-V2',
            'route_id': 'CUBAO-MAKATI',
            'latitude': 14.5600,  # Much farther ahead
            'longitude': 121.1150,
            'speed_kmh': 20,
            'stop_name': 'Robinsons Galleria',
            'timestamp': '2026-06-29T08:15:00Z',
        },
    ]
    alerts_d = detect_bunching_for_snapshot(
        snapshots_d,
        datetime(2026, 6, 29, 8, 15)
    )
    dist_d = haversine_distance_meters(14.5700, 121.1050, 14.5600, 121.1150)
    print(f"  Distance: {dist_d:.1f} m (well-spaced)")
    print(f"  Alerts raised: {len(alerts_d)}")
    print(f"  Expected: 0 alerts ✓" if len(alerts_d) == 0 else f"  Expected: 0 alerts ✗")

    print("\n" + "=" * 80)
    print("All test scenarios completed!")
    print("=" * 80)


# PART 5: Main Entry Point

if __name__ == "__main__":
    print("=" * 80)
    print("SmartRoute: Bunching Detection Reference Implementation")
    print("=" * 80)

    # Run manual examples first (validates logic)
    run_manual_example()

    # Try to run simulation on our CSV data
    import os

    # Check which CSV files exist and run on both
    csv_files = [
        "data/simulated_trips_multiroute.csv",
        "data/simulated_trips.csv",
    ]

    for csv_path in csv_files:
        if os.path.exists(csv_path):
            print(f"\n\nSimulating on: {csv_path}")
            simulate_bunching_from_csv(csv_path)
            break  # Run on first one found
    else:
        print("\nNote: No CSV found in data/ directory.")
        print("Run generate_simulated_gps.py or generate_multiroute_gps.py first.")
        print("Manual examples above are still valid!")
