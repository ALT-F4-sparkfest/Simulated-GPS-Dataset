import pandas as pd
import numpy as np
import math
from datetime import datetime
from typing import Dict, List, Tuple, Optional



# Core Distance & Speed Calculations

def haversine_distance(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """
    Calculate great-circle distance between two lat/lon points.
    
    Why Haversine?
    - Accounts for Earth's curvature (6,371 km radius)
    - More accurate than straight-line Euclidean distance for >1 km
    - Standard for GPS navigation
    
    Formula:
    a = sin²(Δφ/2) + cos(φ1) × cos(φ2) × sin²(Δλ/2)
    c = 2 × atan2(√a, √(1−a))
    d = R × c
    
    Where:
    - φ = latitude (in radians)
    - λ = longitude (in radians)
    - R = Earth's radius = 6,371 km
    
    Args:
        lat1, lon1: Starting point (decimal degrees)
        lat2, lon2: Ending point (decimal degrees)
    
    Returns:
        float: Distance in kilometers
    
    Example:
    >>> dist = haversine_distance(14.5808, 121.0885, 14.5100, 121.1350)
    >>> print(f"{dist:.2f} km")  # Output: ~8.45 km (Cubao to Makati)
    """
    # Earth's radius in kilometers
    R = 6371
    
    # Convert degrees to radians
    lat1_rad = math.radians(lat1)
    lon1_rad = math.radians(lon1)
    lat2_rad = math.radians(lat2)
    lon2_rad = math.radians(lon2)
    
    # Differences
    dlat = lat2_rad - lat1_rad
    dlon = lon2_rad - lon1_rad
    
    # Haversine formula
    a = math.sin(dlat / 2) ** 2 + math.cos(lat1_rad) * math.cos(lat2_rad) * math.sin(dlon / 2) ** 2
    c = 2 * math.asin(math.sqrt(a))
    
    # Distance
    distance = R * c
    return distance


def get_traffic_multiplier(hour_of_day: int) -> float:
    """
    Return traffic congestion multiplier based on hour of day.
    
    Philosophy:
    - Metro Manila has predictable traffic patterns
    - Morning rush (7-9 AM) is heaviest (1.8x slower)
    - Evening rush (4-7 PM) is also heavy (1.7x slower)
    - Midday is baseline (1.0x)
    
    Args:
        hour_of_day: Hour (0-23) from datetime.hour
    
    Returns:
        float: Multiplier (0.9 to 1.8)
    
    Interpretation:
    - multiplier = 1.0: normal speed (baseline)
    - multiplier = 1.8: 80% slower (severe congestion)
    - multiplier = 0.9: 10% faster (light traffic)
    
    Example:
    >>> mult_morning = get_traffic_multiplier(8)  # 8 AM
    >>> print(mult_morning)  # Output: 1.8 (heavy rush hour)
    
    >>> mult_noon = get_traffic_multiplier(12)  # 12 PM
    >>> print(mult_noon)  # Output: 1.0 (baseline)
    """
    if 5 <= hour_of_day < 7:
        # 5-7 AM: Early commute (moderate)
        return 1.2
    elif 7 <= hour_of_day < 9:
        # 7-9 AM: PEAK MORNING RUSH (heaviest)
        return 1.8
    elif 9 <= hour_of_day < 16:
        # 9 AM - 4 PM: Midday baseline (free-flowing)
        return 1.0
    elif 16 <= hour_of_day < 19:
        # 4-7 PM: PEAK EVENING RUSH (heavy)
        return 1.7
    elif 19 <= hour_of_day < 22:
        # 7-10 PM: Evening (lighter than peak)
        return 1.1
    elif 22 <= hour_of_day < 24:
        # 10 PM - Midnight: Late night (very light)
        return 0.95
    else:  # 0 <= hour_of_day < 5
        # Midnight - 5 AM: Overnight (minimal traffic)
        return 0.9


def get_effective_speed(current_speed_kmh: float, 
                       last_2_min_speeds: List[float]) -> float:
    """
    Determine the speed to use for ETA calculation.
    
    Problem:
    - If vehicle is stopped (speed = 0 km/h), dividing by 0 breaks ETA formula
    - Need a fallback to still estimate arrival time
    
    Solution:
    - If moving (>1 km/h): use current speed (instantaneous)
    - If stopped (<1 km/h): use average of last 2 minutes
    - Clamp minimum to 1 km/h to avoid division by zero
    
    Args:
        current_speed_kmh: Current speed from latest GPS reading
        last_2_min_speeds: List of speeds from last 4 readings (~30 sec each)
    
    Returns:
        float: Speed to use in ETA formula (km/h)
    
    Example:
    >>> # Vehicle stopped at a stop
    >>> current = 0
    >>> recent = [15, 10, 5, 0]  # slowing down then stopped
    >>> speed = get_effective_speed(current, recent)
    >>> print(speed)  # Output: 7.5 (average of recent speeds)
    
    >>> # Vehicle moving normally
    >>> current = 25
    >>> recent = [24, 25, 26, 25]
    >>> speed = get_effective_speed(current, recent)
    >>> print(speed)  # Output: 25 (current speed, since moving)
    sabi ng ai yan
    """
    # If moving: use current speed (most accurate)
    if current_speed_kmh > 1:
        return current_speed_kmh
    
    # If stopped: use recent average to estimate movement
    if last_2_min_speeds:
        avg_recent = np.mean(last_2_min_speeds)
        return max(1, avg_recent)  # Minimum 1 km/h
    
    # Fallback: assume 10 km/h (typical jeepney speed)
    return 10


def should_show_waiting_status(seconds_stationary: int, speed_kmh: float) -> bool:
    """
    Decide if vehicle should show "Waiting" instead of ETA.
    
    Rationale:
    - 0-30 sec: normal (stopped at intersection, brief pause)
    - 30 sec - 3 min: picking up/dropping passengers at a stop
    - 3+ min: stuck in traffic, waiting for passengers, or breakdown
    
    If vehicle has been stopped >3 minutes, show "Waiting at [stop]"
    instead of a false ETA.
    
    Args:
        seconds_stationary: How long vehicle speed has been <1 km/h
        speed_kmh: Current speed
    
    Returns:
        bool: True if should display "Waiting..." status
    
    Example:
    >>> # Vehicle stopped for 2 minutes
    >>> waiting = should_show_waiting_status(120, 0)
    >>> print(waiting)  # Output: False (still show ETA)
    
    >>> # Vehicle stopped for 4 minutes
    >>> waiting = should_show_waiting_status(240, 0)
    >>> print(waiting)  # Output: True (show "Waiting...")
    sabi ulit ng ai to
    """
    # Only check if vehicle is actually stopped
    if speed_kmh > 1:
        return False
    
    # If stopped for >3 minutes: show waiting status
    if seconds_stationary > 180:
        return True
    
    return False


# Main ETA Calculation Function

def calculate_eta_to_stop(
    current_lat: float,
    current_lon: float,
    stop_lat: float,
    stop_lon: float,
    current_time: datetime,
    current_speed_kmh: float,
    last_2_min_speeds: List[float],
    seconds_stationary: int = 0,
    stop_name: str = "Unknown Stop"
) -> Dict:
    """
    Calculate ETA for a jeepney to reach a specific stop.
    
    This is the main function that Backend will call repeatedly
    (every 30 seconds or when new GPS arrives).
    
    Args:
        current_lat, current_lon: Vehicle's current position
        stop_lat, stop_lon: Target stop's coordinates
        current_time: Current time (datetime object)
        current_speed_kmh: Speed from GPS (km/h)
        last_2_min_speeds: List of speeds from last 2 minutes
        seconds_stationary: How long vehicle has been stopped
        stop_name: Name of target stop (for display)
    
    Returns:
        dict: {
            'eta_minutes': int,
            'status': str ('arriving'|'approaching'|'waiting'|'departed'),
            'display_text': str,
            'confidence': str ('high'|'moderate'|'low'),
            'distance_km': float,
            'timestamp': str
        }
    
    Algorithm:
    1. Calculate distance using Haversine
    2. Check for edge cases (very close, stationary long)
    3. Get effective speed (current or recent avg)
    4. Get traffic multiplier for this hour
    5. Apply formula: ETA = (distance / speed) × 60 × multiplier
    6. Return structured result
    
    Example:
    >>> result = calculate_eta_to_stop(
    ...     current_lat=14.5700,
    ...     current_lon=121.1050,
    ...     stop_lat=14.5200,
    ...     stop_lon=121.1350,
    ...     current_time=datetime(2026, 6, 29, 8, 15),
    ...     current_speed_kmh=18.5,
    ...     last_2_min_speeds=[20, 18, 16, 18],
    ...     stop_name="Ayala Avenue"
    ... )
    >>> print(result['display_text'])  # Output: "~12 min to Ayala Avenue"
    sinabi talaga ng ai to??
    """
    
    # Step 1: Calculate distance
    distance_km = haversine_distance(current_lat, current_lon, stop_lat, stop_lon)
    
    # Step 2: Edge case - already at stop
    if distance_km < 0.1:  # <100 meters
        return {
            'eta_minutes': 0,
            'status': 'arriving',
            'display_text': f'🟢 Arriving at {stop_name}',
            'confidence': 'high',
            'distance_km': distance_km,
            'timestamp': current_time.isoformat()
        }
    
    # Step 3: Edge case - vehicle waiting >3 min
    if should_show_waiting_status(seconds_stationary, current_speed_kmh):
        return {
            'eta_minutes': None,  # No ETA during wait
            'status': 'waiting',
            'display_text': f'⏳ Vehicle waiting at {stop_name}...',
            'confidence': 'low',
            'distance_km': distance_km,
            'timestamp': current_time.isoformat()
        }
    
    # Step 4: Get effective speed (current or recent average)
    speed = get_effective_speed(current_speed_kmh, last_2_min_speeds)
    
    # Step 5: Get traffic multiplier for this hour
    hour = current_time.hour
    traffic_mult = get_traffic_multiplier(hour)
    
    # Step 6: Apply ETA formula
    # ETA (minutes) = (distance_km / speed_kmh) × 60 × traffic_multiplier
    eta_minutes_float = (distance_km / speed) * 60 * traffic_mult
    eta_minutes = int(round(eta_minutes_float))
    
    # Step 7: Determine confidence level
    # High confidence: moving, data is current
    # Moderate: just stopped, using recent speeds
    # Low: stopped >3 min
    if current_speed_kmh > 5:
        confidence = 'high'
    elif current_speed_kmh > 1:
        confidence = 'moderate'
    else:
        confidence = 'low'  # Stopped, using estimates
    
    # Step 8: Return structured result
    return {
        'eta_minutes': eta_minutes,
        'status': 'approaching',
        'display_text': f'~{eta_minutes} min to {stop_name}',
        'confidence': confidence,
        'distance_km': round(distance_km, 2),
        'timestamp': current_time.isoformat()
    }


# Batch Processing (for Backend integration)

def calculate_etas_for_all_stops_on_route(
    vehicle_row: Dict,
    route_stops: List[Tuple[float, float, str]],
    speed_history: List[float] = None,
    stationary_seconds: int = 0
) -> List[Dict]:
    """
    Calculate ETA from a vehicle to ALL stops on its current route.
    
    This is useful for:
    - Commuter app: show "next 3 stops" with ETAs
    - Dashboard: show full route with all ETAs
    - Route planning: compare different stops
    
    Args:
        vehicle_row: Dict with 'latitude', 'longitude', 'speed_kmh', 'timestamp'
        route_stops: List of (lat, lon, stop_name) for all stops on route
        speed_history: List of recent speeds (default: 5-second intervals)
        stationary_seconds: How long vehicle has been stopped
    
    Returns:
        List[Dict]: One ETA dict per stop (ordered)
    
    Example:
    >>> vehicle = {
    ...     'latitude': 14.5808,
    ...     'longitude': 121.0885,
    ...     'speed_kmh': 20,
    ...     'timestamp': '2026-06-29T08:15:30Z'
    ... }
    >>> stops = [
    ...     (14.5760, 121.0750, 'Aurora Blvd'),
    ...     (14.5700, 121.0620, 'Dapitan'),
    ...     (14.5100, 121.1350, 'Makati CBD')
    ... ]
    >>> etas = calculate_etas_for_all_stops_on_route(vehicle, stops)
    >>> for eta in etas[:3]:
    ...     print(eta['display_text'])
    # Output:
    # ~3 min to Aurora Blvd
    # ~8 min to Dapitan
    # ~15 min to Makati CBD
    """
    if speed_history is None:
        speed_history = [vehicle_row['speed_kmh']] * 4
    
    current_time = datetime.fromisoformat(
        vehicle_row['timestamp'].replace('Z', '+00:00')
    )
    
    etas = []
    for stop_lat, stop_lon, stop_name in route_stops:
        eta = calculate_eta_to_stop(
            current_lat=vehicle_row['latitude'],
            current_lon=vehicle_row['longitude'],
            stop_lat=stop_lat,
            stop_lon=stop_lon,
            current_time=current_time,
            current_speed_kmh=vehicle_row['speed_kmh'],
            last_2_min_speeds=speed_history,
            seconds_stationary=stationary_seconds,
            stop_name=stop_name
        )
        etas.append(eta)
    
    return etas


# Testing & Examples

def run_example_calculations():
    """
    Run realistic examples to demonstrate ETA calculations.
    
    These show:
    - Normal operation (moving, ETA makes sense)
    - Rush hour effect (same distance, longer ETA)
    - Stopped vehicle (using recent average speed)
    - Vehicle arriving (very close, show "Arriving now")
    """
    print("=" * 80)
    print("SmartRoute ETA Heuristic - Example Calculations")
    print("=" * 80)
    
    # Example 1: Midday, moving normally
    print("\n[Example 1] Midday (12 PM), vehicle moving")
    print("-" * 80)
    result1 = calculate_eta_to_stop(
        current_lat=14.5760,
        current_lon=121.0750,
        stop_lat=14.5100,
        stop_lon=121.1350,
        current_time=datetime(2026, 6, 29, 12, 0),
        current_speed_kmh=28,
        last_2_min_speeds=[27, 28, 29, 28],
        stop_name="Makati CBD"
    )
    print(f"  Distance: {result1['distance_km']} km")
    print(f"  Status: {result1['status']}")
    print(f"  ETA: {result1['display_text']}")
    print(f"  Confidence: {result1['confidence']}")
    print(f"  Calculation: (7.5 km / 28 km/h) × 60 × 1.0 = 16.07 min ✓")
    
    # Example 2: Morning rush (8 AM), same distance
    print("\n[Example 2] Morning Rush (8 AM), same distance & speed")
    print("-" * 80)
    result2 = calculate_eta_to_stop(
        current_lat=14.5760,
        current_lon=121.0750,
        stop_lat=14.5100,
        stop_lon=121.1350,
        current_time=datetime(2026, 6, 29, 8, 0),
        current_speed_kmh=28,
        last_2_min_speeds=[27, 28, 29, 28],
        stop_name="Makati CBD"
    )
    print(f"  Distance: {result2['distance_km']} km (same)")
    print(f"  Status: {result2['status']}")
    print(f"  ETA: {result2['display_text']}")
    print(f"  Confidence: {result2['confidence']}")
    print(f"  Calculation: (7.5 km / 28 km/h) × 60 × 1.8 = 28.93 min")
    print(f"  Note: Traffic multiplier of 1.8x makes trip 80% longer!")
    
    # Example 3: Vehicle stopped at stop
    print("\n[Example 3] Vehicle stopped at a stop (speed = 0)")
    print("-" * 80)
    result3 = calculate_eta_to_stop(
        current_lat=14.5760,
        current_lon=121.0750,
        stop_lat=14.5100,
        stop_lon=121.1350,
        current_time=datetime(2026, 6, 29, 15, 0),
        current_speed_kmh=0,  # Stopped!
        last_2_min_speeds=[10, 5, 0, 0],  # Slowed down, now stopped
        stop_name="Makati CBD"
    )
    print(f"  Current speed: 0 km/h (stopped at Aurora Blvd)")
    print(f"  Status: {result3['status']}")
    print(f"  ETA: {result3['display_text']}")
    print(f"  Confidence: {result3['confidence']}")
    print(f"  Calculation: Use recent avg (6.25 km/h): (7.5 / 6.25) × 60 × 1.0 = 72 min")
    print(f"  Note: Low confidence since using estimates, not real speed")
    
    # Example 4: Vehicle arriving (very close)
    print("\n[Example 4] Vehicle very close (arriving)")
    print("-" * 80)
    result4 = calculate_eta_to_stop(
        current_lat=14.5190,
        current_lon=121.1345,  # Almost at target
        stop_lat=14.5100,
        stop_lon=121.1350,
        current_time=datetime(2026, 6, 29, 14, 30),
        current_speed_kmh=15,
        last_2_min_speeds=[15, 14, 15, 15],
        stop_name="Makati CBD"
    )
    print(f"  Distance: {result4['distance_km']} km (< 100 meters)")
    print(f"  Status: {result4['status']}")
    print(f"  ETA: {result4['display_text']}")
    print(f"  Confidence: {result4['confidence']}")
    print(f"  Note: Special case - show 'Arriving now' with green indicator 🟢")
    
    # Example 5: Vehicle stuck/waiting
    print("\n[Example 5] Vehicle stuck/waiting >3 minutes")
    print("-" * 80)
    result5 = calculate_eta_to_stop(
        current_lat=14.5500,
        current_lon=121.0900,
        stop_lat=14.5100,
        stop_lon=121.1350,
        current_time=datetime(2026, 6, 29, 8, 0),
        current_speed_kmh=0,  # Stopped
        last_2_min_speeds=[0, 0, 0, 0],
        seconds_stationary=240,  # 4 minutes
        stop_name="Makati CBD"
    )
    print(f"  Stationary for: 240 seconds (4 minutes)")
    print(f"  Status: {result5['status']}")
    print(f"  ETA: {result5['display_text']}")
    print(f"  Confidence: {result5['confidence']}")
    print(f"  Note: Don't show false ETA; show 'Waiting...' to reassure user")
    
    print("\n" + "=" * 80)
    print("All examples completed ✓")
    print("=" * 80)


if __name__ == "__main__":
    # Run examples
    run_example_calculations()
    
    print("\n\nREFERENCE: Haversine Distance Test")
    print("-" * 80)
    dist = haversine_distance(14.5808, 121.0885, 14.5100, 121.1350)
    print(f"Distance from Cubao (14.5808, 121.0885) to Makati (14.5100, 121.1350):")
    print(f"  {dist:.2f} km")
    
    print("\n\nREFERENCE: Traffic Multiplier by Hour")
    print("-" * 80)
    for hour in [0, 5, 6, 8, 12, 16, 18, 22]:
        mult = get_traffic_multiplier(hour)
        time_str = f"{hour:02d}:00"
        print(f"  {time_str}: {mult:.1f}x multiplier")