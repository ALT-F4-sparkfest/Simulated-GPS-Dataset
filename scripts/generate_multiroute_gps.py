import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import math

# Define 4 NEW Routes with Real Cubao-Area Waypoints

ROUTES = {
    "CUBAO-MAKATI": {
        "display_name": "Cubao → Makati (via EDSA)",
        "waypoints": [
            (14.5808, 121.0885, "Araneta Center, Cubao", True),
            (14.5790, 121.0950, "EDSA Cubao", False),
            (14.5700, 121.1050, "Ortigas Center", True),
            (14.5650, 121.1120, "Megamall/SM Megamall", False),
            (14.5600, 121.1150, "Robinsons Galleria", False),
            (14.5500, 121.1200, "Makati Blvd Junction", False),
            (14.5400, 121.1250, "Salcedo Village", False),
            (14.5300, 121.1300, "Greenbelt", False),
            (14.5200, 121.1350, "Ayala Avenue", True),
            (14.5100, 121.1350, "Makati CBD", True),
        ]
    },
    
    "CUBAO-MARIKINA": {
        "display_name": "Cubao → Marikina (via Aurora Blvd)",
        "waypoints": [
            (14.5808, 121.0885, "Araneta Center, Cubao", True),
            (14.5760, 121.0750, "Aurora Boulevard", True),
            (14.5750, 121.0650, "Kapasigan", False),
            (14.5700, 121.0550, "Katipunan Ave", False),
            (14.5650, 121.0450, "Bayanihan Ave", False),
            (14.5600, 121.0350, "Marikina Pasig Bridge", False),
            (14.5550, 121.0300, "Marikina City Hall", True),
            (14.5500, 121.0250, "Riverbanks Center", False),
            (14.5450, 121.0200, "Marikina CBD", True),
        ]
    },
    
    "CUBAO-PASIG": {
        "display_name": "Cubao → Pasig (via Santolan)",
        "waypoints": [
            (14.5808, 121.0885, "Araneta Center, Cubao", True),
            (14.5850, 121.0900, "Santolan Road", False),
            (14.5900, 121.0920, "Kalentong", False),
            (14.5950, 121.0950, "Pasig City Hall", True),
            (14.6000, 121.0980, "Pasig CBD", True),
            (14.6050, 121.1000, "Ortigas Avenue", False),
            (14.6100, 121.1020, "Pasig Blvd", False),
        ]
    },
    
    "CUBAO-SANJUAN": {
        "display_name": "Cubao → San Juan (via Ilalim Rd)",
        "waypoints": [
            (14.5808, 121.0885, "Araneta Center, Cubao", True),
            (14.5820, 121.0800, "Ilalim Road", False),
            (14.5850, 121.0700, "San Juan Main St", True),
            (14.5880, 121.0650, "San Juan Market", False),
            (14.5900, 121.0600, "San Juan CBD", True),
        ]
    },
}

# Helper Functions

def haversine_distance(lat1, lon1, lat2, lon2):
    """
    Calculate distance between two coordinates using Haversine formula.
    
    Purpose:
    - Compute actual great-circle distance (more accurate than straight line)
    - Used to determine trip duration based on route length
    - Critical for realistic ETA calculations
    
    Formula:
    - a = sin²(Δφ/2) + cos(φ1) × cos(φ2) × sin²(Δλ/2)
    - c = 2 × atan2(√a, √(1−a))
    - d = R × c
    
    Where:
    - φ = latitude, λ = longitude (both in radians)
    - R = Earth's radius (6,371 km)
    """
    R = 6371
    dlat = math.radians(lat2 - lat1)
    dlon = math.radians(lon2 - lon1)
    a = (math.sin(dlat / 2) ** 2 +
         math.cos(math.radians(lat1)) * math.cos(math.radians(lat2)) * math.sin(dlon / 2) ** 2)
    c = 2 * math.asin(math.sqrt(a))
    return R * c


def interpolate_points(lat1, lon1, lat2, lon2, num_points):
    """
    Create smooth path between two waypoints.
    
    Why:
    - Real GPS devices don't teleport; they send continuous readings
    - One GPS reading every ~30 seconds is realistic
    - Linear interpolation creates smooth movement on map
    
    Example:
    - Going from Cubao (14.58, 121.09) to Aurora Blvd (14.58, 121.07)
    - If trip takes 5 minutes → need ~10 interpolated points
    - Result: smooth path instead of single line
    """
    lats = np.linspace(lat1, lat2, num_points)
    lons = np.linspace(lon1, lon2, num_points)
    return list(zip(lats, lons))


def get_traffic_multiplier(hour_of_day):
    """
    Return traffic congestion factor by hour of day.
    
    Higher multiplier = slower travel (longer trip duration)
    
    Metro Manila traffic patterns:
    - 5-7 AM: 1.2x (early commute)
    - 7-9 AM: 1.8x (PEAK morning rush)
    - 9 AM-4 PM: 1.0x (normal midday)
    - 4-7 PM: 1.7x (PEAK evening rush)
    - 7-10 PM: 1.1x (evening)
    - 10 PM-5 AM: 0.9x (night, almost empty)
    
    Used for:
    - Adjusting trip duration realistically
    - Making dataset time-aware (reflect actual commuting patterns)
    - ETA heuristic will use this too
    """
    if 5 <= hour_of_day < 7:
        return 1.2
    elif 7 <= hour_of_day < 9:
        return 1.8
    elif 9 <= hour_of_day < 16:
        return 1.0
    elif 16 <= hour_of_day < 19:
        return 1.7
    elif 19 <= hour_of_day < 22:
        return 1.1
    else:
        return 0.9


def get_base_speed_at_stop(stop_index, total_stops, is_major_stop):
    """
    Determine vehicle speed at a specific stop.
    
    Logic:
    - Terminal (start/end): 0 km/h (loading passengers, waiting)
    - Major stop (busy hub): 5 km/h (lots of activity)
    - Regular stop: 10 km/h (brief stop)
    - Open road: 30+ km/h (full cruise speed)
    
    This gets divided by traffic_multiplier later:
    - In rush hour: 30 km/h → 30/1.8 = 17 km/h (slowed by congestion)
    - At midnight: 30 km/h → 30/0.9 = 33 km/h (free-flowing)
    
    Parameters:
    - stop_index: Position in route (0 = first, n = last)
    - total_stops: Total waypoints in this route
    - is_major_stop: Boolean (major terminals vs small stops)
    """
    if stop_index == 0 or stop_index == total_stops - 1:
        # Terminal: fully stopped
        return 0
    elif is_major_stop:
        # Major hub: very slow
        return 5
    else:
        # Regular stop: slow
        return 10


def generate_trip(vehicle_id, route_id, route_waypoints, start_time, direction="forward"):
    """
    Generate a single trip (one-way journey) for a vehicle.
    
    One trip = journey from terminal A → terminal B (or vice versa)
    Duration: ~15-25 minutes = ~30-50 GPS readings (one every 30 seconds)
    
    Process:
    1. Get traffic multiplier for current hour
    2. Iterate through consecutive waypoints
    3. For each segment:
       - Calculate distance (Haversine)
       - Adjust speed based on stop type + traffic
       - Interpolate points along segment
       - Create GPS readings (one every 30 seconds)
    4. Return list of GPS data dicts
    
    Parameters:
    - vehicle_id: e.g., "CUBAO-MAKATI-V1"
    - route_id: e.g., "CUBAO-MAKATI"
    - route_waypoints: List of (lat, lng, name, is_major)
    - start_time: When trip starts (datetime object)
    - direction: "forward" or "backward" (return trip)
    """
    trip_data = []
    
    # Determine order: forward or reversed
    if direction == "forward":
        waypoint_sequence = route_waypoints
    else:
        waypoint_sequence = list(reversed(route_waypoints))
    
    current_time = start_time
    traffic_mult = get_traffic_multiplier(start_time.hour)
    
    # Travel between consecutive waypoints
    for i in range(len(waypoint_sequence) - 1):
        lat1, lon1, stop1_name, is_major1 = waypoint_sequence[i]
        lat2, lon2, stop2_name, is_major2 = waypoint_sequence[i + 1]
        
        # Distance from current waypoint to next
        distance_km = haversine_distance(lat1, lon1, lat2, lon2)
        
        # Get base speed at current waypoint (before traffic adjustment)
        base_speed = get_base_speed_at_stop(i, len(waypoint_sequence), is_major1)
        
        # Apply traffic congestion multiplier
        if base_speed > 0:
            adjusted_speed = base_speed / traffic_mult  # Higher traffic = lower speed
        else:
            adjusted_speed = 0  # Still stopped at terminal
        
        # Calculate segment duration
        if adjusted_speed > 0:
            travel_time_minutes = (distance_km / adjusted_speed) * 60
        else:
            travel_time_minutes = 2  # Terminal waiting time
        
        # Generate intermediate GPS points (roughly 2 per minute)
        num_intermediate_points = max(2, int(travel_time_minutes * 2))
        intermediate_coords = interpolate_points(lat1, lon1, lat2, lon2, num_intermediate_points)
        
        # Create GPS reading for each intermediate point
        for j, (lat, lon) in enumerate(intermediate_coords):
            # Speed profile: slow at start/end, faster in middle
            progress = j / len(intermediate_coords)
            
            # Acceleration phase (first 30%): speeding up from stop
            if progress < 0.3:
                speed_factor = progress / 0.3
            # Deceleration phase (last 30%): slowing into next stop
            elif progress > 0.7:
                speed_factor = (1 - progress) / 0.3
            # Cruise phase (middle 40%): full speed
            else:
                speed_factor = 1.0
            
            speed = adjusted_speed * speed_factor
            
            # Add realistic noise (small random variation)
            speed_with_noise = max(0, speed + np.random.normal(0, 1))
            
            trip_data.append({
                "vehicle_id": vehicle_id,
                "route_id": route_id,
                "timestamp": current_time.isoformat() + "Z",
                "latitude": round(lat, 6),
                "longitude": round(lon, 6),
                "speed_kmh": round(speed_with_noise, 2),
                "stop_name": stop1_name if j == 0 else stop2_name,
            })
            
            # Move time forward by 30 seconds (realistic GPS interval)
            current_time += timedelta(seconds=30)
    
    return trip_data


# Generate Complete 7-Day Multi-Route Dataset

def generate_full_dataset(days=7, vehicles_per_route=2):
    """
    Generate a full week of trips across all 4 NEW routes.
    
    Daily schedule per vehicle:
    - 6-9 AM: 2 round trips (4 one-way) — MORNING PEAK
    - 10 AM-3 PM: 1 round trip (2 one-way) — MIDDAY
    - 4-7 PM: 2 round trips (4 one-way) — EVENING PEAK
    - 8-10 PM: 1 round trip (2 one-way) — LATE EVENING
    → 12 one-way trips per vehicle per day
    
    Scale:
    - 4 routes × 2 vehicles × 12 trips × 7 days = 672 one-way trips
    - With ~40 GPS readings per trip = ~26,880 total GPS points
    
    Parameters:
    - days: How many days to simulate (default 7)
    - vehicles_per_route: Vehicles assigned to each route (default 2)
    
    Returns:
    - pandas DataFrame with all GPS readings
    """
    all_trips = []
    route_ids = list(ROUTES.keys())
    
    # Start date: June 20, 2026
    start_date = datetime(2026, 6, 20, 0, 0, 0)
    
    for day_offset in range(days):
        current_date = start_date + timedelta(days=day_offset)
        
        for route_id in route_ids:
            route_info = ROUTES[route_id]
            route_waypoints = route_info["waypoints"]
            
            for vehicle_num in range(1, vehicles_per_route + 1):
                # Vehicle ID format: ROUTE_ID-V#
                vehicle_id = f"{route_id}-V{vehicle_num}"
                
                print(f"  Generating {vehicle_id} on {current_date.date()}...")
                
                # ===== Morning Peak (6-9 AM) =====
                trip1_start = current_date.replace(hour=6, minute=0)
                all_trips.extend(generate_trip(vehicle_id, route_id, route_waypoints, trip1_start, "forward"))
                
                trip2_start = trip1_start + timedelta(minutes=25)
                all_trips.extend(generate_trip(vehicle_id, route_id, route_waypoints, trip2_start, "backward"))
                
                trip3_start = trip2_start + timedelta(minutes=25)
                all_trips.extend(generate_trip(vehicle_id, route_id, route_waypoints, trip3_start, "forward"))
                
                # ===== Midday (10 AM - 3 PM) =====
                trip4_start = current_date.replace(hour=10, minute=30)
                all_trips.extend(generate_trip(vehicle_id, route_id, route_waypoints, trip4_start, "backward"))
                
                trip5_start = trip4_start + timedelta(minutes=25)
                all_trips.extend(generate_trip(vehicle_id, route_id, route_waypoints, trip5_start, "forward"))
                
                # ===== Evening Peak (4-7 PM) =====
                trip6_start = current_date.replace(hour=16, minute=0)
                all_trips.extend(generate_trip(vehicle_id, route_id, route_waypoints, trip6_start, "backward"))
                
                trip7_start = trip6_start + timedelta(minutes=25)
                all_trips.extend(generate_trip(vehicle_id, route_id, route_waypoints, trip7_start, "forward"))
                
                trip8_start = trip7_start + timedelta(minutes=25)
                all_trips.extend(generate_trip(vehicle_id, route_id, route_waypoints, trip8_start, "backward"))
                
                # ===== Late Evening (8-10 PM) =====
                trip9_start = current_date.replace(hour=20, minute=0)
                all_trips.extend(generate_trip(vehicle_id, route_id, route_waypoints, trip9_start, "forward"))
    
    # Convert to DataFrame and sort
    df = pd.DataFrame(all_trips)
    df = df.sort_values(by=["route_id", "vehicle_id", "timestamp"]).reset_index(drop=True)
    
    return df


# Run Generation and Save

if __name__ == "__main__":
    print("=" * 80)
    print("SmartRoute: Multi-Route GPS Dataset Generator (Version 2)")
    print("=" * 80)
    
    print("\nNOTE: CUBAO-DIVISORIA is in v1 (generate_simulated_gps.py)")
    print("This script generates the 4 NEW routes:\n")
    
    for route_id, route_info in ROUTES.items():
        num_stops = len(route_info["waypoints"])
        print(f"  ✓ {route_id}: {route_info['display_name']} ({num_stops} stops)")
    
    print("\n[1/4] Generating GPS traces for all 4 routes...")
    df = generate_full_dataset(days=7, vehicles_per_route=2)
    
    print(f"\n[2/4] Dataset Statistics:")
    print(f"  - Total GPS readings: {len(df):,}")
    print(f"  - Date range: {df['timestamp'].min()} to {df['timestamp'].max()}")
    print(f"  - Total routes: {df['route_id'].nunique()}")
    print(f"  - Total vehicles: {df['vehicle_id'].nunique()}")
    print(f"  - Avg readings per vehicle: {len(df) / df['vehicle_id'].nunique():.0f}")
    print(f"  - Speed stats: {df['speed_kmh'].mean():.1f} ± {df['speed_kmh'].std():.1f} km/h")
    
    print(f"\n  - Breakdown by route:")
    for route_id in df['route_id'].unique():
        route_df = df[df['route_id'] == route_id]
        vehicles = route_df['vehicle_id'].nunique()
        readings = len(route_df)
        avg_speed = route_df['speed_kmh'].mean()
        print(f"    {route_id:20s}: {vehicles} vehicles, {readings:6,} readings, avg {avg_speed:.1f} km/h")
    
    print(f"\n[3/4] Saving dataset...")
    output_path = "data/simulated_trips_multiroute.csv"
    df.to_csv(output_path, index=False)
    print(f"  ✓ Saved to: {output_path}")
    
    print(f"\n[4/4] Sample output (first 20 rows):")
    print(df.head(20).to_string(index=False))
    
    print("\n" + "=" * 80)
    print("✓ Complete! Ready for Backend/Frontend integration")
    print("=" * 80)