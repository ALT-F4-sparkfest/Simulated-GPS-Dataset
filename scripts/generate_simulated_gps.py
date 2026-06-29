import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import json
import math

# Fefining the routes
# 15 major waypoints along the cubao-divisoria route, manually extracted from google maps for realism
# Format (lat, long, stop_name, is_major_stop)

ROUTE_WAYPOINTS = [
    (14.5808, 121.0885, "Araneta Center, Cubao", True),          # Start (Cubao terminal)
    (14.5795, 121.0862, "Cubao Corner", False),
    (14.5780, 121.0820, "SM City Cubao", False),
    (14.5760, 121.0750, "Aurora Boulevard", True),
    (14.5700, 121.0620, "Dapitan-Legarda", False),
    (14.5680, 121.0580, "Legarda", False),
    (14.5650, 121.0540, "Recto Avenue", True),
    (14.5620, 121.0500, "Raon", False),
    (14.5600, 121.0450, "Quinta Market", False),
    (14.5580, 121.0380, "San Fernando", False),
    (14.5560, 121.0300, "Bambang", False),
    (14.5540, 121.0200, "Antiguitone", False),
    (14.5520, 121.0100, "Recoletos", False),
    (14.5500, 121.0000, "Escolta", False),
    (14.5480, 120.9950, "Tutuban Center, Divisoria", True),      # End (Divisoria terminal)
]
STOPS = [wp[:3] for wp in ROUTE_WAYPOINTS]  # (lat, lng, name)
MAJOR_STOPS = [wp[2] for wp in ROUTE_WAYPOINTS if wp[3]]  # Names of major stops only

# Helper Functions for Realistic Data Generation

def haversine_distance(lat1, lon1, lat2, lon2):
    """
    Calculate straight-line distance between two lat/lon points in kilometers.
    Used to determine how far a jeepney needs to travel to reach the next stop.
    """
    R = 6371  # Earth radius in km
    dlat = math.radians(lat2 - lat1)
    dlon = math.radians(lon2 - lon1)
    a = (math.sin(dlat / 2) ** 2 +
         math.cos(math.radians(lat1)) * math.cos(math.radians(lat2)) * math.sin(dlon / 2) ** 2)
    c = 2 * math.asin(math.sqrt(a))
    return R * c


def interpolate_points(lat1, lon1, lat2, lon2, num_points):
    """
    Generate evenly-spaced points between two coordinates.
    This smooths out the journey—instead of teleporting between waypoints,
    created a realistic path with many intermediate GPS readings. 
    """
    lats = np.linspace(lat1, lat2, num_points)
    lons = np.linspace(lon1, lon2, num_points)
    return list(zip(lats, lons))


def get_traffic_multiplier(hour_of_day):
    """
    Return a traffic congestion multiplier based on the hour.
    Higher multiplier = slower travel (e.g., rush hour takes 80% longer).
    
    Used to adjust the ETA heuristic and simulate realistic trip times.
    """
    if 5 <= hour_of_day < 7:
        return 1.2  # Early morning mild congestion
    elif 7 <= hour_of_day < 9:
        return 1.8  # HEAVY morning rush (7-9 AM)
    elif 9 <= hour_of_day < 16:
        return 1.0  # Midday. normal flow
    elif 16 <= hour_of_day < 19:
        return 1.7  # HEAVY evening rush (4-7 PM)
    elif 19 <= hour_of_day < 22:
        return 1.1  # Evening, lighter
    else:
        return 0.9  # Night (12 AM - 5 AM), very light


def get_base_speed_at_location(stop_index, total_stops):
    """
    Return a realistic base speed (km/h) at a given stop.
    - At terminals (stop_index 0 or end): speed is 0 (picking up/dropping passengers)
    - At major stops: lower speed (stopping to pick up passengers)
    - On open road: higher speed (20-40 km/h typical for Manila jeepneys)
    """
    if stop_index == 0 or stop_index == total_stops - 1:
        return 0  # Terminal: completely stopped
    elif ROUTE_WAYPOINTS[stop_index][3]:  # If it's a major stop
        return 5  # Major stop: very slow
    else:
        return 25  # Open road: typical jeepney speed


def generate_trip(vehicle_id, start_time, direction="forward"):
    """
    Generate one one-way trip (Cubao → Divisoria or vice versa).
    
    Returns a list of dicts, each representing one GPS reading (~30 seconds apart).
    
    Parameters:
    - vehicle_id: e.g., "JEEP-01"
    - start_time: datetime object when trip starts
    - direction: "forward" (Cubao→Divisoria) or "backward" (Divisoria→Cubao)
    """
    trip_data = []
    
    # Determine route order based on direction
    if direction == "forward":
        waypoint_sequence = ROUTE_WAYPOINTS
    else:
        waypoint_sequence = list(reversed(ROUTE_WAYPOINTS))
    
    current_time = start_time
    traffic_mult = get_traffic_multiplier(start_time.hour)
    
    # Travel between consecutive waypoints
    for i in range(len(waypoint_sequence) - 1):
        lat1, lon1, stop1_name, _ = waypoint_sequence[i]
        lat2, lon2, stop2_name, _ = waypoint_sequence[i + 1]
        
        # Calculate distance between this waypoint and the next
        distance_km = haversine_distance(lat1, lon1, lat2, lon2)
        
        # Get base speed and apply traffic multiplier
        base_speed = get_base_speed_at_location(i, len(waypoint_sequence))
        adjusted_speed = max(0, base_speed / traffic_mult)  # Traffic slows us down
        
        # Calculate time to travel this segment
        if adjusted_speed > 0:
            travel_time_minutes = (distance_km / adjusted_speed) * 60
        else:
            travel_time_minutes = 2  # 2 minutes stopped at terminal
        
        # Generate intermediate points along this segment (one every 30 seconds)
        num_intermediate_points = max(2, int(travel_time_minutes * 2))  # 2 points per minute
        intermediate_coords = interpolate_points(lat1, lon1, lat2, lon2, num_intermediate_points)
        
        # Create GPS readings for each intermediate point
        for j, (lat, lon) in enumerate(intermediate_coords):
            # Speed varies: low at start/end of segment, higher in middle
            if j < len(intermediate_coords) // 3 or j > 2 * len(intermediate_coords) // 3:
                speed = adjusted_speed * 0.7  # Slowing down near stops
            else:
                speed = adjusted_speed  # Full speed in middle
            
            trip_data.append({
                "vehicle_id": vehicle_id,
                "timestamp": current_time.isoformat() + "Z",
                "latitude": round(lat, 6),
                "longitude": round(lon, 6),
                "speed_kmh": round(max(0, speed + np.random.normal(0, 1)), 2),  # Add small random noise
                "stop_name": stop1_name if j == 0 else stop2_name,
            })
            
            current_time += timedelta(seconds=30)  # Next GPS reading is 30 seconds later
    
    return trip_data


# Generate Full 7-Day Dataset

def generate_full_dataset(days=7, vehicles=["JEEP-01", "JEEP-02"]):
    """
    Generate a complete week of trips for all vehicles.
    
    Each vehicle makes trips throughout the day:
    - Morning peak (6-9 AM): 2 round trips
    - Midday (10 AM-3 PM): 1 round trip
    - Evening peak (4-7 PM): 2 round trips
    - Late evening (8 PM-10 PM): 1 round trip
    
    Returns: DataFrame with all trips combined
    """
    all_trips = []
    
    # Start date: June 20, 2026 (Friday)
    start_date = datetime(2026, 6, 20, 0, 0, 0)
    
    for day_offset in range(days):
        current_date = start_date + timedelta(days=day_offset)
        
        for vehicle_id in vehicles:
            print(f"Generating trips for {vehicle_id} on {current_date.date()}...")
            
            # Morning peak (6:00 AM - 9:00 AM)
            # Trip 1: Cubao → Divisoria
            trip1_start = current_date.replace(hour=6, minute=0)
            all_trips.extend(generate_trip(vehicle_id, trip1_start, direction="forward"))
            
            # Trip 2: Divisoria → Cubao
            trip2_start = trip1_start + timedelta(minutes=25)
            all_trips.extend(generate_trip(vehicle_id, trip2_start, direction="backward"))
            
            # Trip 3: Cubao → Divisoria
            trip3_start = trip2_start + timedelta(minutes=25)
            all_trips.extend(generate_trip(vehicle_id, trip3_start, direction="forward"))
            
            # Midday (10:00 AM - 3:00 PM)
            # Trip 4: Divisoria → Cubao
            trip4_start = current_date.replace(hour=10, minute=30)
            all_trips.extend(generate_trip(vehicle_id, trip4_start, direction="backward"))
            
            # Trip 5: Cubao → Divisoria
            trip5_start = trip4_start + timedelta(minutes=25)
            all_trips.extend(generate_trip(vehicle_id, trip5_start, direction="forward"))
            
            # Evening peak (4:00 PM - 7:00 PM)
            # Trip 6: Divisoria → Cubao
            trip6_start = current_date.replace(hour=16, minute=0)
            all_trips.extend(generate_trip(vehicle_id, trip6_start, direction="backward"))
            
            # Trip 7: Cubao → Divisoria
            trip7_start = trip6_start + timedelta(minutes=25)
            all_trips.extend(generate_trip(vehicle_id, trip7_start, direction="forward"))
            
            # Late evening (8:00 PM - 10:00 PM)
            # Trip 8: Divisoria → Cubao
            trip8_start = current_date.replace(hour=20, minute=0)
            all_trips.extend(generate_trip(vehicle_id, trip8_start, direction="backward"))
    
    # Convert to DataFrame
    df = pd.DataFrame(all_trips)
    
    # Sort by vehicle and timestamp for clarity
    df = df.sort_values(by=["vehicle_id", "timestamp"]).reset_index(drop=True)
    
    return df


# Run Generation and Save

if __name__ == "__main__":
    print("=" * 80)
    print("SmartRoute: Generating 7-Day Simulated GPS Dataset")
    print("=" * 80)
    
    # Generate the dataset
    print("\n[1/3] Generating GPS traces...")
    df = generate_full_dataset(days=7, vehicles=["JEEP-01", "JEEP-02"])
    
    # Basic stats before saving
    print(f"\n[2/3] Dataset Statistics:")
    print(f"  - Total GPS readings: {len(df):,}")
    print(f"  - Date range: {df['timestamp'].min()} to {df['timestamp'].max()}")
    print(f"  - Vehicles: {df['vehicle_id'].unique().tolist()}")
    print(f"  - Avg speed across all readings: {df['speed_kmh'].mean():.1f} km/h")
    print(f"  - Speed range: {df['speed_kmh'].min():.1f} to {df['speed_kmh'].max():.1f} km/h")
    
    # Save to CSV
    output_path = "data/simulated_trips.csv"
    df.to_csv(output_path, index=False)
    print(f"\n[3/3] ✓ Dataset saved to: {output_path}")
    print("\nFirst 10 rows:")
    print(df.head(10))