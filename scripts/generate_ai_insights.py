"""
Purpose:
- Read simulated GPS CSV data
- Aggregate into dashboard-ready JSON files
- Create 3 visualization datasets:
  1. Travel Time by Hour (time-series chart)
  2. Demand Hotspots (map markers)
  3. Fleet Performance Stats (summary cards)

Output:
- data/ai_insights/hourly_travel_time.json
- data/ai_insights/demand_hotspots.json
- data/ai_insights/fleet_stats.json

Why aggregated JSON?
- Frontend Lead can plug directly into charts (Recharts, Leaflet, etc.)
- No backend computation needed for dashboard
- Pure data layer → visualization layer
- Shows the "analytics pipeline" judges want to see

How to run:
  python scripts/generate_ai_insights.py

Expected output:
  ✓ 3 JSON files in data/ai_insights/
  ✓ Summary statistics printed to console
"""

import pandas as pd
import numpy as np
import json
import math
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple


# ==============================================================================
# PART 1: Helper Functions
# ==============================================================================

def haversine_distance(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """
    Calculate distance between two GPS points in kilometers.
    
    Used to determine if a vehicle is at a major stop (within 100m).
    """
    R = 6371
    dlat = math.radians(lat2 - lat1)
    dlon = math.radians(lon2 - lon1)
    a = (math.sin(dlat / 2) ** 2 +
         math.cos(math.radians(lat1)) * math.cos(math.radians(lat2)) * math.sin(dlon / 2) ** 2)
    c = 2 * math.asin(math.sqrt(a))
    return R * c


# Define major stops with their coordinates
# These are the terminal and major hubs where we expect high passenger volume
MAJOR_STOPS = {
    # Cubao-Divisoria Route
    "Araneta Center, Cubao": (14.5808, 121.0885, "CUBAO-DIVISORIA"),
    "Recto Avenue": (14.5650, 121.0540, "CUBAO-DIVISORIA"),
    "Tutuban Center, Divisoria": (14.5480, 120.9950, "CUBAO-DIVISORIA"),
    
    # Cubao-Makati Route
    "Ortigas Center": (14.5700, 121.1050, "CUBAO-MAKATI"),
    "Ayala Avenue": (14.5200, 121.1350, "CUBAO-MAKATI"),
    "Makati CBD": (14.5100, 121.1350, "CUBAO-MAKATI"),
    
    # Cubao-Marikina Route
    "Aurora Boulevard": (14.5760, 121.0750, "CUBAO-MARIKINA"),
    "Marikina City Hall": (14.5550, 121.0300, "CUBAO-MARIKINA"),
    "Marikina CBD": (14.5450, 121.0200, "CUBAO-MARIKINA"),
    
    # Cubao-Pasig Route
    "Pasig City Hall": (14.5950, 121.0950, "CUBAO-PASIG"),
    "Pasig CBD": (14.6000, 121.0980, "CUBAO-PASIG"),
    
    # Cubao-San Juan Route
    "San Juan Main St": (14.5850, 121.0700, "CUBAO-SANJUAN"),
    "San Juan CBD": (14.5900, 121.0600, "CUBAO-SANJUAN"),
}


def find_nearest_major_stop(lat: float, lon: float, within_km: float = 0.1) -> str:
    """
    Find the nearest major stop within a distance threshold.
    
    Used to label demand hotspots and group passengers.
    
    Args:
        lat, lon: Current position
        within_km: Only return stops within this distance (default 0.1 km = 100m)
    
    Returns:
        str: Name of nearest stop, or "Other" if none nearby
    """
    min_dist = float('inf')
    nearest = "Other"
    
    for stop_name, (stop_lat, stop_lon, _) in MAJOR_STOPS.items():
        dist = haversine_distance(lat, lon, stop_lat, stop_lon)
        if dist < min_dist and dist <= within_km:
            min_dist = dist
            nearest = stop_name
    
    return nearest


# ==============================================================================
# PART 2: Insight 1 - Travel Time by Hour
# ==============================================================================

def compute_hourly_travel_time(df: pd.DataFrame) -> Dict:
    """
    Calculate average trip duration for each hour of the day.
    
    Purpose:
    - Show commuters realistic ETAs based on time of day
    - Morning rush (7-9 AM) → longer trips
    - Midday (9 AM-4 PM) → shorter trips
    - Evening rush (4-7 PM) → longer trips
    
    Algorithm:
    1. Group data by route
    2. For each route, find trip start/end times (vehicle at terminal)
    3. Calculate trip duration
    4. Bucket by departure hour
    5. Aggregate statistics
    
    Returns:
        dict: {
            'hours': [0, 1, ..., 23],
            'avg_minutes': [...],
            'min_minutes': [...],
            'max_minutes': [...],
            'trip_count': [...]
        }
    """
    print("\n[1/3] Computing hourly travel time...")
    
    # Parse timestamps
    df['timestamp'] = pd.to_datetime(df['timestamp'], utc=True)
    df['hour'] = df['timestamp'].dt.hour
    df['route_id'] = df.get('route_id', 'CUBAO-DIVISORIA')  # Handle single-route CSVs
    
    # For each vehicle on each route, identify "trips"
    # A trip is inferred by detecting when vehicle is at a terminal
    trips = []
    
    for route_id in df['route_id'].unique():
        route_data = df[df['route_id'] == route_id].copy()
        
        for vehicle_id in route_data['vehicle_id'].unique():
            vehicle_data = route_data[route_data['vehicle_id'] == vehicle_id].sort_values('timestamp')
            
            if len(vehicle_data) < 2:
                continue
            
            # Detect trip boundaries: vehicle speed goes from 0 → moving → 0
            # Or we can use time gaps: if >30 min gap, assume new trip
            prev_time = None
            trip_start = None
            
            for idx, row in vehicle_data.iterrows():
                current_time = row['timestamp']
                speed = row['speed_kmh']
                
                # Start of trip: speed > 5 km/h (moving)
                if speed > 5 and trip_start is None:
                    trip_start = current_time
                
                # End of trip: speed < 1 km/h (stopped)
                elif speed < 1 and trip_start is not None:
                    trip_duration_minutes = (current_time - trip_start).total_seconds() / 60
                    
                    # Only count reasonable trips (5-60 minutes)
                    if 5 <= trip_duration_minutes <= 60:
                        trip_hour = trip_start.hour
                        trips.append({
                            'vehicle_id': vehicle_id,
                            'route_id': route_id,
                            'hour': trip_hour,
                            'duration_minutes': trip_duration_minutes,
                        })
                    
                    trip_start = None
                
                prev_time = current_time
    
    # Aggregate by hour
    trip_df = pd.DataFrame(trips)
    
    if len(trip_df) == 0:
        print("  Warning: No complete trips detected. Using raw segment durations.")
        # Fallback: use consecutive GPS reading deltas
        df = df.sort_values(['vehicle_id', 'timestamp'])
        df['time_delta'] = df.groupby('vehicle_id')['timestamp'].diff().dt.total_seconds() / 60
        df = df[df['time_delta'] > 0.5]  # >30 seconds
        trip_df = df[['hour', 'time_delta']].rename(columns={'time_delta': 'duration_minutes'})
    
    hourly_stats = trip_df.groupby('hour')['duration_minutes'].agg([
        ('avg', 'mean'),
        ('min', 'min'),
        ('max', 'max'),
        ('count', 'count'),
        ('std', 'std'),
    ]).reset_index()
    
    # Fill missing hours with interpolated values
    all_hours = pd.DataFrame({'hour': range(24)})
    hourly_stats = all_hours.merge(hourly_stats, on='hour', how='left')
    
    # Interpolate missing hours
    hourly_stats['avg'] = hourly_stats['avg'].interpolate()
    hourly_stats['avg'] = hourly_stats['avg'].fillna(hourly_stats['avg'].mean())
    hourly_stats['count'] = hourly_stats['count'].fillna(0)
    
    result = {
        'hours': list(range(24)),
        'avg_minutes': [round(x, 1) for x in hourly_stats['avg'].values],
        'min_minutes': [round(x, 1) if not pd.isna(x) else hourly_stats['avg'].mean() 
                       for x in hourly_stats['min'].values],
        'max_minutes': [round(x, 1) if not pd.isna(x) else hourly_stats['avg'].mean() 
                       for x in hourly_stats['max'].values],
        'trip_count': [int(x) for x in hourly_stats['count'].values],
    }
    
    print(f"  ✓ Computed travel times for 24 hours")
    print(f"  - Busiest hour: {result['hours'][np.argmax(result['avg_minutes'])]}:00 "
          f"({max(result['avg_minutes']):.1f} min avg)")
    print(f"  - Lightest hour: {result['hours'][np.argmin(result['avg_minutes'])]}:00 "
          f"({min(result['avg_minutes']):.1f} min avg)")
    
    return result


# ==============================================================================
# PART 3: Insight 2 - Demand Hotspots
# ==============================================================================

def compute_demand_hotspots(df: pd.DataFrame) -> List[Dict]:
    """
    Identify major stops and estimate waiting passengers per stop.
    
    Purpose:
    - Show LGUs where passenger demand is highest
    - Help cooperative prioritize which stops need more vehicles
    - Support route optimization decisions
    
    Algorithm:
    1. For each GPS reading, find nearest major stop (within 100m)
    2. Count how many unique vehicles visited this stop
    3. Estimate avg waiting time (time stopped at stop)
    4. Calculate "demand score" = vehicles × avg_wait_time
    
    Returns:
        list: [
            {
                'stop_name': str,
                'latitude': float,
                'longitude': float,
                'route_id': str,
                'vehicle_count': int,
                'avg_daily_waiting': int,
                'demand_score': float
            },
            ...
        ]
    """
    print("\n[2/3] Computing demand hotspots...")
    
    df['timestamp'] = pd.to_datetime(df['timestamp'], utc=True)
    df['route_id'] = df.get('route_id', 'CUBAO-DIVISORIA')
    
    # For each major stop, calculate statistics
    hotspots = []
    
    for stop_name, (stop_lat, stop_lon, route_id) in MAJOR_STOPS.items():
        # Find all GPS readings within 100m of this stop
        distances = df.apply(
            lambda row: haversine_distance(row['latitude'], row['longitude'], 
                                           stop_lat, stop_lon),
            axis=1
        )
        nearby = df[distances <= 0.1].copy()
        
        if len(nearby) == 0:
            continue
        
        # Count unique vehicles that visited this stop
        unique_vehicles = nearby['vehicle_id'].nunique()
        
        # Calculate time spent at stop (consecutive readings with speed < 5 km/h)
        wait_times = []
        for vehicle_id in nearby['vehicle_id'].unique():
            vehicle_readings = nearby[nearby['vehicle_id'] == vehicle_id].sort_values('timestamp')
            
            current_wait = 0
            for idx, row in vehicle_readings.iterrows():
                if row['speed_kmh'] < 5:
                    current_wait += 0.5  # Each GPS reading is ~30 seconds
                else:
                    if current_wait > 0:
                        wait_times.append(current_wait)
                    current_wait = 0
            if current_wait > 0:
                wait_times.append(current_wait)
        
        avg_wait_minutes = np.mean(wait_times) if wait_times else 2  # Default 2 min
        
        # Demand score: combines frequency and wait time
        # High score = many vehicles AND long waits = high demand
        demand_score = unique_vehicles * (avg_wait_minutes / 5)
        
        hotspots.append({
            'stop_name': stop_name,
            'latitude': round(stop_lat, 6),
            'longitude': round(stop_lon, 6),
            'route_id': route_id,
            'vehicle_count': int(unique_vehicles),
            'avg_wait_minutes': round(avg_wait_minutes, 1),
            'demand_score': round(demand_score, 2),
            'avg_daily_waiting': int(unique_vehicles * (avg_wait_minutes / 2)),  # Estimated daily
        })
    
    # Sort by demand score
    hotspots = sorted(hotspots, key=lambda x: x['demand_score'], reverse=True)
    
    print(f"  ✓ Identified {len(hotspots)} major stops")
    print(f"  - Busiest: {hotspots[0]['stop_name']} "
          f"(demand score: {hotspots[0]['demand_score']})")
    
    return hotspots


# ==============================================================================
# PART 4: Insight 3 - Fleet Performance Stats
# ==============================================================================

def compute_fleet_stats(df: pd.DataFrame) -> Dict:
    """
    Calculate overall fleet performance metrics.
    
    Purpose:
    - Show cooperative dispatcher fleet health at a glance
    - Support pitch narrative: "we're moving X km, Y trips per day"
    - Demonstrate data collection readiness
    
    Metrics:
    - Total trips completed
    - Total distance traveled
    - Average speed (typical commute speed)
    - On-time performance (% of trips within expected time)
    - Vehicle utilization (% of day actively moving)
    
    Returns:
        dict: {
            'total_trips': int,
            'total_km': float,
            'avg_speed': float,
            'on_time_percentage': int,
            'avg_utilization': float,
            'peak_hour': int,
            'avg_daily_revenue_potential': float,
        }
    """
    print("\n[3/3] Computing fleet statistics...")
    
    df['timestamp'] = pd.to_datetime(df['timestamp'], utc=True)
    
    # 1. Count trips (time between terminals)
    total_trips = 0
    trip_times = []
    
    for vehicle_id in df['vehicle_id'].unique():
        vehicle_df = df[df['vehicle_id'] == vehicle_id].sort_values('timestamp')
        
        # Detect terminal visits (stop_name contains "Center" or "CBD")
        terminals = vehicle_df[
            vehicle_df['stop_name'].str.contains('Center|CBD|Terminal', case=False, na=False)
        ]
        
        total_trips += len(terminals) // 2  # Each round trip = 2 terminal visits
        trip_times.extend(vehicle_df['timestamp'].diff().dt.total_seconds().dropna() / 60)
    
    # 2. Calculate distance (sum of consecutive readings)
    df['distance_from_prev'] = 0.0
    for vehicle_id in df['vehicle_id'].unique():
        vehicle_df = df[df['vehicle_id'] == vehicle_id].sort_values('timestamp')
        for i in range(1, len(vehicle_df)):
            dist = haversine_distance(
                vehicle_df.iloc[i-1]['latitude'], vehicle_df.iloc[i-1]['longitude'],
                vehicle_df.iloc[i]['latitude'], vehicle_df.iloc[i]['longitude']
            )
            df.loc[vehicle_df.index[i], 'distance_from_prev'] = dist
    
    total_km = df['distance_from_prev'].sum()
    
    # 3. Average speed (exclude stopped readings)
    moving = df[df['speed_kmh'] > 5]
    avg_speed = moving['speed_kmh'].mean() if len(moving) > 0 else 25
    
    # 4. On-time performance
    # Define "on-time" as within 10% of expected time
    expected_trip_time = df['timestamp'].diff().dt.total_seconds().median() / 60
    on_time_count = len(trip_times) - sum(1 for t in trip_times if t > expected_trip_time * 1.1)
    on_time_pct = int(100 * on_time_count / len(trip_times)) if len(trip_times) > 0 else 85
    
    # 5. Vehicle utilization
    # % of day that vehicle was moving (speed > 5 km/h)
    utilization = 100 * len(moving) / len(df) if len(df) > 0 else 50
    
    # 6. Peak hour
    df['hour'] = df['timestamp'].dt.hour
    peak_hour = df[df['speed_kmh'] > 10].groupby('hour').size().idxmax() if len(df) > 0 else 8
    
    # 7. Revenue potential
    # Assume: avg 15 passengers per trip, ₱15 per trip
    revenue_per_trip = 15 * 15  # 15 passengers × ₱15 per ride
    avg_daily_revenue = (total_trips / 7) * revenue_per_trip  # 7 days simulated
    
    result = {
        'total_trips': max(1, int(total_trips)),
        'total_km': round(total_km, 1),
        'avg_speed_kmh': round(avg_speed, 1),
        'on_time_percentage': min(100, on_time_pct),
        'avg_utilization_percent': round(utilization, 1),
        'peak_hour': int(peak_hour),
        'avg_daily_revenue_php': round(avg_daily_revenue, 2),
        'vehicles_active': int(df['vehicle_id'].nunique()),
        'days_simulated': 7,
    }
    
    print(f"  ✓ Fleet Statistics:")
    print(f"    - {result['total_trips']} trips across {result['vehicles_active']} vehicles")
    print(f"    - {result['total_km']} km total distance")
    print(f"    - {result['avg_speed_kmh']} km/h average speed")
    print(f"    - {result['on_time_percentage']}% on-time performance")
    print(f"    - Peak usage at {result['peak_hour']}:00")
    
    return result
    

# ==============================================================================
# PART 5: Main Function
# ==============================================================================

def generate_ai_insights(csv_path: str = None) -> None:
    """
    Main function: load CSV, compute insights, save JSON files.
    """
    print("=" * 80)
    print("AI Insights Dashboard Data Aggregator")
    print("=" * 80)
    
    print("\nLoading and combining CSV datasets...")
    
    try:
        # Load both CSV files and combine them
        df_single = pd.read_csv("data/simulated_trips.csv")
        df_multi = pd.read_csv("data/simulated_trips_multiroute.csv")
        df = pd.concat([df_single, df_multi], ignore_index=True)
        print(f"  ✓ Loaded combined dataset: {len(df):,} GPS readings")
    except FileNotFoundError as e:
        print(f"\n❌ Error: Missing a required CSV file - {e.filename}")
        print("   Expected both: data/simulated_trips.csv AND data/simulated_trips_multiroute.csv")
        print("   Run: python scripts/generate_simulated_gps.py")
        return

    print(f"  - Vehicles: {df['vehicle_id'].nunique()}")
    print(f"  - Date range: {df['timestamp'].min()} to {df['timestamp'].max()}")
    
    # Compute insights
    hourly_travel = compute_hourly_travel_time(df)
    demand_hotspots = compute_demand_hotspots(df)
    fleet_stats = compute_fleet_stats(df)
    
    # Create output directory
    output_dir = Path('data/ai_insights')
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save JSON files
    print("\n" + "=" * 80)
    print("Saving insights to JSON...")
    print("=" * 80)
    
    # 1. Hourly travel time
    with open(output_dir / 'hourly_travel_time.json', 'w') as f:
        json.dump(hourly_travel, f, indent=2)
    print(f"✓ {output_dir / 'hourly_travel_time.json'}")
    
    # 2. Demand hotspots
    with open(output_dir / 'demand_hotspots.json', 'w') as f:
        json.dump(demand_hotspots, f, indent=2)
    print(f"✓ {output_dir / 'demand_hotspots.json'} ({len(demand_hotspots)} stops)")
    
    # 3. Fleet stats
    with open(output_dir / 'fleet_stats.json', 'w') as f:
        json.dump(fleet_stats, f, indent=2)
    print(f"✓ {output_dir / 'fleet_stats.json'}")
    
    print("\n" + "=" * 80)
    print("✅ AI Insights generation complete!")
    print("=" * 80)
    print("\nFrontend Lead can now use these JSON files for:")
    print("  1. hourly_travel_time.json → Recharts line/bar chart")
    print("  2. demand_hotspots.json → Leaflet map markers")
    print("  3. fleet_stats.json → Summary cards/KPIs")


if __name__ == "__main__":
    generate_ai_insights()