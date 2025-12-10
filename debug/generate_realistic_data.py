"""
Genereer 7 dagen realistische verkeersdata met spitsuren.
Dit script vult de database met testdata voor het trainen van het model.
"""
import random
from datetime import datetime, timedelta
import requests
import argparse

# API endpoint
API_URL = "http://localhost:8001/api/v1/observation"


def get_traffic_count(hour: int, day_of_week: int) -> dict:
    """
    Genereer realistische traffic counts op basis van uur en dag.
    
    Args:
        hour: Uur van de dag (0-23)
        day_of_week: Dag van de week (0=maandag, 6=zondag)
        
    Returns:
        Dict met voertuig counts
    """
    is_weekend = day_of_week >= 5
    
    # Base multiplier voor weekend (minder verkeer)
    weekend_factor = 0.6 if is_weekend else 1.0
    
    # Spitsuur patronen
    if 7 <= hour <= 9:
        # Ochtendspits
        if is_weekend:
            car = random.randint(15, 30)
        else:
            car = random.randint(50, 90)
    elif 16 <= hour <= 18:
        # Avondspits
        if is_weekend:
            car = random.randint(20, 35)
        else:
            car = random.randint(55, 95)
    elif 10 <= hour <= 15:
        # Overdag
        car = random.randint(int(20 * weekend_factor), int(45 * weekend_factor))
    elif 19 <= hour <= 22:
        # Avond
        car = random.randint(int(15 * weekend_factor), int(30 * weekend_factor))
    elif 6 <= hour <= 7:
        # Vroege ochtend
        car = random.randint(int(10 * weekend_factor), int(25 * weekend_factor))
    else:
        # Nacht (23-5)
        car = random.randint(1, 8)
    
    # Andere voertuigen (relatief aan auto's)
    truck = random.randint(0, max(1, car // 10)) if not is_weekend else random.randint(0, 2)
    bus = random.randint(0, 3) if 6 <= hour <= 22 else 0
    motorcycle = random.randint(0, max(1, car // 15))
    bicycle = random.randint(0, max(1, car // 8)) if 6 <= hour <= 20 else 0
    
    total = car + truck + bus + motorcycle + bicycle
    
    return {
        "car": car,
        "truck": truck,
        "bus": bus,
        "motorcycle": motorcycle,
        "bicycle": bicycle,
        "total_vehicles": total
    }


def generate_day_data(date: datetime, camera_id: str = "cam_A") -> list:
    """
    Genereer data voor één dag.
    
    Args:
        date: De datum
        camera_id: Camera ID
        
    Returns:
        List van observaties
    """
    day_of_week = date.weekday()
    observations = []
    
    for hour in range(24):
        # Meerdere observaties per uur voor meer realistische data
        samples_per_hour = random.randint(3, 6)
        
        for i in range(samples_per_hour):
            ts = date.replace(hour=hour, minute=random.randint(0, 59), second=random.randint(0, 59))
            
            counts = get_traffic_count(hour, day_of_week)
            # Variatie per sample binnen het uur
            variation = random.uniform(0.7, 1.3)
            
            observation = {
                "ts": ts.isoformat(),
                "camera_id": camera_id,
                "total_vehicles": int(counts["total_vehicles"] * variation / samples_per_hour),
                "car": int(counts["car"] * variation / samples_per_hour),
                "truck": int(counts["truck"] * variation / samples_per_hour),
                "bus": int(counts["bus"] * variation / samples_per_hour),
                "motorcycle": int(counts["motorcycle"] * variation / samples_per_hour),
                "bicycle": int(counts["bicycle"] * variation / samples_per_hour),
            }
            
            observations.append(observation)
    
    return observations


def send_observation(observation: dict, api_url: str) -> bool:
    """Stuur een observatie naar de API."""
    try:
        response = requests.post(api_url, json=observation, timeout=10)
        return response.status_code == 200
    except requests.exceptions.RequestException as e:
        print(f"Error sending observation: {e}")
        return False


def generate_week_data(start_date: datetime = None, days: int = 7, api_url: str = None):
    """
    Genereer data voor meerdere dagen en stuur naar de API.
    
    Args:
        start_date: Startdatum (default: 7 dagen geleden)
        days: Aantal dagen
        api_url: API URL
    """
    if start_date is None:
        start_date = datetime.now() - timedelta(days=days - 1)
        start_date = start_date.replace(hour=0, minute=0, second=0, microsecond=0)
    
    if api_url is None:
        api_url = API_URL
    
    total_observations = 0
    successful = 0
    
    print(f"Generating {days} days of traffic data...")
    print(f"Start date: {start_date.strftime('%Y-%m-%d')}")
    print(f"End date: {(start_date + timedelta(days=days-1)).strftime('%Y-%m-%d')}")
    print(f"API URL: {api_url}")
    print("-" * 50)
    
    for day_offset in range(days):
        current_date = start_date + timedelta(days=day_offset)
        day_name = current_date.strftime('%A')
        
        observations = generate_day_data(current_date)
        
        print(f"\n{current_date.strftime('%Y-%m-%d')} ({day_name}): {len(observations)} observations")
        
        day_success = 0
        for obs in observations:
            if send_observation(obs, api_url):
                day_success += 1
                successful += 1
            total_observations += 1
        
        print(f"  ✓ Sent {day_success}/{len(observations)} successfully")
    
    print("\n" + "=" * 50)
    print(f"Total: {successful}/{total_observations} observations sent successfully")
    
    return successful, total_observations


def main():
    parser = argparse.ArgumentParser(description='Generate realistic traffic data')
    parser.add_argument('--days', type=int, default=7, help='Number of days to generate')
    parser.add_argument('--start-date', type=str, help='Start date (YYYY-MM-DD)')
    parser.add_argument('--api-url', type=str, default=API_URL, help='API URL')
    parser.add_argument('--dry-run', action='store_true', help='Print data without sending')
    
    args = parser.parse_args()
    
    start_date = None
    if args.start_date:
        start_date = datetime.strptime(args.start_date, '%Y-%m-%d')
    
    if args.dry_run:
        # Toon alleen wat er gegenereerd zou worden
        if start_date is None:
            start_date = datetime.now() - timedelta(days=args.days - 1)
            start_date = start_date.replace(hour=0, minute=0, second=0, microsecond=0)
        
        print("DRY RUN - Data would be generated:")
        for day_offset in range(args.days):
            current_date = start_date + timedelta(days=day_offset)
            observations = generate_day_data(current_date)
            
            print(f"\n{current_date.strftime('%Y-%m-%d')} ({current_date.strftime('%A')})")
            
            # Toon hourly totals
            hourly = {}
            for obs in observations:
                hour = datetime.fromisoformat(obs['ts']).hour
                if hour not in hourly:
                    hourly[hour] = 0
                hourly[hour] += obs['car']
            
            for hour in sorted(hourly.keys()):
                bar = "█" * (hourly[hour] // 5)
                print(f"  {hour:02d}:00 | {hourly[hour]:3d} cars | {bar}")
    else:
        generate_week_data(start_date, args.days, args.api_url)


if __name__ == "__main__":
    main()
