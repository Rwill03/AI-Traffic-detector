"""
Genereer 7 dagen realistische verkeersdata met spitsuren.
Dit script vult de database met testdata voor het trainen van het model.
"""
import os
import random
from datetime import datetime, timedelta
import argparse
try:
    from sqlalchemy import create_engine, text
    HAS_SQLALCHEMY = True
except ImportError:
    HAS_SQLALCHEMY = False
import json
from urllib import request, error

# API endpoint
API_URL = "http://localhost:8001/api/v1/observation"
DEFAULT_DB_URL = os.getenv(
    "DATABASE_URL",
    "postgresql+psycopg2://traffic_user:supersecretpassword@db/traffic_db",
)


def get_traffic_count(hour: int, day_of_week: int) -> dict:
    """
    Genereer realistische traffic counts op basis van uur en dag.
    Elk dag type heeft eigen patroon.
    
    Args:
        hour: Uur van de dag (0-23)
        day_of_week: Dag van de week (0=maandag, 6=zondag)
        
    Returns:
        Dict met voertuig counts
    """
    is_weekend = day_of_week >= 5
    
    # Verschillende patronen per dag type
    if is_weekend:
        # Weekend: minder verkeer, geen echte spits
        if 7 <= hour <= 9:
            car = random.randint(20, 40)
        elif 16 <= hour <= 18:
            car = random.randint(25, 45)
        elif 10 <= hour <= 15:
            car = random.randint(20, 35)
        elif 19 <= hour <= 22:
            car = random.randint(10, 20)
        elif 6 <= hour:
            car = random.randint(5, 15)
        else:
            car = random.randint(1, 6)
    else:
        # Weekdag: duidelijke spitsuren
        # Vrijdag iets drukker in de middag (mensen gaan eerder weg)
        if day_of_week == 4:  # Vrijdag
            friday_boost = 1.1
        else:
            friday_boost = 1.0
        
        if 7 <= hour <= 9:
            # Ochtendspits weekdag
            car = random.randint(90, 150)
        elif 16 <= hour <= 18:
            # Avondspits weekdag (vrijdag iets drukker)
            base = random.randint(95, 155)
            car = int(base * friday_boost)
        elif 10 <= hour <= 15:
            # Middag weekdag
            car = random.randint(30, 60)
        elif 19 <= hour <= 22:
            # Avond weekdag
            car = random.randint(15, 30)
        elif 6 <= hour:
            # Vroege ochtend
            car = random.randint(10, 25)
        else:
            # Nacht (0-5)
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
    """Stuur een observatie naar de API zonder externe dependencies."""
    data = json.dumps(observation).encode("utf-8")
    req = request.Request(
        api_url,
        data=data,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with request.urlopen(req, timeout=10) as resp:
            return resp.status == 200
    except error.HTTPError as e:
        print(f"HTTP error: {e.code} {e.reason}")
        return False
    except error.URLError as e:
        print(f"Connection error: {e.reason}")
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


def clear_database(db_url: str = DEFAULT_DB_URL) -> bool:
    """Leeg de traffic_samples tabel."""
    if not HAS_SQLALCHEMY:
        print("SQLAlchemy not available, skipping database clear")
        return False
    try:
        engine = create_engine(db_url, future=True)
        with engine.begin() as conn:
            conn.execute(text("TRUNCATE TABLE traffic_samples RESTART IDENTITY CASCADE"))
        print(f"Database geleegd via {db_url}")
        return True
    except Exception as e:
        print(f"Kon database niet legen: {e}")
        return False


def main():
    parser = argparse.ArgumentParser(description='Generate realistic traffic data')
    parser.add_argument('--days', type=int, default=30, help='Number of days to generate')
    parser.add_argument('--start-date', type=str, help='Start date (YYYY-MM-DD)')
    parser.add_argument('--api-url', type=str, default=API_URL, help='API URL')
    parser.add_argument('--db-url', type=str, default=None, help='Database URL voor reset')
    parser.add_argument('--reset-db', action='store_true', help='Leeg de traffic_samples tabel voor genereren')
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
        db_url = args.db_url or DEFAULT_DB_URL
        if args.reset_db:
            clear_database(db_url)
        generate_week_data(start_date, args.days, args.api_url)


if __name__ == "__main__":
    main()
