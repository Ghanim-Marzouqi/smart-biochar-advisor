"""
Generate sample training data without pandas dependency
Simple CSV generation for initial dataset
"""

import csv
import random
from pathlib import Path

# Seed for reproducibility
random.seed(42)


def calculate_pgi(soil_type, ph, ec, temp, humidity, biochar, npk):
    """Calculate Plant Growth Index (simplified)."""
    pgi = 50.0

    # pH factor
    if 6.0 <= ph <= 7.5:
        pgi += 15
    elif 5.5 <= ph < 6.0 or 7.5 < ph <= 8.0:
        pgi += 10
    else:
        pgi += 5

    # EC factor
    if ec < 1.5:
        pgi += 15
    elif ec < 2.5:
        pgi += 10
    elif ec < 4.0:
        pgi += 5

    # Temperature
    if 20 <= temp <= 30:
        pgi += 10
    elif 15 <= temp < 20 or 30 < temp <= 35:
        pgi += 5

    # Humidity
    if 40 <= humidity <= 70:
        pgi += 5

    # Biochar benefits
    pgi += min(biochar * 1.2, 15)
    if ec > 2.0:
        pgi += min(biochar * 0.5, 8)

    # NPK benefits
    pgi += min(npk * 0.8, 12)
    if npk > 15:
        pgi -= (npk - 15) * 0.5

    # Soil type modifiers
    if soil_type == 'sandy':
        pgi += biochar * 0.3
        if biochar + npk < 12:
            pgi -= 5
    elif soil_type == 'clay':
        pgi += 5
        if humidity > 70:
            pgi -= 5
    else:
        pgi += 3

    # Balance bonus
    ratio = biochar / (npk + 0.1)
    if 0.5 <= ratio <= 1.5:
        pgi += 5

    # Add variation
    pgi += random.gauss(0, 2)

    return max(0, min(100, round(pgi, 2)))


def generate_samples(n=500):
    """Generate training samples."""
    samples = []
    soil_types = ['sandy', 'clay', 'mixed']
    formulations = [(5, 5), (10, 10), (10, 15), (15, 10)]

    for i in range(n):
        soil_type = random.choice(soil_types)
        ph = round(random.uniform(4.5, 9.0), 2)
        ec = round(random.uniform(0.5, 6.0), 2)
        temp = round(random.uniform(10, 40), 1)
        humidity = round(random.uniform(20, 90), 1)

        # 30% use standard formulations, 70% random
        if random.random() < 0.3:
            biochar, npk = random.choice(formulations)
        else:
            biochar = round(random.uniform(0, 20), 2)
            npk = round(random.uniform(0, 20), 2)

        pgi = calculate_pgi(soil_type, ph, ec, temp, humidity, biochar, npk)
        yield_inc = round((pgi - 50) * 0.8, 2)
        retention = round(min(100, 50 + biochar * 2 + npk * 0.5), 2)

        samples.append({
            'soil_type': soil_type,
            'ph': ph,
            'ec': ec,
            'temperature': temp,
            'humidity': humidity,
            'biochar_g': biochar,
            'npk_g': npk,
            'plant_growth_index': pgi,
            'yield_increase_pct': yield_inc,
            'nutrient_retention_score': retention
        })

    return samples


def main():
    """Generate and save training data."""
    data_dir = Path(__file__).parent / "samples"
    data_dir.mkdir(exist_ok=True)

    print("🌱 Generating training dataset...")
    samples = generate_samples(500)

    # Write CSV
    output_file = data_dir / "training_data.csv"
    fieldnames = ['soil_type', 'ph', 'ec', 'temperature', 'humidity',
                  'biochar_g', 'npk_g', 'plant_growth_index',
                  'yield_increase_pct', 'nutrient_retention_score']

    with open(output_file, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(samples)

    print(f"✅ Training data saved: {output_file}")
    print(f"   - Samples: {len(samples)}")
    print(f"   - Features: {len(fieldnames)}")

    # Show statistics
    pgi_values = [s['plant_growth_index'] for s in samples]
    print(f"\n📊 PGI Statistics:")
    print(f"   - Min: {min(pgi_values):.2f}")
    print(f"   - Max: {max(pgi_values):.2f}")
    print(f"   - Mean: {sum(pgi_values)/len(pgi_values):.2f}")

    # Show sample
    print(f"\n📋 Sample Data (first 3 rows):")
    for i, sample in enumerate(samples[:3], 1):
        print(f"   {i}. {sample['soil_type']}, pH:{sample['ph']}, "
              f"Biochar:{sample['biochar_g']}g, NPK:{sample['npk_g']}g → "
              f"PGI:{sample['plant_growth_index']}")


if __name__ == "__main__":
    main()
