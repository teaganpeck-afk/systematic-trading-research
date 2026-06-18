from pathlib import Path
import zipfile

import requests


BASE_URL = "https://data.binance.vision/data/futures/um/monthly/klines/BTCUSDT"
SYMBOL = "BTCUSDT"
INTERVALS = ["5m", "1d"]
START_YEAR = 2020
START_MONTH = 1
END_YEAR = 2022
END_MONTH = 12

PROJECT_ROOT = Path(__file__).resolve().parent
ZIP_ROOT = PROJECT_ROOT / "data" / "raw" / "binance" / "zips"
CSV_ROOT = PROJECT_ROOT / "data" / "raw"
KEEP_ZIPS = False


def generate_months():
    """Generate year/month pairs from January 2020 through December 2022."""
    months = []

    year = START_YEAR
    month = START_MONTH

    while (year, month) <= (END_YEAR, END_MONTH):
        months.append((year, month))

        month += 1
        if month == 13:
            month = 1
            year += 1

    return months


def download_file(url, destination):
    """Download one ZIP file unless it already exists locally."""
    if destination.exists():
        print(f"Skipping existing file {destination}")
        return "skipped"

    destination.parent.mkdir(parents=True, exist_ok=True)
    temp_destination = destination.with_suffix(destination.suffix + ".part")

    try:
        response = requests.get(url, stream=True, timeout=30)

        if response.status_code == 404:
            print(f"Missing remote file {url}")
            return "failed"

        if response.status_code != 200:
            print(f"Failed {url}: HTTP {response.status_code}")
            return "failed"

        with temp_destination.open("wb") as file:
            for chunk in response.iter_content(chunk_size=1024 * 1024):
                if chunk:
                    file.write(chunk)

        temp_destination.replace(destination)
        print(f"Downloaded {destination.name}")
        return "downloaded"

    except requests.RequestException as error:
        print(f"Failed {url}: {error}")
        return "failed"
    finally:
        if temp_destination.exists() and not destination.exists():
            temp_destination.unlink()


def extract_zip(zip_path, csv_directory):
    """Extract the expected CSV from a downloaded ZIP unless it already exists."""
    csv_directory.mkdir(parents=True, exist_ok=True)
    expected_csv = csv_directory / f"{zip_path.stem}.csv"

    if expected_csv.exists():
        print(f"Skipping existing file {expected_csv}")
        return "skipped"

    if not zip_path.exists():
        print(f"Cannot extract missing ZIP {zip_path}")
        return "failed"

    try:
        with zipfile.ZipFile(zip_path) as zip_file:
            csv_members = [
                member
                for member in zip_file.namelist()
                if member.endswith(".csv") and not member.endswith("/")
            ]

            if not csv_members:
                print(f"No CSV found in {zip_path.name}")
                return "failed"

            member_name = csv_members[0]
            with zip_file.open(member_name) as source, expected_csv.open("wb") as target:
                target.write(source.read())

        print(f"Extracted {expected_csv.name}")
        return "extracted"

    except zipfile.BadZipFile:
        print(f"Failed to extract {zip_path.name}: bad ZIP file")
        return "failed"


def process_interval(interval):
    """Download and extract all monthly ZIP files for one kline interval."""
    zip_directory = ZIP_ROOT / interval
    csv_directory = CSV_ROOT

    summary = {
        "downloaded": 0,
        "skipped": 0,
        "extracted": 0,
        "failures": 0,
    }

    for year, month in generate_months():
        file_name = f"{SYMBOL}-{interval}-{year}-{month:02d}"
        zip_name = f"{file_name}.zip"
        url = f"{BASE_URL}/{interval}/{zip_name}"
        zip_path = zip_directory / zip_name
        csv_path = csv_directory / f"{file_name}.csv"

        if csv_path.exists():
            print(f"Skipping existing file {csv_path}")
            summary["skipped"] += 1
            continue

        download_status = download_file(url, zip_path)
        if download_status == "downloaded":
            summary["downloaded"] += 1
        elif download_status == "skipped":
            summary["skipped"] += 1
        else:
            summary["failures"] += 1
            continue

        extract_status = extract_zip(zip_path, csv_directory)
        if extract_status == "extracted":
            summary["extracted"] += 1
            if not KEEP_ZIPS and zip_path.exists():
                zip_path.unlink()
                print(f"Deleted {zip_path.name}")
        elif extract_status == "failed":
            summary["failures"] += 1

    return summary


def main():
    """Download and extract BTCUSDT 5m and 1d monthly futures klines."""
    total_summary = {
        "downloaded": 0,
        "skipped": 0,
        "extracted": 0,
        "failures": 0,
    }

    for interval in INTERVALS:
        print(f"\nProcessing {SYMBOL} {interval} monthly klines")
        print("-" * 45)

        interval_summary = process_interval(interval)
        for key, value in interval_summary.items():
            total_summary[key] += value

    print("\nDownload Summary")
    print("================")
    print(f"ZIP files downloaded: {total_summary['downloaded']}")
    print(f"ZIP files skipped: {total_summary['skipped']}")
    print(f"CSV files extracted: {total_summary['extracted']}")
    print(f"Failures: {total_summary['failures']}")


if __name__ == "__main__":
    main()
