import os
import requests
import zipfile

ZENODO_API = "https://zenodo.org/api/records/4010759"

BASE_DIR = r"D:\ISLR_DATA\INCLUDE"
ARCHIVE_DIR = os.path.join(BASE_DIR, "archives")
VIDEO_DIR = os.path.join(BASE_DIR, "videos")

os.makedirs(ARCHIVE_DIR, exist_ok=True)
os.makedirs(VIDEO_DIR, exist_ok=True)


def download_file(url, output_path):

    filename = os.path.basename(output_path)

    # If the file already exists, verify that it is actually a ZIP.
    if os.path.exists(output_path):

        try:
            with zipfile.ZipFile(output_path, "r") as test_zip:
                test_zip.testzip()

            print(
                f"\nAlready downloaded and valid: {filename}"
            )
            return

        except (zipfile.BadZipFile, OSError):

            print(
                f"\nInvalid/corrupt archive found: {filename}"
            )

            print("Deleting and downloading again...")

            os.remove(output_path)


    print(f"\nDownloading: {filename}")


    # Temporary file prevents a partially downloaded
    # archive from being mistaken for a completed one.
    temp_path = output_path + ".part"


    if os.path.exists(temp_path):
        os.remove(temp_path)


    with requests.get(
        url,
        stream=True,
        timeout=120
    ) as response:

        response.raise_for_status()

        total = int(
            response.headers.get(
                "content-length",
                0
            )
        )

        downloaded = 0


        with open(
            temp_path,
            "wb"
        ) as file:

            for chunk in response.iter_content(
                chunk_size=1024 * 1024
            ):

                if chunk:

                    file.write(chunk)

                    downloaded += len(chunk)


                    if total:

                        percent = (
                            downloaded * 100 / total
                        )

                        print(
                            f"\rProgress: {percent:6.2f}%",
                            end=""
                        )


    print("\nDownload complete.")


    # Verify the downloaded archive BEFORE
    # replacing the final file.
    try:

        with zipfile.ZipFile(
            temp_path,
            "r"
        ) as test_zip:

            bad_file = test_zip.testzip()

            if bad_file:

                raise zipfile.BadZipFile(
                    f"Corrupt file inside archive: {bad_file}"
                )


    except (zipfile.BadZipFile, OSError):

        if os.path.exists(temp_path):
            os.remove(temp_path)

        raise RuntimeError(
            f"Downloaded archive is invalid: {filename}"
        )


    # Only now move it to its final name.
    os.replace(
        temp_path,
        output_path
    )

    print(
        f"Verified: {filename}"
    )


def extract_archive(zip_path):

    filename = os.path.basename(zip_path)

    print(
        f"\nExtracting: {filename}"
    )


    try:

        with zipfile.ZipFile(
            zip_path,
            "r"
        ) as archive:

            bad_file = archive.testzip()

            if bad_file:

                raise zipfile.BadZipFile(
                    f"Corrupt file: {bad_file}"
                )

            archive.extractall(
                VIDEO_DIR
            )


    except zipfile.BadZipFile:

        print(
            f"ERROR: {filename} is corrupted."
        )

        print(
            "It will be redownloaded on the next run."
        )

        os.remove(zip_path)

        raise


    print(
        "Extraction complete."
    )


def main():

    print("Fetching Zenodo file list...")

    response = requests.get(
        ZENODO_API,
        timeout=60
    )

    response.raise_for_status()

    data = response.json()

    files = data["files"]

    zip_files = [
        f for f in files
        if f["key"].lower().endswith(".zip")
    ]

    print()
    print("=" * 60)
    print("INCLUDE DOWNLOAD")
    print("=" * 60)
    print(f"ZIP archives: {len(zip_files)}")
    print(f"Destination:  {BASE_DIR}")
    print("=" * 60)

    for index, file_info in enumerate(
        zip_files,
        start=1
    ):

        filename = file_info["key"]
        url = file_info["links"]["self"]

        archive_path = os.path.join(
            ARCHIVE_DIR,
            filename
        )

        print()
        print(
            f"[{index}/{len(zip_files)}] {filename}"
        )

        download_file(
            url,
            archive_path
        )

        # Extract immediately
        extract_archive(
            archive_path
        )

    print()
    print("=" * 60)
    print("ALL INCLUDE ARCHIVES PROCESSED")
    print("=" * 60)
    print(f"Videos: {VIDEO_DIR}")
    print(f"Archives: {ARCHIVE_DIR}")


if __name__ == "__main__":
    main()